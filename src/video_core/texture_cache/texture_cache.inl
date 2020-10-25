// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "video_core/texture_cache/formatter.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

template <class P>
TextureCache<P>::TextureCache(Runtime& runtime_, VideoCore::RasterizerInterface& rasterizer_,
                              Tegra::Engines::Maxwell3D& maxwell3d_,
                              Tegra::Engines::KeplerCompute& kepler_compute_,
                              Tegra::MemoryManager& gpu_memory_)
    : runtime{runtime_}, rasterizer{rasterizer_}, maxwell3d{maxwell3d_},
      kepler_compute{kepler_compute_}, gpu_memory{gpu_memory_} {
    // Make sure the first index is reserved for the null image view
    // This way the null image view becomes a compile time constant
    (void)slot_image_views.insert(runtime, NullImageParams{});
}

template <class P>
void TextureCache<P>::ImplicitDescriptorInvalidations() {
    if (!maxwell3d.dirty.flags[Dirty::Descriptors] && !has_deleted_images) {
        return;
    }
    maxwell3d.dirty.flags[Dirty::Descriptors] = false;

    InvalidateImageDescriptorTable();
    InvalidateSamplerDescriptorTable();
}

template <class P>
typename P::Sampler* TextureCache<P>::GetGraphicsSampler(size_t index) {
    return &slot_samplers[tables_3d.samplers[index]];
}

template <class P>
typename P::Sampler* TextureCache<P>::GetComputeSampler(size_t index) {
    return &slot_samplers[tables_compute.samplers[index]];
}

template <class P>
void TextureCache<P>::UpdateRenderTargets() {
    // TODO: Use dirty flags
    for (size_t index = 0; index < NUM_RT; ++index) {
        ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
        color_buffer_id = FindColorBuffer(index);
        if (color_buffer_id) {
            Image& image = slot_images[slot_image_views[color_buffer_id].image_id];
            image.flags |= ImageFlagBits::GpuModified;
            UpdateImageContents(image);
        }
    }
    render_targets.depth_buffer_id = FindDepthBuffer();
    if (render_targets.depth_buffer_id) {
        Image& image = slot_images[slot_image_views[render_targets.depth_buffer_id].image_id];
        image.flags |= ImageFlagBits::GpuModified;
        UpdateImageContents(image);
    }
    for (size_t index = 0; index < NUM_RT; ++index) {
        render_targets.draw_buffers[index] = maxwell3d.regs.rt_control.GetMap(index);
    }
    render_targets.size = Extent2D{
        maxwell3d.regs.render_area.width,
        maxwell3d.regs.render_area.height,
    };
    if (has_deleted_images) {
        InvalidateImageDescriptorTable();
    }
}

template <class P>
typename P::Framebuffer* TextureCache<P>::GetFramebuffer() {
    return GetFramebuffer(render_targets);
}

template <class P>
typename P::Framebuffer* TextureCache<P>::GetFramebuffer(const RenderTargets& key) {
    const auto [pair, is_new] = framebuffers.try_emplace(key);
    FramebufferId& framebuffer = pair->second;
    if (!is_new) {
        return &slot_framebuffers[framebuffer];
    }

    std::array<ImageView*, NUM_RT> color_buffers;
    std::ranges::transform(key.color_buffer_ids, color_buffers.begin(),
                           [this](ImageViewId id) { return id ? &slot_image_views[id] : nullptr; });
    ImageView* const depth_buffer =
        key.depth_buffer_id ? &slot_image_views[key.depth_buffer_id] : nullptr;
    framebuffer = slot_framebuffers.insert(runtime, slot_images, color_buffers, depth_buffer,
                                           key.draw_buffers, key.size);

    return &slot_framebuffers[framebuffer];
}

template <class P>
void TextureCache<P>::WriteMemory(VAddr cpu_addr, size_t size) {
    ForEachImageInRegion(cpu_addr, size, [this](ImageId image_id, Image& image) {
        if (True(image.flags & ImageFlagBits::CpuModified)) {
            return;
        }
        image.flags |= ImageFlagBits::CpuModified;
        UntrackImage(image);
    });
}

template <class P>
void TextureCache<P>::DownloadMemory(VAddr cpu_addr, size_t size) {
    ForEachImageInRegion(cpu_addr, size, [this](ImageId, Image& image) {
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            return;
        }
        image.flags &= ~ImageFlagBits::GpuModified;

        auto map = runtime.MapDownloadBuffer(image.unswizzled_size_bytes);
        const auto copies = FullDownloadCopies(image.info);
        image.DownloadMemory(map, 0, copies);
        SwizzleImage(gpu_memory, image.gpu_addr, image.info, copies, map.Span());
        LOG_INFO(HW_GPU, "Download");
    });
}

template <class P>
void TextureCache<P>::UnmapMemory(VAddr cpu_addr, size_t size) {
    std::vector<ImageId> deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        LOG_INFO(HW_GPU, "Deleting image: {}", id.index);
        Image& image = slot_images[id];
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image);
        }
        UnregisterImage(id);
        DeleteImage(id);
    }
    if (has_deleted_images) {
        InvalidateImageDescriptorTable();
    }
}

template <class P>
void TextureCache<P>::BlitImage(const Tegra::Engines::Fermi2D::Surface& dst,
                                const Tegra::Engines::Fermi2D::Surface& src,
                                const Tegra::Engines::Fermi2D::Config& copy) {
    const BlitImages images = GetBlitImages(dst, src);
    ImageBase& dst_image = slot_images[images.dst_id];
    const ImageBase& src_image = slot_images[images.src_id];
    dst_image.flags |= ImageFlagBits::GpuModified;

    const std::optional dst_base = dst_image.FindSubresourceFromAddress(dst.Address());
    UNIMPLEMENTED_IF(dst_base->mipmap != 0);
    const SubresourceRange dst_range{.base = dst_base.value(), .extent = {1, 1}};
    const ImageViewInfo dst_view_info(ImageViewType::e2D, images.dst_format, dst_range);
    const ImageViewId dst_view_id = FindOrEmplaceImageView(images.dst_id, dst_view_info);
    const Extent3D dst_extent = dst_image.info.size; // TODO: Apply mips
    const RenderTargets dst_targets{
        .color_buffer_ids = {dst_view_id},
        .size = {dst_extent.width, dst_extent.height},
    };
    Framebuffer* const dst_framebuffer = GetFramebuffer(dst_targets);

    const std::optional src_base = src_image.FindSubresourceFromAddress(src.Address());
    const SubresourceRange src_range{.base = src_base.value(), .extent = {1, 1}};
    UNIMPLEMENTED_IF(src_base->mipmap != 0);
    const ImageViewInfo src_view_info(ImageViewType::e2D, images.src_format, src_range);
    const ImageViewId src_view_id = FindOrEmplaceImageView(images.src_id, src_view_info);

    if constexpr (FRAMEBUFFER_BLITS) {
        // OpenGL blits framebuffers, not images
        const Extent3D src_extent = src_image.info.size; // TODO: Apply mips
        Framebuffer* const src_framebuffer = GetFramebuffer(RenderTargets{
            .color_buffer_ids = {src_view_id},
            .size = {src_extent.width, src_extent.height},
        });
        runtime.BlitFramebuffer(dst_framebuffer, src_framebuffer, copy);
    } else {
        ImageView& dst_view = slot_image_views[dst_view_id];
        ImageView& src_view = slot_image_views[src_view_id];
        runtime.BlitImage(dst_framebuffer, dst_view, src_view, copy);
    }
}

template <class P>
void TextureCache<P>::InvalidateColorBuffer(size_t index) {
    ImageView*& color_buffer = render_targets.color_buffers[index];
    color_buffer = FindColorBuffer(index);
    if (!color_buffer) {
        LOG_ERROR(HW_GPU, "Invalidating invalid color buffer in index={}", index);
        return;
    }
    // When invalidating a color buffer, the old contents are no longer relevant
    Image& image = slot_images[color_buffer->image];
    image.flags &= ~ImageFlagBits::CpuModified;
    image.flags &= ~ImageFlagBits::GpuModified;

    runtime.InvalidateColorBuffer(color_buffer, index);
}

template <class P>
void TextureCache<P>::InvalidateDepthBuffer() {
    ImageViewId& depth_buffer_id = render_targets.depth_buffer;
    depth_buffer_id = FindDepthBuffer();
    if (!depth_buffer_id) {
        LOG_ERROR(HW_GPU, "Invalidating invalid depth buffer");
        return;
    }
    // When invalidating the depth buffer, the old contents are no longer relevant
    Image& image = slot_images[slot_image_views[depth_buffer_id].image_id];
    image.flags &= ~ImageFlagBits::CpuModified;
    image.flags &= ~ImageFlagBits::GpuModified;

    runtime.InvalidateDepthBuffer(depth_buffer_id);
}

template <class P>
void TextureCache<P>::InvalidateImageDescriptorTable() {
    UpdateImageDescriptorTable(tables_3d, maxwell3d.regs.tic.Address(),
                               maxwell3d.regs.tic.limit + 1);
    UpdateImageDescriptorTable(tables_compute, kepler_compute.regs.tic.Address(),
                               kepler_compute.regs.tic.limit + 1);
}

template <class P>
void TextureCache<P>::InvalidateSamplerDescriptorTable() {
    UpdateSamplerDescriptorTable(tables_3d, maxwell3d.regs.tsc.Address(),
                                 maxwell3d.regs.tsc.limit + 1);
    UpdateSamplerDescriptorTable(tables_compute, kepler_compute.regs.tsc.Address(),
                                 kepler_compute.regs.tsc.limit + 1);
}

template <class P>
typename P::ImageView* TextureCache<P>::TryFindFramebufferImageView(VAddr cpu_addr) {
    // TODO: Properly implement this
    const auto it = page_table.find(cpu_addr >> PAGE_SHIFT);
    if (it == page_table.end()) {
        return nullptr;
    }
    const auto& image_ids = it->second;
    for (const ImageId image_id : image_ids) {
        const Image& image = slot_images[image_id];
        if (image.cpu_addr != cpu_addr) {
            continue;
        }
        if (image.image_view_ids.empty()) {
            continue;
        }
        return &slot_image_views[image.image_view_ids.at(0)];
    }
    return nullptr;
}

template <class P>
void TextureCache<P>::UpdateImageDescriptorTable(ClassDescriptorTables& tables,
                                                 GPUVAddr tic_address, size_t num_tics) {
    if (tic_address == 0) {
        tables.tic_entries.clear();
        return;
    }
    tables.tic_entries.resize(num_tics);
    gpu_memory.ReadBlock(tic_address, tables.tic_entries.data(), num_tics * sizeof(TICEntry));
}

template <class P>
void TextureCache<P>::UpdateSamplerDescriptorTable(ClassDescriptorTables& tables,
                                                   GPUVAddr tsc_address, size_t num_tscs) {
    if (tsc_address == 0) {
        tables.tsc_entries.clear();
        tables.samplers.clear();
        return;
    }
    tables.tsc_entries.resize(num_tscs);
    tables.samplers.resize(num_tscs);
    tables.stored_tsc_entries.resize(num_tscs);
    gpu_memory.ReadBlock(tsc_address, tables.tsc_entries.data(), num_tscs * sizeof(TSCEntry));

    while (num_tscs--) {
        const size_t index = num_tscs;
        TSCEntry& stored_config = tables.stored_tsc_entries[index];
        const SamplerId existing_id = tables.samplers[index];
        const TSCEntry& config = tables.tsc_entries[index];
        tables.samplers[index] = FindSampler(stored_config, existing_id, config);
    }
}

inline u32 MapSizeBytes(const ImageBase& image) {
    if (True(image.flags & ImageFlagBits::AcceleratedUpload)) {
        return image.guest_size_bytes;
    } else if (True(image.flags & ImageFlagBits::Converted)) {
        return image.converted_size_bytes;
    } else {
        return image.unswizzled_size_bytes;
    }
}

template <class P>
void TextureCache<P>::UpdateImageContents(Image& image) {
    if (False(image.flags & ImageFlagBits::CpuModified)) {
        // Only upload modified images
        return;
    }
    image.flags &= ~ImageFlagBits::CpuModified;

    auto map = runtime.MapUploadBuffer(MapSizeBytes(image));
    UploadImageContents(image, map, 0);
    runtime.InsertUploadMemoryBarrier();
}

template <class P>
template <typename MapBuffer>
void TextureCache<P>::UploadImageContents(Image& image, MapBuffer& map, size_t buffer_offset) {
    const std::span<u8> mapped_span = map.Span();
    const GPUVAddr gpu_addr = image.gpu_addr;

    if (True(image.flags & ImageFlagBits::AcceleratedUpload)) {
        gpu_memory.ReadBlockUnsafe(gpu_addr, mapped_span.data(), mapped_span.size_bytes());
        const auto uploads = FullUploadSwizzles(image.info);
        runtime.AccelerateImageUpload(image, map, buffer_offset, uploads);
    } else if (True(image.flags & ImageFlagBits::Converted)) {
        std::vector<u8> unswizzled_data(image.unswizzled_size_bytes);
        auto copies = UnswizzleImage(gpu_memory, gpu_addr, image.info, unswizzled_data);
        ConvertImage(unswizzled_data, image.info, mapped_span, copies);
        image.UploadMemory(map, buffer_offset, copies);
    } else if (image.info.type == ImageType::Buffer) {
        const std::array copies{UploadBufferCopy(gpu_memory, gpu_addr, image, mapped_span)};
        image.UploadMemory(map, buffer_offset, copies);
    } else {
        const auto copies = UnswizzleImage(gpu_memory, gpu_addr, image.info, mapped_span);
        image.UploadMemory(map, buffer_offset, copies);
    }
    TrackImage(image);
}

template <class P>
ImageViewId TextureCache<P>::FindImageView(const TICEntry& config) {
    // TODO: Add fast path here comparing to previous TIC

    if (!IsValidAddress(gpu_memory, config)) {
        return NULL_IMAGE_VIEW_ID;
    }
    const auto [pair, is_new] = image_views.try_emplace(config);
    ImageViewId& image_view_id = pair->second;
    if (is_new) {
        image_view_id = CreateImageView(config);
    }
    return image_view_id;
}

template <class P>
ImageViewId TextureCache<P>::CreateImageView(const TICEntry& config) {
    const ImageInfo info(config);
    const GPUVAddr image_gpu_addr = config.Address() - config.BaseLayer() * info.layer_stride;
    const ImageId image_id = FindOrInsertImage(info, image_gpu_addr);
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }
    const ImageViewInfo view_info(config);
    const ImageViewId image_view_id = FindOrEmplaceImageView(image_id, view_info);
    ImageBase& image = slot_images[image_id];
    ImageViewBase& image_view = slot_image_views[image_view_id];
    image_view.flags |= ImageViewFlagBits::Strong;
    image.flags |= ImageFlagBits::Strong;
    return image_view_id;
}

template <class P>
ImageId TextureCache<P>::FindOrInsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                           RelaxedOptions options) {
    if (const ImageId image_id = FindImage(info, gpu_addr, options); image_id) {
        return image_id;
    }
    return InsertImage(info, gpu_addr, options);
}

template <class P>
ImageId TextureCache<P>::FindImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                   RelaxedOptions options) {
    const std::optional<VAddr> cpu_addr = gpu_memory.GpuToCpuAddress(gpu_addr);
    if (!cpu_addr) {
        return ImageId{};
    }
    ImageId image_id;
    const size_t size = CalculateGuestSizeInBytes(info);
    ForEachImageInRegion(*cpu_addr, size, [&](ImageId existing_image_id, Image& existing_image) {
        if (info.type == ImageType::Linear || existing_image.info.type == ImageType::Linear) {
            const bool strict_size = False(options & RelaxedOptions::Size) &&
                                     True(existing_image.flags & ImageFlagBits::Strong);
            const ImageInfo& existing = existing_image.info;
            if (existing.type == info.type && existing.pitch == info.pitch &&
                IsPitchLinearSameSize(existing, info, strict_size) &&
                IsViewCompatible(existing.format, info.format)) {
                image_id = existing_image_id;
                return true;
            }
        } else if (IsSubresource(info, existing_image, gpu_addr, options)) {
            image_id = existing_image_id;
            return true;
        }
        return false;
    });
    return image_id;
}

template <class P>
ImageId TextureCache<P>::InsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                     RelaxedOptions options) {
    const std::optional<VAddr> cpu_addr = gpu_memory.GpuToCpuAddress(gpu_addr);
    ASSERT_MSG(cpu_addr, "Tried to insert an image to an invalid gpu_addr=0x{:x}", gpu_addr);
    const ImageId image_id = ResolveImageOverlaps(info, gpu_addr, *cpu_addr, options);
    const Image& image = slot_images[image_id];
    // Using "image.gpu_addr" instead of "gpu_addr" is important because it might be different
    const auto [it, is_new] = image_allocs_table.try_emplace(image.gpu_addr);
    if (is_new) {
        it->second = slot_image_allocs.insert();
    }
    slot_image_allocs[it->second].images.push_back(image_id);
    return image_id;
}

template <class P>
ImageId TextureCache<P>::ResolveImageOverlaps(const ImageInfo& info, GPUVAddr gpu_addr,
                                              VAddr cpu_addr, RelaxedOptions options) {
    ImageInfo new_info = info;
    const size_t size_bytes = CalculateGuestSizeInBytes(new_info);
    std::vector<ImageId> overlap_ids;
    if (new_info.type != ImageType::Linear) {
        ForEachImageInRegion(cpu_addr, size_bytes, [&](ImageId overlap_id, Image& overlap) {
            if (overlap.info.type == ImageType::Linear) {
                return;
            }
            if (!IsLayerStrideCompatible(new_info, overlap.info)) {
                return;
            }
            if (!IsCopyCompatible(overlap.info.format, new_info.format)) {
                return;
            }
            const bool strict_size = False(options & RelaxedOptions::Size) &&
                                     True(overlap.flags & ImageFlagBits::Strong);
            const std::optional solution =
                ResolveOverlap(new_info, gpu_addr, cpu_addr, overlap, strict_size);
            if (!solution) {
                return;
            }
            gpu_addr = solution->gpu_addr;
            cpu_addr = solution->cpu_addr;
            new_info.resources = solution->resources;
            overlap_ids.push_back(overlap_id);
        });
    }
    const ImageId new_image_id = slot_images.insert(runtime, new_info, gpu_addr, cpu_addr);
    Image& new_image = slot_images[new_image_id];

    // TODO: Only upload what we need
    UpdateImageContents(new_image);

    for (const ImageId overlap_id : overlap_ids) {
        Image& overlap = slot_images[overlap_id];
        const SubresourceBase base = new_image.FindSubresourceFromAddress(overlap.gpu_addr).value();
        runtime.CopyImage(new_image, overlap, MakeShrinkImageCopies(new_info, overlap.info, base));

        UntrackImage(overlap);
        UnregisterImage(overlap_id);
        DeleteImage(overlap_id);
    }
    RegisterImage(new_image_id);
    return new_image_id;
}

template <class P>
typename TextureCache<P>::BlitImages TextureCache<P>::GetBlitImages(
    const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src) {
    static constexpr auto FIND_OPTIONS = RelaxedOptions::Size | RelaxedOptions::Format;
    const GPUVAddr dst_addr = dst.Address();
    const GPUVAddr src_addr = src.Address();
    ImageInfo dst_info(dst);
    ImageInfo src_info(src);
    ImageId dst_id;
    ImageId src_id;
    do {
        has_deleted_images = false;
        dst_id = FindImage(dst_info, dst_addr, FIND_OPTIONS);
        src_id = FindImage(src_info, src_addr, FIND_OPTIONS);
        const ImageBase* const dst_image = dst_id ? &slot_images[dst_id] : nullptr;
        const ImageBase* const src_image = src_id ? &slot_images[src_id] : nullptr;
        DeduceBlitImages(dst_info, src_info, dst_image, src_image);
        if (GetFormatType(dst_info.format) != GetFormatType(src_info.format)) {
            continue;
        }
        if (!dst_id) {
            dst_id = InsertImage(dst_info, dst_addr, RelaxedOptions::Size);
        }
        if (!src_id) {
            src_id = InsertImage(src_info, src_addr, RelaxedOptions::Size);
        }
    } while (has_deleted_images);
    return BlitImages{
        .dst_id = dst_id,
        .src_id = src_id,
        .dst_format = dst_info.format,
        .src_format = src_info.format,
    };
}

template <class P>
SamplerId TextureCache<P>::FindSampler(TSCEntry& stored_config, SamplerId existing_id,
                                       const TSCEntry& config) {
    if (stored_config == config) {
        return existing_id;
    }
    stored_config = config;

    if (std::ranges::all_of(config.raw, [](u64 value) { return value == 0; })) {
        return SamplerId{};
    }
    const auto [pair, is_new] = samplers.try_emplace(config);
    if (is_new) {
        pair->second = slot_samplers.insert(runtime, config);
    }
    return pair->second;
}

template <class P>
ImageViewId TextureCache<P>::FindColorBuffer(size_t index) {
    const auto& regs = maxwell3d.regs;
    if (index >= regs.rt_control.count) {
        return ImageViewId{};
    }
    const auto& rt = regs.rt[index];
    const GPUVAddr gpu_addr = rt.Address();
    if (gpu_addr == 0) {
        return ImageViewId{};
    }
    if (rt.format == Tegra::RenderTargetFormat::NONE) {
        return ImageViewId{};
    }
    const ImageInfo info(regs, index);
    return FindRenderTargetView(info, gpu_addr);
}

template <class P>
ImageViewId TextureCache<P>::FindDepthBuffer() {
    const auto& regs = maxwell3d.regs;
    if (!regs.zeta_enable) {
        return ImageViewId{};
    }
    const GPUVAddr gpu_addr = regs.zeta.Address();
    if (gpu_addr == 0) {
        return ImageViewId{};
    }
    const ImageInfo info(regs);
    return FindRenderTargetView(info, gpu_addr);
}

template <class P>
ImageViewId TextureCache<P>::FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr) {
    const ImageId image_id = FindOrInsertImage(info, gpu_addr, RelaxedOptions::Size);
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }
    Image& image = slot_images[image_id];
    const ImageViewType view_type = RenderTargetImageViewType(info);
    SubresourceBase base;
    if (image.info.type == ImageType::Linear) {
        base = SubresourceBase{.mipmap = 0, .layer = 0};
    } else {
        base = image.FindSubresourceFromAddress(gpu_addr).value();
    }
    const u32 layers = image.info.type == ImageType::e3D ? info.size.depth : info.resources.layers;
    const SubresourceRange range{
        .base = base,
        .extent = {.mipmaps = 1, .layers = layers},
    };
    return FindOrEmplaceImageView(image_id, ImageViewInfo(view_type, info.format, range));
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachImageInRegion(VAddr cpu_addr, size_t size, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 32> images;
    ForEachPage(cpu_addr, size, [this, &images, cpu_addr, size, func](u64 page) {
        const auto it = page_table.find(page);
        if (it == page_table.end()) {
            if constexpr (BOOL_BREAK) {
                return false;
            } else {
                return;
            }
        }
        for (const ImageId image_id : it->second) {
            Image& image = slot_images[image_id];
            if (True(image.flags & ImageFlagBits::Picked)) {
                continue;
            }
            if (!image.Overlaps(cpu_addr, size)) {
                continue;
            }
            image.flags |= ImageFlagBits::Picked;
            images.push_back(image_id);
            if constexpr (BOOL_BREAK) {
                if (func(image_id, image)) {
                    return true;
                }
            } else {
                func(image_id, image);
            }
        }
        if constexpr (BOOL_BREAK) {
            return false;
        }
    });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
}

template <class P>
ImageId TextureCache<P>::CreateNewImage(const ImageInfo& info, GPUVAddr gpu_addr, VAddr cpu_addr) {
    const ImageId image_id = slot_images.insert(runtime, info, gpu_addr, cpu_addr);
    InitializeNewImage(image_id);
    return image_id;
}

template <class P>
void TextureCache<P>::InitializeNewImage(ImageId image_id) {
    // Memory of new images has to be uploaded because there was no track of them before
    UpdateImageContents(slot_images[image_id]);
    // Track memory stores and reads for this image
    RegisterImage(image_id);
}

template <class P>
ImageViewId TextureCache<P>::FindOrEmplaceImageView(ImageId image_id, const ImageViewInfo& info) {
    Image& image = slot_images[image_id];
    if (const ImageViewId image_view_id = image.FindView(info); image_view_id) {
        return image_view_id;
    }
    const ImageViewId image_view_id = slot_image_views.insert(runtime, info, image_id, image);
    image.InsertView(info, image_view_id);
    return image_view_id;
}

template <class P>
void TextureCache<P>::RegisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    ForEachPage(image.cpu_addr, image.guest_size_bytes,
                [this, image_id](u64 page) { page_table[page].push_back(image_id); });
}

template <class P>
void TextureCache<P>::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already registered image");
    image.flags &= ~ImageFlagBits::Registered;
    ForEachPage(image.cpu_addr, image.guest_size_bytes, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == page_table.end()) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << PAGE_SHIFT);
            return;
        }
        std::vector<ImageId>& image_ids = page_it->second;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            UNREACHABLE_MSG("Unregistering unregistered image in page=0x{:x}", page << PAGE_SHIFT);
            return;
        }
        image_ids.erase(vector_it);
    });
}

template <class P>
void TextureCache<P>::TrackImage(Image& image) {
    ASSERT(False(image.flags & ImageFlagBits::Tracked));
    image.flags |= ImageFlagBits::Tracked;
    rasterizer.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, 1);
}

template <class P>
void TextureCache<P>::UntrackImage(Image& image) {
    ASSERT(True(image.flags & ImageFlagBits::Tracked));
    image.flags &= ~ImageFlagBits::Tracked;
    rasterizer.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, -1);
}

template <class P>
void TextureCache<P>::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    const GPUVAddr gpu_addr = image.gpu_addr;
    const auto alloc_it = image_allocs_table.find(gpu_addr);
    if (alloc_it == image_allocs_table.end()) {
        UNREACHABLE_MSG("Trying to delete an image alloc that does not exist in address 0x{:x}",
                        gpu_addr);
        return;
    }
    const ImageAllocId alloc_id = alloc_it->second;
    std::vector<ImageId>& alloc_images = slot_image_allocs[alloc_id].images;
    const auto alloc_image_it = std::ranges::find(alloc_images, image_id);
    if (alloc_image_it == alloc_images.end()) {
        UNREACHABLE_MSG("Trying to delete an image that does not exist");
        return;
    }
    ASSERT_MSG(False(image.flags & ImageFlagBits::Tracked), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    const std::span<const ImageViewId> image_view_ids = image.image_view_ids;
    if constexpr (ENABLE_VALIDATION) {
        std::ranges::replace(render_targets.color_buffer_ids, image_id, CORRUPT_ID);
        if (render_targets.depth_buffer_id == image_id) {
            render_targets.depth_buffer_id = CORRUPT_ID;
        }
    }
    RemoveImageViewReferences(image_view_ids);
    RemoveFramebuffers(image_view_ids);

    for (const ImageViewId image_view_id : image_view_ids) {
        sentenced_image_view.push_back(std::move(slot_image_views[image_view_id]));
        slot_image_views.erase(image_view_id);
    }

    sentenced_images.push_back(std::move(slot_images[image_id]));
    slot_images.erase(image_id);

    alloc_images.erase(alloc_image_it);
    if (alloc_images.empty()) {
        image_allocs_table.erase(alloc_it);
    }

    has_deleted_images = true;
}

template <class P>
void TextureCache<P>::RemoveImageViewReferences(std::span<const ImageViewId> removed_views) {
    auto it = image_views.begin();
    while (it != image_views.end()) {
        const auto found = std::ranges::find(removed_views, it->second);
        if (found != removed_views.end()) {
            it = image_views.erase(it);
        } else {
            ++it;
        }
    }
}

template <class P>
void TextureCache<P>::RemoveFramebuffers(std::span<const ImageViewId> removed_views) {
    auto it = framebuffers.begin();
    while (it != framebuffers.end()) {
        const RenderTargets& render_targets = it->first;
        if (render_targets.Contains(removed_views)) {
            it = framebuffers.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace VideoCommon
