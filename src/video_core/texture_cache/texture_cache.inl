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
    if (!maxwell3d.dirty.flags[Dirty::Descriptors]) {
        return;
    }
    maxwell3d.dirty.flags[Dirty::Descriptors] = false;

    InvalidateImageDescriptorTable();
    InvalidateSamplerDescriptorTable();
}

template <class P>
typename P::ImageView* TextureCache<P>::GetGraphicsImageView(size_t index) {
    return &slot_image_views[tables_3d.image_views[index]];
}

template <class P>
typename P::ImageView* TextureCache<P>::GetComputeImageView(size_t index) {
    return &slot_image_views[tables_compute.image_views[index]];
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
    framebuffer =
        slot_framebuffers.insert(runtime, color_buffers, depth_buffer, key.draw_buffers, key.size);

    return &slot_framebuffers[framebuffer];
}

template <class P>
void TextureCache<P>::WriteMemory(VAddr cpu_addr, size_t size) {
    ForEachImageInRegion(cpu_addr, size, [this](ImageId, Image& image) {
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

        auto map = runtime.MapDownloadBuffer(image.host_size_in_bytes);
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
void TextureCache<P>::BlitImage(const Tegra::Engines::Fermi2D::Regs::Surface& dst,
                                const Tegra::Engines::Fermi2D::Regs::Surface& src,
                                const Tegra::Engines::Fermi2D::Config& copy) {
    return;
    /*
    UNIMPLEMENTED_IF(image_alloc_page_table.Find(src.Address()));
    UNIMPLEMENTED_IF(image_alloc_page_table.Find(dst.Address()));

    ImageAlloc* const src_alloc = GetImageAlloc(src.Address());
    ImageAlloc* const dst_alloc = GetImageAlloc(dst.Address());
    const ImageInfo src_info(src);
    const ImageInfo dst_info(dst);

    // TODO: Properly choose these
    // TODO: and use relaxed size comparisons
    Image* const src_image = FindImage(src_alloc->images, src_info);
    Image* const dst_image = FindImage(dst_alloc->images, dst_info);
    dst_image->flags |= ImageFlagBits::GpuModified;

    const ImageViewId src_image_view_id = src_image->image_view_ids.at(0);
    const ImageViewId dst_image_view_id = dst_image->image_view_ids.at(0);

    if constexpr (FRAMEBUFFER_BLITS) {
        Framebuffer* const dst_framebuffer = GetFramebuffer({
            .color_buffer_ids =
                {
                    dst_image_view_id,
                },
        });
        Framebuffer* const src_framebuffer = GetFramebuffer({
            .color_buffer_ids =
                {
                    src_image_view_id,
                },
        });
        runtime.BlitFramebuffer(dst_framebuffer, src_framebuffer, copy);
    } else {
        UNIMPLEMENTED();
    }
    */
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
    ++invalidation_tick;

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
void TextureCache<P>::InvalidateContents() {
    UploadModifiedImageViews(tables_3d);
    UploadModifiedImageViews(tables_compute);
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
        if (image.cpu_addr == cpu_addr) {
            return &slot_image_views[image.image_view_ids.at(0)];
        }
    }
    return nullptr;
}

template <class P>
void TextureCache<P>::UpdateImageDescriptorTable(ClassDescriptorTables& tables,
                                                 GPUVAddr tic_address, size_t num_tics) {
    if (tic_address == 0) {
        tables.tic_entries.clear();
        tables.image_views.clear();
        return;
    }
    tables.tic_entries.resize(num_tics);
    tables.image_views.resize(num_tics);
    gpu_memory.ReadBlock(tic_address, tables.tic_entries.data(), num_tics * sizeof(TICEntry));

    do {
        has_deleted_images = false;
        for (size_t index = num_tics; index--;) {
            tables.image_views[index] = FindImageView(tables.tic_entries[index]);
        }
    } while (has_deleted_images);
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

template <class P>
void TextureCache<P>::UploadModifiedImageViews(ClassDescriptorTables& tables) {
    for (const ImageId image_id : tables.active_images) {
        UpdateImageContents(slot_images[image_id]);
    }
}

template <class P>
void TextureCache<P>::UpdateImageContents(Image& image) {
    // Only upload modified images
    if (False(image.flags & ImageFlagBits::CpuModified)) {
        return;
    }
    image.flags &= ~ImageFlagBits::CpuModified;

    const bool accelerated = True(image.flags & ImageFlagBits::AcceleratedUpload);
    const size_t size_bytes = accelerated ? image.guest_size_in_bytes : image.host_size_in_bytes;
    auto map = runtime.MapUploadBuffer(size_bytes);
    UploadImageContents(image, map, 0);

    runtime.InsertUploadMemoryBarrier();
}

template <class P>
template <typename MapBuffer>
void TextureCache<P>::UploadImageContents(Image& image, MapBuffer& map, size_t buffer_offset) {
    const std::span<u8> span = map.Span();
    const GPUVAddr gpu_addr = image.gpu_addr;

    if (True(image.flags & ImageFlagBits::AcceleratedUpload)) {
        gpu_memory.ReadBlockUnsafe(gpu_addr, span.data(), image.guest_size_in_bytes);
        const auto uploads = FullUploadSwizzles(image.info);
        runtime.AccelerateImageUpload(image, map, buffer_offset, uploads);
    } else {
        const auto copies = UnswizzleImage(gpu_memory, gpu_addr, image.info, span);
        image.UploadMemory(map, buffer_offset, copies);
    }
    TrackImage(image);
}

template <class P>
ImageViewId TextureCache<P>::FindImageView(const TICEntry& config) {
    // TODO: Add fast path here comparing to previous TIC

    if (!IsValid(gpu_memory, config)) {
        return NULL_IMAGE_VIEW_ID;
    }
    const auto [pair, is_new] = image_views.try_emplace(config);
    ImageViewId& image_view_id = pair->second;
    if (is_new) {
        image_view_id = CreateImageView(config);
    }
    TouchImageView(image_view_id);
    return image_view_id;
}

template <class P>
ImageViewId TextureCache<P>::CreateImageView(const TICEntry& config) {
    const ImageId image_id = CreateImageIfNecessary(ImageInfo(config), config.Address(), true);
    const ImageViewInfo view_info(config);
    Image& image = slot_images[image_id];
    ImageViewId image_view_id = image.FindView(view_info);
    if (!image_view_id) {
        image_view_id = EmplaceImageView(image_id, view_info);
    }
    ImageView& image_view = slot_image_views[image_view_id];
    image_view.flags |= ImageViewFlagBits::Strong;
    image.flags |= ImageFlagBits::Strong;
    return image_view_id;
}

template <class P>
typename P::ImageAlloc& TextureCache<P>::GetImageAlloc(GPUVAddr gpu_addr) {
    if (const ImageAllocId id = image_alloc_page_table.Find(gpu_addr); id) {
        return slot_image_allocs[id];
    }
    const ImageAllocId id = slot_image_allocs.insert();
    image_alloc_page_table.PushFront(gpu_addr, id);
    return slot_image_allocs[id];
}

template <class P>
ImageId TextureCache<P>::CreateImageIfNecessary(const ImageInfo& info, GPUVAddr gpu_addr,
                                                bool strict_size) {
    ImageAlloc& alloc = GetImageAlloc(gpu_addr);
    std::vector<ImageId>& images = alloc.images;
    if (const ImageId image_id = FindImage(images, info); image_id) {
        return image_id;
    }
    const std::optional<VAddr> cpu_addr = gpu_memory.GpuToCpuAddress(gpu_addr);
    if (!cpu_addr) {
        return ImageId{};
    }
    const ImageId image_id = ResolveImageOverlaps(info, gpu_addr, *cpu_addr, strict_size);
    images.push_back(image_id);
    return image_id;
}

template <class P>
ImageId TextureCache<P>::ResolveImageOverlaps(ImageInfo new_info, GPUVAddr gpu_addr, VAddr cpu_addr,
                                              bool strict_size) {
    const size_t size_bytes = CalculateGuestSizeInBytes(new_info);
    std::vector<ImageId> overlap_ids;
    if (new_info.type != ImageType::Linear) {
        ForEachImageInRegion(cpu_addr, size_bytes, [&](ImageId overlap_id, Image& overlap) {
            const bool created_from_table = True(overlap.flags & ImageFlagBits::Strong);
            const std::optional solution = ResolveOverlap(new_info, gpu_addr, cpu_addr, overlap,
                                                          strict_size && created_from_table);
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
        const SubresourceBase base = new_image.FindSubresource(overlap.gpu_addr).value();
        runtime.CopyImage(new_image, overlap, MakeShrinkImageCopies(new_info, overlap.info, base));

        UntrackImage(overlap);
        UnregisterImage(overlap_id);
        DeleteImage(overlap_id);
    }

    RegisterImage(new_image_id);
    return new_image_id;
}

template <class P>
ImageId TextureCache<P>::FindUnderlyingImage(const TICEntry& config) const {
    const GPUVAddr gpu_addr = CalculateBaseAddress(config);
    const ImageAllocId alloc_id = image_alloc_page_table.Find(gpu_addr);
    ASSERT_MSG(alloc_id, "Image does not exist");
    const std::vector<ImageId>& images = slot_image_allocs[alloc_id].images;

    const ImageInfo new_info(config);
    for (const ImageId image_id : images) {
        const ImageInfo& info = slot_images[image_id].info;
        if (!IsViewCompatible(info.format, new_info.format) || info.type != new_info.type ||
            info.size != new_info.size || info.num_samples != new_info.num_samples ||
            info.resources.mipmaps < new_info.resources.mipmaps ||
            info.resources.layers < new_info.resources.layers) {
            continue;
        }
        if (info.type == ImageType::Linear) {
            if (info.pitch != new_info.pitch) {
                continue;
            }
        } else {
            if (info.block != new_info.block) {
                continue;
            }
        }
        return image_id;
    }

    UNREACHABLE();
    return ImageId{};
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
    return FindRenderTargetView(ImageInfo(regs, index), gpu_addr);
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
    return FindRenderTargetView(ImageInfo(regs), gpu_addr);
}

template <class P>
ImageViewId TextureCache<P>::FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr) {
    const ImageId image_id = CreateImageIfNecessary(info, gpu_addr, false);
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }

    Image& image = slot_images[image_id];
    const ImageViewType view_type = RenderTargetImageViewType(info);
    const SubresourceRange range{
        .base = image.FindSubresource(gpu_addr).value(),
        .extent =
            {
                .mipmaps = 1,
                .layers = info.resources.layers,
            },
    };
    const ImageViewInfo view_info(view_type, info.format, range);
    if (const ImageViewId image_view_id = image.FindView(view_info); image_view_id) {
        return image_view_id;
    }
    return EmplaceImageView(image_id, view_info);
}

template <class P>
ImageViewId TextureCache<P>::CreateRenderTargetFromScratch(const ImageInfo& info, GPUVAddr gpu_addr,
                                                           VAddr cpu_addr) {
    ImageAlloc& alloc = GetImageAlloc(gpu_addr);
    const ImageId image_id = alloc.images.emplace_back(CreateNewImage(info, gpu_addr, cpu_addr));
    UNIMPLEMENTED_IF(info.resources.layers != 1);

    const ImageViewInfo view_info(RenderTargetImageViewType(info), info.format);
    return EmplaceImageView(image_id, view_info);
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachImageInRegion(VAddr cpu_addr, size_t size, Func&& func) {
    using FuncReturn = std::invoke_result<Func, ImageId>;
    static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;

    boost::container::small_vector<ImageId, 32> images;

    ForEachPage(cpu_addr, size, [this, &images, cpu_addr, size, func](u64 page) {
        const auto it = page_table.find(page);
        if (it == page_table.end()) {
            return;
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
ImageViewId TextureCache<P>::EmplaceImageView(ImageId image_id, const ImageViewInfo& info) {
    Image& image = slot_images[image_id];
    const ImageViewId image_view_id = slot_image_views.insert(runtime, info, image_id, image);
    image.InsertView(info, image_view_id);
    return image_view_id;
}

template <class P>
void TextureCache<P>::TouchImageView(ImageViewId image_view_id) {
    ImageView& image_view = slot_image_views[image_view_id];
    image_view.invalidation_tick = invalidation_tick;
    slot_images[image_view.image_id].invalidation_tick = invalidation_tick;
}

template <class P>
void TextureCache<P>::RegisterImage(ImageId image_id) {
    const Image& image = slot_images[image_id];
    ForEachPage(image.cpu_addr, image.guest_size_in_bytes,
                [this, image_id](u64 page) { page_table[page].push_back(image_id); });
}

template <class P>
void TextureCache<P>::UnregisterImage(ImageId image_id) {
    const Image& image = slot_images[image_id];
    ForEachPage(image.cpu_addr, image.guest_size_in_bytes, [this, image_id](u64 page) {
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

    rasterizer.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_in_bytes, 1);
}

template <class P>
void TextureCache<P>::UntrackImage(Image& image) {
    ASSERT(True(image.flags & ImageFlagBits::Tracked));
    image.flags &= ~ImageFlagBits::Tracked;

    rasterizer.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_in_bytes, -1);
}

template <class P>
void TextureCache<P>::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    const GPUVAddr gpu_addr = image.gpu_addr;
    const ImageAllocId alloc_id = image_alloc_page_table.Find(gpu_addr);
    if (!alloc_id) {
        UNREACHABLE_MSG("Trying to delete an image that does not exist");
        return;
    }
    std::vector<ImageId>& alloc_images = slot_image_allocs[alloc_id].images;
    const auto alloc_image_it = std::ranges::find(alloc_images, image_id);
    if (alloc_image_it == alloc_images.end()) {
        UNREACHABLE_MSG("Trying to delete an image that does not exist");
        return;
    }

    ASSERT_MSG(False(image.flags & ImageFlagBits::Tracked), "Image was not untracked");

    const std::span<const ImageViewId> image_view_ids = image.image_view_ids;
    if constexpr (ENABLE_VALIDATION) {
        // TODO: Pollute render targets
        ReplaceRemovedInImageDescriptorTables(tables_3d, image_id, image_view_ids);
        ReplaceRemovedInImageDescriptorTables(tables_compute, image_id, image_view_ids);
    }
    RemoveImageViewReferences(image_view_ids);
    RemoveFramebuffers(image_view_ids);

    for (const ImageViewId image_view_id : image_view_ids) {
        slot_image_views.erase(image_view_id);
    }

    slot_images.erase(image_id);
    alloc_images.erase(alloc_image_it);
    if (alloc_images.empty()) {
        image_alloc_page_table.Erase(gpu_addr);
    }

    has_deleted_images = true;
}

template <class P>
void TextureCache<P>::ReplaceRemovedInImageDescriptorTables(
    ClassDescriptorTables& tables, ImageId image, std::span<const ImageViewId> removed_views) {
    std::ranges::replace(tables.active_images, image, ImageId{});

    for (const auto& removed_view : removed_views) {
        std::ranges::replace(tables.image_views, removed_view, ImageViewId{});
    }
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
