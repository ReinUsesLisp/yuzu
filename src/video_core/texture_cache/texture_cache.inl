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
#include "video_core/texture_cache/samples_helper.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

template <class P>
TextureCache<P>::TextureCache(Runtime& runtime_, VideoCore::RasterizerInterface& rasterizer_,
                              Tegra::Engines::Maxwell3D& maxwell3d_,
                              Tegra::Engines::KeplerCompute& kepler_compute_,
                              Tegra::MemoryManager& gpu_memory_)
    : runtime{runtime_}, rasterizer{rasterizer_}, maxwell3d{maxwell3d_},
      kepler_compute{kepler_compute_}, gpu_memory{gpu_memory_} {
    TSCEntry sampler_descriptor{};
    sampler_descriptor.min_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    sampler_descriptor.mag_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    sampler_descriptor.mipmap_filter.Assign(Tegra::Texture::TextureMipmapFilter::Linear);
    sampler_descriptor.cubemap_anisotropy.Assign(1);

    // Make sure the first index is reserved for the null resources
    // This way the null resource becomes a compile time constant
    (void)slot_image_views.insert(runtime, NullImageParams{});
    (void)slot_samplers.insert(runtime, sampler_descriptor);
}

template <class P>
void TextureCache<P>::FillGraphicsImageViews(std::span<const u32> indices,
                                             std::span<ImageViewId> image_view_ids) {
    FillImageViews(graphics_image_table, graphics_image_view_ids, indices, image_view_ids);
}

template <class P>
void TextureCache<P>::FillComputeImageViews(std::span<const u32> indices,
                                            std::span<ImageViewId> image_view_ids) {
    FillImageViews(compute_image_table, compute_image_view_ids, indices, image_view_ids);
}

template <class P>
typename P::Sampler* TextureCache<P>::GetGraphicsSampler(u32 index) {
    [[unlikely]] if (index > graphics_sampler_table.Limit()) {
        LOG_ERROR(HW_GPU, "Invalid sampler index={}", index);
        return &slot_samplers[NULL_SAMPLER_ID];
    }
    const auto [descriptor, is_new] = graphics_sampler_table.Read(index);
    SamplerId& id = graphics_sampler_ids[index];
    [[unlikely]] if (is_new) {
        id = FindSampler(descriptor);
    }
    return &slot_samplers[id];
}

template <class P>
typename P::Sampler* TextureCache<P>::GetComputeSampler(u32 index) {
    [[unlikely]] if (index > compute_sampler_table.Limit()) {
        LOG_ERROR(HW_GPU, "Invalid sampler index={}", index);
        return &slot_samplers[NULL_SAMPLER_ID];
    }
    const auto [descriptor, is_new] = compute_sampler_table.Read(index);
    SamplerId& id = compute_sampler_ids[index];
    [[unlikely]] if (is_new) {
        id = FindSampler(descriptor);
    }
    return &slot_samplers[id];
}

template <class P>
void TextureCache<P>::SynchronizeGraphicsDescriptors() {
    using SamplerIndex = Tegra::Engines::Maxwell3D::Regs::SamplerIndex;
    const bool linked_tsc = maxwell3d.regs.sampler_index == SamplerIndex::ViaHeaderIndex;
    const u32 tic_limit = maxwell3d.regs.tic.limit;
    const u32 tsc_limit = linked_tsc ? tic_limit : maxwell3d.regs.tsc.limit;
    if (graphics_sampler_table.Synchornize(maxwell3d.regs.tsc.Address(), tsc_limit)) {
        graphics_sampler_ids.resize(tsc_limit + 1, CORRUPT_ID);
    }
    if (graphics_image_table.Synchornize(maxwell3d.regs.tic.Address(), tic_limit)) {
        graphics_image_view_ids.resize(tic_limit + 1, CORRUPT_ID);
    }
}

template <class P>
void TextureCache<P>::SynchronizeComputeDescriptors() {
    const bool linked_tsc = kepler_compute.launch_description.linked_tsc;
    const u32 tic_limit = kepler_compute.regs.tic.limit;
    const u32 tsc_limit = linked_tsc ? tic_limit : kepler_compute.regs.tsc.limit;
    const GPUVAddr tsc_gpu_addr = kepler_compute.regs.tsc.Address();
    if (compute_sampler_table.Synchornize(tsc_gpu_addr, tsc_limit)) {
        compute_sampler_ids.resize(tsc_limit + 1, CORRUPT_ID);
    }
    if (compute_image_table.Synchornize(kepler_compute.regs.tic.Address(), tic_limit)) {
        compute_image_view_ids.resize(tic_limit + 1, CORRUPT_ID);
    }
}

template <class P>
void TextureCache<P>::UpdateRenderTargets(bool is_clear) {
    using namespace VideoCommon::Dirty;
    auto& flags = maxwell3d.dirty.flags;
    if (!flags[Dirty::RenderTargets]) {
        return;
    }
    flags[Dirty::RenderTargets] = false;

    // Render target control is used on all render targets, so force look ups when this one is up
    const bool force = flags[Dirty::RenderTargetControl];
    flags[Dirty::RenderTargetControl] = false;

    for (size_t index = 0; index < NUM_RT; ++index) {
        ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
        if (flags[Dirty::ColorBuffer0 + index] || force) {
            flags[Dirty::ColorBuffer0 + index] = false;
            BindRenderTarget(&color_buffer_id, FindColorBuffer(index, is_clear));
        }
        PrepareRenderTarget(color_buffer_id);
    }
    if (flags[Dirty::ZetaBuffer] || force) {
        flags[Dirty::ZetaBuffer] = false;
        BindRenderTarget(&render_targets.depth_buffer_id, FindDepthBuffer(is_clear));
    }
    PrepareRenderTarget(render_targets.depth_buffer_id);

    for (size_t index = 0; index < NUM_RT; ++index) {
        render_targets.draw_buffers[index] = static_cast<u8>(maxwell3d.regs.rt_control.Map(index));
    }
    render_targets.size = Extent2D{
        maxwell3d.regs.render_area.width,
        maxwell3d.regs.render_area.height,
    };
}

template <class P>
typename P::Framebuffer* TextureCache<P>::GetFramebuffer() {
    return &slot_framebuffers[GetFramebufferId(render_targets)];
}

template <class P>
void TextureCache<P>::FillImageViews(DescriptorTable<TICEntry>& table,
                                     std::span<ImageViewId> cached_image_view_ids,
                                     std::span<const u32> indices,
                                     std::span<ImageViewId> image_view_ids) {
    const size_t num_indices = indices.size();
    ASSERT(num_indices <= image_view_ids.size());
    do {
        has_deleted_images = false;
        for (size_t i = num_indices; i--;) {
            const u32 index = indices[i];
            [[unlikely]] if (index > table.Limit()) {
                LOG_ERROR(HW_GPU, "Invalid image view index={}", index);
                image_view_ids[i] = NULL_IMAGE_VIEW_ID;
                continue;
            }
            const auto [descriptor, is_new] = table.Read(index);
            ImageViewId& image_view_id = cached_image_view_ids[index];
            if (is_new) {
                image_view_id = FindImageView(descriptor);
            }
            image_view_ids[i] = image_view_id;

            if (image_view_id != NULL_IMAGE_VIEW_ID) {
                ImageView* const image_view = &slot_image_views[image_view_id];
                const ImageId image_id = image_view->image_id;
                UpdateImageContents(slot_images[image_id]);
                SynchronizeAliases(image_id);
            }
        }
    } while (has_deleted_images);
}

template <class P>
FramebufferId TextureCache<P>::GetFramebufferId(const RenderTargets& key) {
    const auto [pair, is_new] = framebuffers.try_emplace(key);
    FramebufferId& framebuffer_id = pair->second;
    if (!is_new) {
        return framebuffer_id;
    }
    std::array<ImageView*, NUM_RT> color_buffers;
    std::ranges::transform(key.color_buffer_ids, color_buffers.begin(),
                           [this](ImageViewId id) { return id ? &slot_image_views[id] : nullptr; });
    ImageView* const depth_buffer =
        key.depth_buffer_id ? &slot_image_views[key.depth_buffer_id] : nullptr;
    framebuffer_id = slot_framebuffers.insert(runtime, slot_images, color_buffers, depth_buffer,
                                              key.draw_buffers, key.size);
    return framebuffer_id;
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
    std::vector<ImageId> images;
    ForEachImageInRegion(cpu_addr, size, [this, &images](ImageId image_id, ImageBase& image) {
        // Skip images that were not modified from the GPU
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            return;
        }
        // Skip images that .are. modified from the CPU
        // We don't want to write sensitive data from the guest
        if (True(image.flags & ImageFlagBits::CpuModified)) {
            return;
        }
        image.flags &= ~ImageFlagBits::GpuModified;
        images.push_back(image_id);
    });
    if (images.empty()) {
        return;
    }
    LOG_INFO(HW_GPU, "Download");
    std::ranges::sort(images, [this](ImageId lhs, ImageId rhs) {
        return slot_images[lhs].modification_tick < slot_images[rhs].modification_tick;
    });
    for (const ImageId image_id : images) {
        Image& image = slot_images[image_id];
        auto map = runtime.MapDownloadBuffer(image.unswizzled_size_bytes);
        const auto copies = FullDownloadCopies(image.info);
        image.DownloadMemory(map, 0, copies);
        SwizzleImage(gpu_memory, image.gpu_addr, image.info, copies, map.Span());
    }
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
}

template <class P>
void TextureCache<P>::BlitImage(const Tegra::Engines::Fermi2D::Surface& dst,
                                const Tegra::Engines::Fermi2D::Surface& src,
                                const Tegra::Engines::Fermi2D::Config& copy) {
    const BlitImages images = GetBlitImages(dst, src);
    const ImageId dst_id = images.dst_id;
    const ImageId src_id = images.src_id;
    SynchronizeAliases(dst_id);
    SynchronizeAliases(src_id);

    ImageBase& dst_image = slot_images[dst_id];
    const ImageBase& src_image = slot_images[src_id];

    MarkModification(dst_image);

    // TODO: Deduplicate
    const std::optional dst_base = dst_image.TryFindBase(dst.Address());
    const SubresourceRange dst_range{.base = dst_base.value(), .extent = {1, 1}};
    const ImageViewInfo dst_view_info(ImageViewType::e2D, images.dst_format, dst_range);
    const auto [dst_framebuffer_id, dst_view_id] = RenderTargetFromImage(dst_id, dst_view_info);
    const auto [src_samples_x, src_samples_y] = SamplesLog2(src_image.info.num_samples);
    const std::array src_region{
        Offset2D{.x = copy.src_x0 >> src_samples_x, .y = copy.src_y0 >> src_samples_y},
        Offset2D{.x = copy.src_x1 >> src_samples_x, .y = copy.src_y1 >> src_samples_y},
    };

    const std::optional src_base = src_image.TryFindBase(src.Address());
    const SubresourceRange src_range{.base = src_base.value(), .extent = {1, 1}};
    const ImageViewInfo src_view_info(ImageViewType::e2D, images.src_format, src_range);
    const auto [src_framebuffer_id, src_view_id] = RenderTargetFromImage(src_id, src_view_info);
    const auto [dst_samples_x, dst_samples_y] = SamplesLog2(dst_image.info.num_samples);
    const std::array dst_region{
        Offset2D{.x = copy.dst_x0 >> dst_samples_x, .y = copy.dst_y0 >> dst_samples_y},
        Offset2D{.x = copy.dst_x1 >> dst_samples_x, .y = copy.dst_y1 >> dst_samples_y},
    };

    // Always call this after src_framebuffer_id was queried, as the address might be invalidated.
    Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_framebuffer_id];
    if constexpr (FRAMEBUFFER_BLITS) {
        // OpenGL blits from framebuffers, not images
        Framebuffer* const src_framebuffer = &slot_framebuffers[src_framebuffer_id];
        runtime.BlitFramebuffer(dst_framebuffer, src_framebuffer, dst_region, src_region,
                                copy.filter, copy.operation);
    } else {
        // Vulkan can blit images, but it lacks format reinterpretations
        // Provide a framebuffer in case it's necessary
        ImageView& dst_view = slot_image_views[dst_view_id];
        ImageView& src_view = slot_image_views[src_view_id];
        runtime.BlitImage(dst_framebuffer, dst_view, src_view, dst_region, src_region, copy.filter,
                          copy.operation);
    }
}

template <class P>
void TextureCache<P>::InvalidateColorBuffer(size_t index) {
    ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
    color_buffer_id = FindColorBuffer(index, false);
    if (!color_buffer_id) {
        LOG_ERROR(HW_GPU, "Invalidating invalid color buffer in index={}", index);
        return;
    }
    // When invalidating a color buffer, the old contents are no longer relevant
    ImageView& color_buffer = slot_image_views[color_buffer_id];
    Image& image = slot_images[color_buffer.image_id];
    image.flags &= ~ImageFlagBits::CpuModified;
    image.flags &= ~ImageFlagBits::GpuModified;

    runtime.InvalidateColorBuffer(color_buffer, index);
}

template <class P>
void TextureCache<P>::InvalidateDepthBuffer() {
    ImageViewId& depth_buffer_id = render_targets.depth_buffer_id;
    depth_buffer_id = FindDepthBuffer(false);
    if (!depth_buffer_id) {
        LOG_ERROR(HW_GPU, "Invalidating invalid depth buffer");
        return;
    }
    // When invalidating the depth buffer, the old contents are no longer relevant
    ImageBase& image = slot_images[slot_image_views[depth_buffer_id].image_id];
    image.flags &= ~ImageFlagBits::CpuModified;
    image.flags &= ~ImageFlagBits::GpuModified;

    ImageView& depth_buffer = slot_image_views[depth_buffer_id];
    runtime.InvalidateDepthBuffer(depth_buffer);
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
        const ImageBase& image = slot_images[image_id];
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
bool TextureCache<P>::HasUncommittedFlushes() const noexcept {
    return uncommitted_downloads.empty();
}

template <class P>
bool TextureCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return !committed_downloads.empty() && !committed_downloads.front().empty();
}

template <class P>
void TextureCache<P>::CommitAsyncFlushes() {
    // This is intentionally passing the value by copy
    committed_downloads.push(uncommitted_downloads);
    uncommitted_downloads.clear();
}

template <class P>
void TextureCache<P>::PopAsyncFlushes() {
    if (committed_downloads.empty()) {
        return;
    }
    for (const ImageId image_id : committed_downloads.front()) {
        // TODO: Optimize flush allocations to use a single map
        Image& image = slot_images[image_id];
        auto map = runtime.MapDownloadBuffer(image.unswizzled_size_bytes);
        const auto copies = FullDownloadCopies(image.info);
        image.DownloadMemory(map, 0, copies);
        SwizzleImage(gpu_memory, image.gpu_addr, image.info, copies, map.Span());
    }
    committed_downloads.pop();
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
    TrackImage(image);

    if (image.info.num_samples > 1) {
        LOG_WARNING(HW_GPU, "MSAA image uploads are not implemented");
        return;
    }
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
    ImageBase& image = slot_images[image_id];
    const SubresourceBase base = image.TryFindBase(config.Address()).value();
    ASSERT(base.mipmap == 0);
    const ImageViewInfo view_info(config, base.layer);
    const ImageViewId image_view_id = FindOrEmplaceImageView(image_id, view_info);
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
    const auto lambda = [&](ImageId existing_image_id, ImageBase& existing_image) {
        if (info.type == ImageType::Linear || existing_image.info.type == ImageType::Linear) {
            const bool strict_size = False(options & RelaxedOptions::Size) &&
                                     True(existing_image.flags & ImageFlagBits::Strong);
            const ImageInfo& existing = existing_image.info;
            if (existing_image.gpu_addr == gpu_addr && existing.type == info.type &&
                existing.pitch == info.pitch &&
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
    };
    ForEachImageInRegion(*cpu_addr, CalculateGuestSizeInBytes(info), lambda);
    return image_id;
}

template <class P>
ImageId TextureCache<P>::InsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                     RelaxedOptions options) {
    const std::optional<VAddr> cpu_addr = gpu_memory.GpuToCpuAddress(gpu_addr);
    ASSERT_MSG(cpu_addr, "Tried to insert an image to an invalid gpu_addr=0x{:x}", gpu_addr);
    const ImageId image_id = ResolveImageOverlaps(info, gpu_addr, *cpu_addr);
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
                                              VAddr cpu_addr) {
    ImageInfo new_info = info;
    const size_t size_bytes = CalculateGuestSizeInBytes(new_info);
    std::vector<ImageId> overlap_ids;
    std::vector<ImageId> left_aliased_ids;
    std::vector<ImageId> right_aliased_ids;
    ForEachImageInRegion(cpu_addr, size_bytes, [&](ImageId overlap_id, ImageBase& overlap) {
        if (info.type == ImageType::Linear) {
            return;
        }
        if (overlap.info.type == ImageType::Linear) {
            return;
        }
        const auto solution = ResolveOverlap(new_info, gpu_addr, cpu_addr, overlap, true);
        if (solution) {
            gpu_addr = solution->gpu_addr;
            cpu_addr = solution->cpu_addr;
            new_info.resources = solution->resources;
            overlap_ids.push_back(overlap_id);
            return;
        }
        static constexpr auto options = RelaxedOptions::Size | RelaxedOptions::Format;
        const ImageBase new_image_base(new_info, gpu_addr, cpu_addr);
        if (IsSubresource(new_info, overlap, gpu_addr, options)) {
            left_aliased_ids.push_back(overlap_id);
        } else if (IsSubresource(overlap.info, new_image_base, overlap.gpu_addr, options)) {
            right_aliased_ids.push_back(overlap_id);
        }
    });
    const ImageId new_image_id = slot_images.insert(runtime, new_info, gpu_addr, cpu_addr);
    Image& new_image = slot_images[new_image_id];

    // TODO: Only upload what we need
    UpdateImageContents(new_image);

    for (const ImageId overlap_id : overlap_ids) {
        Image& overlap = slot_images[overlap_id];
        if (overlap.info.num_samples != new_image.info.num_samples) {
            LOG_WARNING(HW_GPU, "Copying between images with different samples is not implemented");
        } else {
            const SubresourceBase base = new_image.TryFindBase(overlap.gpu_addr).value();
            const auto copies = MakeShrinkImageCopies(new_info, overlap.info, base);
            runtime.CopyImage(new_image, overlap, copies);
        }
        if (True(overlap.flags & ImageFlagBits::Tracked)) {
            UntrackImage(overlap);
        }
        UnregisterImage(overlap_id);
        DeleteImage(overlap_id);
    }
    ImageBase& new_image_base = new_image;
    for (const ImageId aliased_id : right_aliased_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        AddImageAlias(new_image_base, aliased, new_image_id, aliased_id);
    }
    for (const ImageId aliased_id : left_aliased_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        AddImageAlias(aliased, new_image_base, aliased_id, new_image_id);
    }
    RegisterImage(new_image_id);
    return new_image_id;
}

template <class P>
typename TextureCache<P>::BlitImages TextureCache<P>::GetBlitImages(
    const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src) {
    static constexpr auto FIND_OPTIONS = RelaxedOptions::Format | RelaxedOptions::Samples;
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
            dst_id = InsertImage(dst_info, dst_addr, RelaxedOptions{});
        }
        if (!src_id) {
            src_id = InsertImage(src_info, src_addr, RelaxedOptions{});
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
SamplerId TextureCache<P>::FindSampler(const TSCEntry& config) {
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
ImageViewId TextureCache<P>::FindColorBuffer(size_t index, bool is_clear) {
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
    return FindRenderTargetView(info, gpu_addr, is_clear);
}

template <class P>
ImageViewId TextureCache<P>::FindDepthBuffer(bool is_clear) {
    const auto& regs = maxwell3d.regs;
    if (!regs.zeta_enable) {
        return ImageViewId{};
    }
    const GPUVAddr gpu_addr = regs.zeta.Address();
    if (gpu_addr == 0) {
        return ImageViewId{};
    }
    const ImageInfo info(regs);
    return FindRenderTargetView(info, gpu_addr, is_clear);
}

template <class P>
ImageViewId TextureCache<P>::FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr,
                                                  bool is_clear) {
    const auto options = is_clear ? RelaxedOptions::Samples : RelaxedOptions{};
    const ImageId image_id = FindOrInsertImage(info, gpu_addr, options);
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }
    Image& image = slot_images[image_id];
    const ImageViewType view_type = RenderTargetImageViewType(info);
    SubresourceBase base;
    if (image.info.type == ImageType::Linear) {
        base = SubresourceBase{.mipmap = 0, .layer = 0};
    } else {
        base = image.TryFindBase(gpu_addr).value();
    }
    const s32 layers = image.info.type == ImageType::e3D ? info.size.depth : info.resources.layers;
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
    ImageBase& image = slot_images[image_id];
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
void TextureCache<P>::TrackImage(ImageBase& image) {
    ASSERT(False(image.flags & ImageFlagBits::Tracked));
    image.flags |= ImageFlagBits::Tracked;
    rasterizer.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, 1);
}

template <class P>
void TextureCache<P>::UntrackImage(ImageBase& image) {
    ASSERT(True(image.flags & ImageFlagBits::Tracked));
    image.flags &= ~ImageFlagBits::Tracked;
    rasterizer.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, -1);
}

template <class P>
void TextureCache<P>::DeleteImage(ImageId image_id) {
    ImageBase& image = slot_images[image_id];
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

    // Mark render targets as dirty
    auto& dirty = maxwell3d.dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    const std::span<const ImageViewId> image_view_ids = image.image_view_ids;
    if constexpr (ENABLE_VALIDATION) {
        std::ranges::replace(render_targets.color_buffer_ids, image_id, CORRUPT_ID);
        if (render_targets.depth_buffer_id == image_id) {
            render_targets.depth_buffer_id = CORRUPT_ID;
        }
    }
    RemoveImageViewReferences(image_view_ids);
    RemoveFramebuffers(image_view_ids);

    for (const AliasedImage& alias : image.aliased_images) {
        ImageBase& other_image = slot_images[alias.id];
        [[maybe_unused]] const size_t num_removed_aliases =
            std::erase_if(other_image.aliased_images, [image_id](const AliasedImage& other_alias) {
                return other_alias.id == image_id;
            });
        ASSERT_MSG(num_removed_aliases == 1, "Invalid number of removed aliases: {}",
                   num_removed_aliases);
    }
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
    if constexpr (ENABLE_VALIDATION) {
        std::ranges::fill(graphics_image_view_ids, CORRUPT_ID);
        std::ranges::fill(compute_image_view_ids, CORRUPT_ID);
    }
    graphics_image_table.Invalidate();
    compute_image_table.Invalidate();
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

template <class P>
void TextureCache<P>::MarkModification(ImageBase& image) noexcept {
    image.flags |= ImageFlagBits::GpuModified;
    image.modification_tick = ++modification_tick;
}

template <class P>
void TextureCache<P>::SynchronizeAliases(ImageId image_id) {
    boost::container::small_vector<const AliasedImage*, 1> aliased_images;
    ImageBase& image = slot_images[image_id];
    u64 most_recent_tick = image.modification_tick;
    for (const AliasedImage& aliased : image.aliased_images) {
        ImageBase& aliased_image = slot_images[aliased.id];
        if (image.modification_tick < aliased_image.modification_tick) {
            most_recent_tick = std::max(most_recent_tick, aliased_image.modification_tick);
            aliased_images.push_back(&aliased);
        }
    }
    if (aliased_images.empty()) {
        return;
    }
    image.modification_tick = most_recent_tick;
    std::ranges::sort(aliased_images, [this](const AliasedImage* lhs, const AliasedImage* rhs) {
        const ImageBase& lhs_image = slot_images[lhs->id];
        const ImageBase& rhs_image = slot_images[rhs->id];
        return lhs_image.modification_tick < rhs_image.modification_tick;
    });
    for (const AliasedImage* const aliased : aliased_images) {
        CopyImage(image_id, aliased->id, aliased->copies);
    }
}

template <class P>
void TextureCache<P>::CopyImage(ImageId dst_id, ImageId src_id, std::span<const ImageCopy> copies) {
    Image& dst = slot_images[dst_id];
    Image& src = slot_images[src_id];
    const auto dst_format_type = GetFormatType(dst.info.format);
    const auto src_format_type = GetFormatType(src.info.format);
    if (src_format_type == dst_format_type) {
        if constexpr (HAS_EMULATED_COPIES) {
            if (!runtime.CanImageBeCopied(dst, src)) {
                return runtime.EmulateCopyImage(dst, src, copies);
            }
        }
        return runtime.CopyImage(dst, src, copies);
    }
    UNIMPLEMENTED_IF(dst.info.type != ImageType::e2D);
    UNIMPLEMENTED_IF(src.info.type != ImageType::e2D);
    for (const ImageCopy& copy : copies) {
        UNIMPLEMENTED_IF(copy.dst_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_offset != Offset3D{});
        UNIMPLEMENTED_IF(copy.dst_offset != Offset3D{});

        const SubresourceBase dst_base{
            .mipmap = copy.dst_subresource.base_mipmap,
            .layer = copy.dst_subresource.base_layer,
        };
        const SubresourceBase src_base{
            .mipmap = copy.src_subresource.base_mipmap,
            .layer = copy.src_subresource.base_layer,
        };
        const SubresourceExtent dst_extent{.mipmaps = 1, .layers = 1};
        const SubresourceExtent src_extent{.mipmaps = 1, .layers = 1};
        const SubresourceRange dst_range{.base = dst_base, .extent = dst_extent};
        const SubresourceRange src_range{.base = src_base, .extent = src_extent};
        const ImageViewInfo dst_view_info(ImageViewType::e2D, dst.info.format, dst_range);
        const ImageViewInfo src_view_info(ImageViewType::e2D, src.info.format, src_range);
        const auto [dst_framebuffer_id, dst_view_id] = RenderTargetFromImage(dst_id, dst_view_info);
        Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_framebuffer_id];
        const ImageViewId src_view_id = FindOrEmplaceImageView(src_id, src_view_info);
        ImageView& dst_view = slot_image_views[dst_view_id];
        ImageView& src_view = slot_image_views[src_view_id];
        [[maybe_unused]] const Extent3D expected_size{
            .width = std::min(dst_view.size.width, src_view.size.width),
            .height = std::min(dst_view.size.height, src_view.size.height),
            .depth = std::min(dst_view.size.depth, src_view.size.depth),
        };
        UNIMPLEMENTED_IF(copy.extent != expected_size);

        runtime.ConvertImage(dst_framebuffer, dst_view, src_view);
    }
}

template <class P>
void TextureCache<P>::BindRenderTarget(ImageViewId* old_id, ImageViewId new_id) {
    if (*old_id == new_id) {
        return;
    }
    if (*old_id) {
        const ImageViewBase& old_view = slot_image_views[*old_id];
        if (True(old_view.flags & ImageViewFlagBits::PreemtiveDownload)) {
            uncommitted_downloads.push_back(old_view.image_id);
        }
    }
    *old_id = new_id;
}

template <class P>
void TextureCache<P>::PrepareRenderTarget(ImageViewId id) {
    if (!id) {
        return;
    }
    const ImageId image_id = slot_image_views[id].image_id;
    Image& image = slot_images[image_id];
    UpdateImageContents(image);
    SynchronizeAliases(image_id);
    MarkModification(image);
}

template <class P>
std::pair<FramebufferId, ImageViewId> TextureCache<P>::RenderTargetFromImage(
    ImageId image_id, const ImageViewInfo& view_info) {
    const ImageViewId view_id = FindOrEmplaceImageView(image_id, view_info);
    const ImageBase& image = slot_images[image_id];
    const bool is_color = GetFormatType(image.info.format) == SurfaceType::ColorTexture;
    const ImageViewId color_view_id = is_color ? view_id : ImageViewId{};
    const ImageViewId depth_view_id = is_color ? ImageViewId{} : view_id;
    const Extent3D extent = MipSize(image.info.size, view_info.range.base.mipmap);
    const u32 num_samples = image.info.num_samples;
    const auto [samples_x, samples_y] = SamplesLog2(num_samples);
    const FramebufferId framebuffer_id = GetFramebufferId(RenderTargets{
        .color_buffer_ids = {color_view_id},
        .depth_buffer_id = depth_view_id,
        .size = {extent.width >> samples_x, extent.height >> samples_y},
    });
    return {framebuffer_id, view_id};
}

} // namespace VideoCommon
