// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <bit>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "common/logging/log.h"
#include "video_core/compatible_formats.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/descriptor_table.h"
#include "video_core/texture_cache/format_lookup_table.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_info_page_table.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/image_view_info.h"
#include "video_core/texture_cache/render_targets.h"
#include "video_core/texture_cache/slot_vector.h"
#include "video_core/texture_cache/types.h"
#include "video_core/textures/texture.h"

namespace VideoCommon {

using Tegra::Texture::SwizzleSource;
using Tegra::Texture::TextureType;
using Tegra::Texture::TICEntry;
using Tegra::Texture::TSCEntry;
using VideoCore::Surface::GetFormatType;
using VideoCore::Surface::IsCopyCompatible;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::PixelFormatFromDepthFormat;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;
using VideoCore::Surface::SurfaceType;

template <class P>
class TextureCache {
    static constexpr u64 PAGE_SHIFT = 20;

    static constexpr bool ENABLE_VALIDATION = P::ENABLE_VALIDATION;
    static constexpr bool FRAMEBUFFER_BLITS = P::FRAMEBUFFER_BLITS;
    static constexpr bool HAS_EMULATED_COPIES = P::HAS_EMULATED_COPIES;

    static constexpr ImageViewId NULL_IMAGE_VIEW_ID{0};
    static constexpr SamplerId NULL_SAMPLER_ID{0};

    using Runtime = typename P::Runtime;
    using Image = typename P::Image;
    using ImageAlloc = typename P::ImageAlloc;
    using ImageView = typename P::ImageView;
    using Sampler = typename P::Sampler;
    using Framebuffer = typename P::Framebuffer;

    struct BlitImages {
        ImageId dst_id;
        ImageId src_id;
        PixelFormat dst_format;
        PixelFormat src_format;
    };

public:
    explicit TextureCache(Runtime&, VideoCore::RasterizerInterface&, Tegra::Engines::Maxwell3D&,
                          Tegra::Engines::KeplerCompute&, Tegra::MemoryManager&);

    void TickFrame();

    [[nodiscard]] std::unique_lock<std::mutex> AcquireLock() {
        return std::unique_lock{mutex};
    }

    [[nodiscard]] const ImageView& GetImageView(ImageViewId id) const noexcept {
        return slot_image_views[id];
    }

    [[nodiscard]] ImageView& GetImageView(ImageViewId id) noexcept {
        return slot_image_views[id];
    }

    void FillGraphicsImageViews(std::span<const u32> indices,
                                std::span<ImageViewId> image_view_ids);

    void FillComputeImageViews(std::span<const u32> indices, std::span<ImageViewId> image_view_ids);

    /**
     * Get the sampler from the graphics descriptor table in the specified index
     *
     * @param index Index in elements of the sampler
     * @returns     Sampler in the specified index
     *
     * @pre @a index is less than the size of the descriptor table
     */
    Sampler* GetGraphicsSampler(u32 index);

    /**
     * Get the sampler from the compute descriptor table in the specified index
     *
     * @param index Index in elements of the sampler
     * @returns     Sampler in the specified index
     *
     * @pre @a index is less than the size of the descriptor table
     */
    Sampler* GetComputeSampler(u32 index);

    void SynchronizeGraphicsDescriptors();

    void SynchronizeComputeDescriptors();

    /*
     * Update bound render targets and upload memory if necessary
     */
    void UpdateRenderTargets(bool is_clear);

    /**
     * Create or find a framebuffer with the currently bound render targets
     * @sa UpdateRenderTargets should be called before calling this method
     *
     * @returns Pointer to a valid framebuffer
     */
    Framebuffer* GetFramebuffer();

    /**
     * Mark images in a range as modified from the CPU
     *
     * @param cpu_addr Virtual CPU address where memory has been written
     * @param size     Size in bytes of the written memory
     */
    void WriteMemory(VAddr cpu_addr, size_t size);

    /**
     * Download contents of host images to guest memory in a region
     *
     * @note Memory download happens immediately when this method is called
     *
     * @param cpu_addr Virtual CPU address of the desired memory download
     * @param size     Size in bytes of the desired memory download
     */
    void DownloadMemory(VAddr cpu_addr, size_t size);

    void UnmapMemory(VAddr cpu_addr, size_t size);

    void BlitImage(const Tegra::Engines::Fermi2D::Surface& dst,
                   const Tegra::Engines::Fermi2D::Surface& src,
                   const Tegra::Engines::Fermi2D::Config& copy);

    /**
     * Invalidate the contents of the color buffer index
     * These contents become unspecified, the cache can assume aggressive optimizations.
     * @sa UpdateRenderTargets does not have to be called before this
     *
     * @param index Index of the color buffer to invalidate
     */
    void InvalidateColorBuffer(size_t index);

    /**
     * Invalidate the contents of the depth buffer
     * These contents become unspecified, the cache can assume aggressive optimizations.
     * @sa UpdateRenderTargets does not have to be called before this
     */
    void InvalidateDepthBuffer();

    /**
     * Tries to find a cached image view in the given CPU address
     *
     * @param cpu_addr Virtual CPU address
     * @returns        Pointer to image view
     */
    [[nodiscard]] ImageView* TryFindFramebufferImageView(VAddr cpu_addr);

    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    void CommitAsyncFlushes();

    void PopAsyncFlushes();

    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

private:
    /**
     * Iterate over all page indices in a range
     *
     * @param addr Start of the address to iterate
     * @param size Size in bytes of the address to iterate
     * @param func Function to call on each page index
     */
    template <typename Func>
    static void ForEachPage(VAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> PAGE_SHIFT;
        for (u64 page = addr >> PAGE_SHIFT; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    void FillImageViews(DescriptorTable<TICEntry>& table,
                        std::span<ImageViewId> cached_image_view_ids, std::span<const u32> indices,
                        std::span<ImageViewId> image_view_ids);

    FramebufferId GetFramebufferId(const RenderTargets& key);

    void UpdateImageContents(Image& image);

    template <typename MapBuffer>
    void UploadImageContents(Image& image, MapBuffer& map, size_t buffer_offset);

    [[nodiscard]] ImageViewId FindImageView(const TICEntry& config);

    [[nodiscard]] ImageViewId CreateImageView(const TICEntry& config);

    [[nodiscard]] ImageId FindOrInsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                            RelaxedOptions options = RelaxedOptions{});

    [[nodiscard]] ImageId FindImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                    RelaxedOptions options);

    [[nodiscard]] ImageId InsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                      RelaxedOptions options);

    [[nodiscard]] ImageId ResolveImageOverlaps(const ImageInfo& info, GPUVAddr gpu_addr,
                                               VAddr cpu_addr);

    [[nodiscard]] BlitImages GetBlitImages(const Tegra::Engines::Fermi2D::Surface& dst,
                                           const Tegra::Engines::Fermi2D::Surface& src);

    [[nodiscard]] SamplerId FindSampler(const TSCEntry& config);

    /**
     * Find or create an image view for the given color buffer index
     *
     * @param index Index of the color buffer to find
     * @returns     Image view of the given color buffer
     */
    [[nodiscard]] ImageViewId FindColorBuffer(size_t index, bool is_clear);

    /**
     * Find or create an image view for the depth buffer
     *
     * @returns Image view for the depth buffer
     */
    [[nodiscard]] ImageViewId FindDepthBuffer(bool is_clear);

    [[nodiscard]] ImageViewId FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr,
                                                   bool is_clear);

    template <typename Func>
    void ForEachImageInRegion(VAddr cpu_addr, size_t size, Func&& func);

    [[nodiscard]] ImageViewId FindOrEmplaceImageView(ImageId image_id, const ImageViewInfo& info);

    /**
     * Register image in the page table
     *
     * @param Image to register
     */
    void RegisterImage(ImageId image);

    /**
     * Unregister image from the page table
     *
     * @param Image to unregister
     */
    void UnregisterImage(ImageId image);

    void TrackImage(ImageBase& image);

    void UntrackImage(ImageBase& image);

    void DeleteImage(ImageId image);

    void RemoveImageViewReferences(std::span<const ImageViewId> removed_views);

    void RemoveFramebuffers(std::span<const ImageViewId> removed_views);

    void MarkModification(ImageBase& image) noexcept;

    void SynchronizeAliases(ImageId image_id);

    void PrepareImage(ImageId image_id, bool is_modification);

    void PrepareImageView(ImageViewId image_view_id, bool is_modification);

    void CopyImage(ImageId dst_id, ImageId src_id, std::span<const ImageCopy> copies);

    void BindRenderTarget(ImageViewId* old_id, ImageViewId new_id);

    [[nodiscard]] std::pair<FramebufferId, ImageViewId> RenderTargetFromImage(
        ImageId, const ImageViewInfo& view_info);

    Runtime& runtime;
    VideoCore::RasterizerInterface& rasterizer;
    Tegra::Engines::Maxwell3D& maxwell3d;
    Tegra::Engines::KeplerCompute& kepler_compute;
    Tegra::MemoryManager& gpu_memory;

    DescriptorTable<TICEntry> graphics_image_table{gpu_memory};
    DescriptorTable<TSCEntry> graphics_sampler_table{gpu_memory};
    std::vector<SamplerId> graphics_sampler_ids;
    std::vector<ImageViewId> graphics_image_view_ids;

    DescriptorTable<TICEntry> compute_image_table{gpu_memory};
    DescriptorTable<TSCEntry> compute_sampler_table{gpu_memory};
    std::vector<SamplerId> compute_sampler_ids;
    std::vector<ImageViewId> compute_image_view_ids;

    RenderTargets render_targets;

    std::mutex mutex;

    std::unordered_map<TICEntry, ImageViewId> image_views;
    std::unordered_map<TSCEntry, SamplerId> samplers;
    std::unordered_map<RenderTargets, FramebufferId> framebuffers;

    std::unordered_map<u64, std::vector<ImageId>> page_table;

    bool has_deleted_images = false;

    SlotVector<Image> slot_images;
    SlotVector<ImageView> slot_image_views;
    SlotVector<ImageAlloc> slot_image_allocs;
    SlotVector<Sampler> slot_samplers;
    SlotVector<Framebuffer> slot_framebuffers;

    // TODO: This data structure is not optimal and it should be reworked
    std::vector<ImageId> uncommitted_downloads;
    std::queue<std::vector<ImageId>> committed_downloads;

    static constexpr size_t TICKS_TO_DESTROY = 6;
    DelayedDestructionRing<Image, TICKS_TO_DESTROY> sentenced_images;
    DelayedDestructionRing<ImageView, TICKS_TO_DESTROY> sentenced_image_view;
    DelayedDestructionRing<Framebuffer, TICKS_TO_DESTROY> sentenced_framebuffers;

    std::unordered_map<GPUVAddr, ImageAllocId> image_allocs_table;

    u64 modification_tick = 0;
    u64 frame_tick = 0;
};

} // namespace VideoCommon

#include "texture_cache.inl"
