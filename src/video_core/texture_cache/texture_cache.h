// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <bit>
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
#include "video_core/dirty_flags.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/surface.h"
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

template <class P>
class TextureCache {
    static constexpr u64 PAGE_SHIFT = 20;

    static constexpr bool ENABLE_VALIDATION = P::ENABLE_VALIDATION;
    static constexpr bool FRAMEBUFFER_BLITS = P::FRAMEBUFFER_BLITS;

    static constexpr ImageViewId NULL_IMAGE_VIEW_ID{0};

    using Runtime = typename P::Runtime;
    using Image = typename P::Image;
    using ImageAlloc = typename P::ImageAlloc;
    using ImageView = typename P::ImageView;
    using Sampler = typename P::Sampler;
    using Framebuffer = typename P::Framebuffer;

    /// Descriptor table of a class
    struct ClassDescriptorTables {
        std::vector<TICEntry> tic_entries; ///< Image entries
        std::vector<TSCEntry> tsc_entries; ///< Sampler entries

        std::vector<SamplerId> samplers;          ///< Samplers matching @sa tsc_entries
        std::vector<TSCEntry> stored_tsc_entries; ///< Sampler keys stored in @sa samplers
    };

    struct BlitImages {
        ImageId dst_id;
        ImageId src_id;
        PixelFormat dst_format;
        PixelFormat src_format;
    };

public:
    explicit TextureCache(Runtime&, VideoCore::RasterizerInterface&, Tegra::Engines::Maxwell3D&,
                          Tegra::Engines::KeplerCompute&, Tegra::MemoryManager&);

    void ImplicitDescriptorInvalidations();

    int frame_no = 0;
    void TickFrame() {
        ++frame_no;
    }

    void FillImageViews(std::span<const TICEntry> entries, std::span<const u32> indices,
                        std::span<ImageView*> image_views) {
        const size_t num_indices = indices.size();
        ASSERT(num_indices <= image_views.size());
        do {
            has_deleted_images = false;
            for (size_t i = num_indices; i--;) {
                const ImageViewId image_view_id = FindImageView(entries[indices[i]]);
                ImageView* const image_view = &slot_image_views[image_view_id];
                image_views[i] = image_view;
                if (image_view_id != NULL_IMAGE_VIEW_ID) {
                    UpdateImageContents(slot_images[image_view->image_id]);
                }
            }
        } while (has_deleted_images);
    }

    void FillGraphicsImageViews(std::span<const u32> indices, std::span<ImageView*> image_views) {
        FillImageViews(tables_3d.tic_entries, indices, image_views);
    }

    void FillComputeImageViews(std::span<const u32> indices, std::span<ImageView*> image_views) {
        FillImageViews(tables_compute.tic_entries, indices, image_views);
    }

    /**
     * Get the sampler from the graphics descriptor table in the specified index
     *
     * @param index Index in elements of the sampler
     * @returns     Sampler in the specified index
     *
     * @pre @a index is less than the size of the descriptor table
     */
    Sampler* GetGraphicsSampler(size_t index);

    /**
     * Get the sampler from the compute descriptor table in the specified index
     *
     * @param index Index in elements of the sampler
     * @returns     Sampler in the specified index
     *
     * @pre @a index is less than the size of the descriptor table
     */
    Sampler* GetComputeSampler(size_t index);

    /*
     * Update bound render targets and upload memory if necessary
     */
    void UpdateRenderTargets();

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
     * Invalidate the image descriptor tables
     */
    void InvalidateImageDescriptorTable();

    /**
     * Invalidate the sampler descriptor tables
     */
    void InvalidateSamplerDescriptorTable();

    /**
     * Tries to find a cached image view in the given CPU address
     *
     * @param cpu_addr Virtual CPU address
     * @returns        Pointer to image view
     */
    ImageView* TryFindFramebufferImageView(VAddr cpu_addr);

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

    void UpdateImageDescriptorTable(ClassDescriptorTables& tables, GPUVAddr tic_address,
                                    size_t num_tics);

    /**
     * Update a sampler descriptor table
     *
     * @param tables      Container of the table to invalidate
     * @param tsc_address Address of the descriptor table
     * @param num_tscs    Number of entries in the descriptor table
     */
    void UpdateSamplerDescriptorTable(ClassDescriptorTables& tables, GPUVAddr tsc_address,
                                      size_t num_tscs);

    Framebuffer* GetFramebuffer(const RenderTargets& key);

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
                                               VAddr cpu_addr, RelaxedOptions options);

    [[nodiscard]] BlitImages GetBlitImages(const Tegra::Engines::Fermi2D::Surface& dst,
                                           const Tegra::Engines::Fermi2D::Surface& src);

    /**
     * Find or create if necessary a sampler with the given properties
     *
     * @param stored_config Existing sampler config, it can be modified
     * @param existing_id   Existing sampler id, matches stored_config
     * @param config        Sampler properties
     * @returns             Sampler with the given properties
     */
    [[nodiscard]] SamplerId FindSampler(TSCEntry& stored_config, SamplerId existing_id,
                                        const TSCEntry& config);

    /**
     * Find or create an image view for the given color buffer index
     *
     * @param index Index of the color buffer to find
     * @returns     Image view of the given color buffer
     */
    [[nodiscard]] ImageViewId FindColorBuffer(size_t index);

    /**
     * Find or create an image view for the depth buffer
     *
     * @returns Image view for the depth buffer
     */
    [[nodiscard]] ImageViewId FindDepthBuffer();

    [[nodiscard]] ImageViewId FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr);

    template <typename Func>
    void ForEachImageInRegion(VAddr cpu_addr, size_t size, Func&& func);

    /**
     * Setup an image immediately created before calling this method
     */
    [[nodiscard]] ImageId CreateNewImage(const ImageInfo& info, GPUVAddr gpu_addr, VAddr cpu_addr);

    void InitializeNewImage(ImageId image_id);

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

    void TrackImage(Image& image);

    void UntrackImage(Image& image);

    void DeleteImage(ImageId image);

    void RemoveImageViewReferences(std::span<const ImageViewId> removed_views);

    void RemoveFramebuffers(std::span<const ImageViewId> removed_views);

    Runtime& runtime;
    VideoCore::RasterizerInterface& rasterizer;
    Tegra::Engines::Maxwell3D& maxwell3d;
    Tegra::Engines::KeplerCompute& kepler_compute;
    Tegra::MemoryManager& gpu_memory;

    ClassDescriptorTables tables_3d;
    ClassDescriptorTables tables_compute;

    RenderTargets render_targets;

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

    std::vector<Image> sentenced_images;
    std::vector<ImageView> sentenced_image_view;
    std::vector<Framebuffer> sentenced_framebuffers;

    std::unordered_map<GPUVAddr, ImageAllocId> image_allocs_table;
};

} // namespace VideoCommon

#include "texture_cache.inl"
