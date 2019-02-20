// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <tuple>
#include <unordered_map>
#include <boost/icl/interval_map.hpp>
#include "common/assert.h"
#include "common/common_types.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/math_util.h"
#include "video_core/rasterizer_cache.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_image.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"
#include "video_core/surface.h"
#include "video_core/textures/decoders.h"
#include "video_core/textures/texture.h"

namespace Core {
class System;
}

namespace VideoCore {
class RasterizerInterface;
}

namespace Vulkan {

class RasterizerVulkan;
class VKDevice;
class VKResourceManager;
class VKScheduler;

using VideoCore::Surface::ComponentType;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::SurfaceTarget;
using VideoCore::Surface::SurfaceType;

class CachedSurface;
class CachedView;
using Surface = CachedSurface*;
using View = CachedView*;

struct SurfaceParams {
    /// Creates SurfaceParams from a texture configuration
    static SurfaceParams CreateForTexture(Core::System& system,
                                          const Tegra::Texture::FullTextureInfo& config,
                                          const VKShader::SamplerEntry& entry);

    /// Creates SurfaceParams for a depth buffer configuration
    static SurfaceParams CreateForDepthBuffer(
        Core::System& system, u32 zeta_width, u32 zeta_height, Tegra::DepthFormat format,
        u32 block_width, u32 block_height, u32 block_depth,
        Tegra::Engines::Maxwell3D::Regs::InvMemoryLayout type);

    /// Creates SurfaceParams from a framebuffer configuration
    static SurfaceParams CreateForFramebuffer(Core::System& system, std::size_t index);

    /// Returns the total size of this surface in bytes, adjusted for compression
    std::size_t GetSizeInBytes(bool ignore_tiled = false) const {
        const u32 compression_factor{GetCompressionFactor(pixel_format)};
        const u32 bytes_per_pixel{GetBytesPerPixel(pixel_format)};
        const bool tiled{ignore_tiled ? false : is_tiled};
        const std::size_t uncompressed_size{Tegra::Texture::CalculateSize(
            tiled, bytes_per_pixel, width, height, depth, block_height, block_depth)};

        // Divide by compression_factor^2, as height and width are factored by this
        return uncompressed_size / (compression_factor * compression_factor);
    }

    /// Returns the size of this surface as an Vulkan texture in bytes
    std::size_t GetSizeInBytesVK() const {
        return GetSizeInBytes(true);
    }

    /// Initializes parameters for caching, should be called after everything has been initialized
    void CalculateSizes();

    vk::ImageCreateInfo CreateInfo(const VKDevice& device) const;

    bool is_tiled;
    u32 block_width;
    u32 block_height;
    u32 block_depth;
    u32 tile_width_spacing;
    PixelFormat pixel_format;
    ComponentType component_type;
    SurfaceType type;
    SurfaceTarget target;
    u32 width;
    u32 height;
    u32 depth;
    u32 unaligned_height;

    // Cached data
    std::size_t size_in_bytes;
    std::size_t size_in_bytes_vk;
};

struct SurfaceReserveKey : Common::HashableStruct<SurfaceParams> {
    static SurfaceReserveKey Create(const SurfaceParams& params) {
        SurfaceReserveKey res;
        res.state = params;
        // res.state.identity = {}; // Ignore the origin of the texture
        // res.state.rt = {};       // Ignore rt config in caching
        return res;
    }
};

struct ViewKey {
    u32 base_layer;
    u32 layers;
    u32 base_level;
    u32 levels;

    bool operator==(const ViewKey& rhs) const {
        return std::tie(base_layer, layers, base_level, levels) ==
               std::tie(rhs.base_layer, rhs.layers, rhs.base_level, rhs.levels);
    }
};

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::SurfaceReserveKey> {
    std::size_t operator()(const Vulkan::SurfaceReserveKey& k) const {
        return k.Hash();
    }
};

template <>
struct hash<Vulkan::ViewKey> {
    std::size_t operator()(const Vulkan::ViewKey& key) const {
        if constexpr (sizeof(std::size_t) >= sizeof(u64)) {
            return key.base_layer ^ static_cast<u64>(key.layers) << 16 ^
                   static_cast<u64>(key.base_level) << 32 ^ static_cast<u64>(key.levels) << 48;
        } else {
            return key.base_layer ^ key.layers << 8 ^ key.base_level << 16 ^ key.levels << 24;
        }
    }
};

} // namespace std

namespace Vulkan {

class CachedSurface : public VKImage {
public:
    explicit CachedSurface(Core::System& system, const VKDevice& device,
                           VKResourceManager& resource_manager, VKMemoryManager& memory_manager,
                           VKScheduler& sched, const SurfaceParams& params);
    ~CachedSurface();

    // Read/Write data in Switch memory to/from vk_buffer
    void LoadVKBuffer();
    VKExecutionContext FlushVKBuffer(VKExecutionContext exctx);

    // Upload data in vk_buffer to this surface's texture
    VKExecutionContext UploadVKTexture(VKExecutionContext exctx);

    void Transition(vk::CommandBuffer cmdbuf, vk::ImageLayout layout,
                    vk::PipelineStageFlags stage_flags, vk::AccessFlags access_flags);

    View TryGetView(const SurfaceParams& rhs);

    VAddr GetAddress() const {
        return address;
    }

    std::size_t GetSizeInBytes() const {
        return cached_size_in_bytes;
    }

    void MarkAsModified(bool is_modified_) {
        is_modified = is_modified_;
    }

    const SurfaceParams& GetSurfaceParams() const {
        return params;
    }

    View GetView(const SurfaceParams& rhs) {
        const View view = TryGetView(rhs);
        ASSERT(view != nullptr);
        return view;
    }

    bool IsFamiliar(const SurfaceParams& rhs) const {
        return std::tie(params.is_tiled, params.block_width, params.block_height,
                        params.block_depth, params.tile_width_spacing, params.pixel_format,
                        params.component_type, params.type, params.width, params.height,
                        params.depth, params.unaligned_height) ==
               std::tie(rhs.is_tiled, rhs.block_width, rhs.block_height, rhs.block_depth,
                        rhs.tile_width_spacing, rhs.pixel_format, rhs.component_type, rhs.type,
                        rhs.width, rhs.height, rhs.depth, rhs.unaligned_height);
    }

    void Register(VAddr address_) {
        ASSERT(!is_registered);
        is_registered = true;
        address = address_;
    }

    void Unregister() {
        ASSERT(is_registered);
        is_registered = false;
    }

    bool IsRegistered() const {
        return is_registered;
    }

private:
    View GetView(u32 base_layer, u32 layers, u32 base_level, u32 levels);

    vk::BufferImageCopy GetBufferImageCopy() const;

    vk::ImageSubresourceRange GetImageSubresourceRange() const;

    const VKDevice& device;
    VKResourceManager& resource_manager;
    VKMemoryManager& memory_manager;
    VKScheduler& sched;
    const SurfaceParams params;
    const std::size_t buffer_size;

    VKMemoryCommit image_commit;

    UniqueBuffer buffer;
    VKMemoryCommit buffer_commit;
    u8* vk_buffer{};

    std::unordered_map<ViewKey, std::unique_ptr<CachedView>> views;

    VAddr address{};
    std::size_t cached_size_in_bytes{};
    bool is_modified{};
    bool is_registered{};
};

class CachedView {
public:
    CachedView(const VKDevice& device, Surface surface, u32 base_layer, u32 layers, u32 base_level,
               u32 levels);
    ~CachedView();

    vk::ImageView GetHandle(Tegra::Shader::TextureType texture_type, bool is_array);

    vk::ImageView GetHandle() {
        return GetHandle(Tegra::Shader::TextureType::Texture2D, false);
    }

    u32 GetWidth() const {
        // return params.GetMipWidth(level);
        return params.width;
    }

    u32 GetHeight() const {
        // return params.GetMipHeight(level);
        return params.height;
    }

    vk::Image GetImage() const {
        return image;
    }

    vk::ImageSubresourceRange GetImageSubresourceRange() const {
        return {aspect_mask, base_level, levels, base_layer, layers};
    }

    void Transition(vk::CommandBuffer cmdbuf, vk::ImageLayout new_layout,
                    vk::PipelineStageFlags new_stage_mask, vk::AccessFlags new_access) const {
        surface->Transition(cmdbuf, new_layout, new_stage_mask, new_access);
    }

    void MarkAsModified(bool is_modified) const {
        surface->MarkAsModified(is_modified);
    }

private:
    // Store a copy of these values to avoid double dereference when reading them
    const SurfaceParams params;
    const vk::Image image;
    const vk::ImageAspectFlags aspect_mask;

    const VKDevice& device;
    const Surface surface;
    const u32 base_layer;
    const u32 layers;
    const u32 base_level;
    const u32 levels;
    UniqueImageView image_view_2d;
    UniqueImageView image_view_2d_array;
};

class VKTextureCache {
public:
    explicit VKTextureCache(Core::System& system, VideoCore::RasterizerInterface& rasterizer,
                            const VKDevice& device, VKResourceManager& resource_manager,
                            VKMemoryManager& memory_manager, VKScheduler& sched);
    ~VKTextureCache();

    void InvalidateRegion(VAddr address, std::size_t size);

    /// Get a surface based on the texture configuration
    [[nodiscard]] std::tuple<View, VKExecutionContext> GetTextureSurface(
        VKExecutionContext exctx, const Tegra::Texture::FullTextureInfo& config,
        const VKShader::SamplerEntry& entry);

    /// Get the depth surface based on the framebuffer configuration
    [[nodiscard]] std::tuple<View, VKExecutionContext> GetDepthBufferSurface(
        VKExecutionContext exctx, bool preserve_contents);

    /// Get the color surface based on the framebuffer configuration and the specified render target
    [[nodiscard]] std::tuple<View, VKExecutionContext> GetColorBufferSurface(
        VKExecutionContext exctx, std::size_t index, bool preserve_contents);

    /// Tries to find a framebuffer using on the provided CPU address
    [[nodiscard]] Surface TryFindFramebufferSurface(VAddr address) const;

private:
    [[nodiscard]] VKExecutionContext LoadSurface(VKExecutionContext exctx, const Surface& surface);

    [[nodiscard]] std::tuple<View, VKExecutionContext> GetSurfaceView(VKExecutionContext exctx,
                                                                      VAddr address,
                                                                      const SurfaceParams& params,
                                                                      bool preserve_contents);

    [[nodiscard]] std::tuple<View, VKExecutionContext> LoadSurfaceView(VKExecutionContext exctx,
                                                                       VAddr address,
                                                                       const SurfaceParams& params,
                                                                       bool preserve_contents);

    [[nodiscard]] std::tuple<View, VKExecutionContext> TryFastGetSurfaceView(
        VKExecutionContext exctx, VAddr address, const SurfaceParams& params,
        bool preserve_contents, const std::vector<Surface>& overlaps);

    [[nodiscard]] std::tuple<View, VKExecutionContext> FastCopySurface(
        VKExecutionContext exctx, Surface src_surface, VAddr address,
        const SurfaceParams& dst_params);

    [[nodiscard]] std::vector<Surface> GetSurfacesInRegion(VAddr address, std::size_t size) const;

    void Register(Surface surface, VAddr address);

    void Unregister(Surface surface);

    Surface GetUncachedSurface(const SurfaceParams& params);

    void ReserveSurface(std::unique_ptr<CachedSurface> surface);

    Surface TryGetReservedSurface(const SurfaceParams& params);

    Core::System& system;
    VideoCore::RasterizerInterface& rasterizer;
    const VKDevice& device;
    VKResourceManager& resource_manager;
    VKMemoryManager& memory_manager;
    VKScheduler& sched;

    boost::icl::interval_map<VAddr, std::set<Surface>> registered_surfaces;

    /// The surface reserve is a "backup" cache, this is where we put unique surfaces that have
    /// previously been used. This is to prevent surfaces from being constantly created and
    /// destroyed when used with different surface parameters.
    std::unordered_map<SurfaceReserveKey, std::list<std::unique_ptr<CachedSurface>>>
        surface_reserve;
};

} // namespace Vulkan
