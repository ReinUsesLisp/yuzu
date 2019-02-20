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
using VideoCore::Surface::SurfaceFamily;
using VideoCore::Surface::SurfaceTarget;
using VideoCore::Surface::SurfaceType;

class CachedSurface;
class CachedView;
using Surface = CachedSurface*;
using View = const CachedView*;

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
    SurfaceFamily family;
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
    u32 layer;
    u32 level;

    bool operator==(const ViewKey& rhs) const {
        return std::tie(layer, level) == std::tie(rhs.layer, rhs.level);
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
            return key.layer | static_cast<u64>(key.level) << 32;
        } else {
            return key.layer ^ key.level;
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

    View TryGetView(const SurfaceParams& rhs) {
        if (params.width == rhs.width && params.height == rhs.height) {
            return GetView(0, 0);
        }
        // Unimplemented
        return nullptr;
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
    View GetView(u32 layer, u32 level);

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
    CachedView(const VKDevice& device, Surface surface, u32 layer, u32 level);
    ~CachedView();

    vk::ImageView GetHandle() const {
        return *image_view;
    }

    Surface GetSurface() const {
        return surface;
    }

    void MarkAsModified(bool is_modified) const {
        surface->MarkAsModified(is_modified);
    }

private:
    const VKDevice& device;
    Surface surface;
    UniqueImageView image_view;
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
