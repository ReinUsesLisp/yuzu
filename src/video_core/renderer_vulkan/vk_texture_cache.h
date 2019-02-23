// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <tuple>
#include <unordered_map>
#include <boost/functional/hash.hpp>
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

    u32 GetMipWidth(u32 level) const {
        return std::max(1U, width >> level);
    }

    u32 GetMipHeight(u32 level) const {
        return std::max(1U, height >> level);
    }

    u32 GetMipDepth(u32 level) const {
        return IsLayered() ? depth : std::max(1U, depth >> level);
    }

    u32 GetLayersCount() const {
        switch (target) {
        case SurfaceTarget::Texture1D:
        case SurfaceTarget::Texture2D:
        case SurfaceTarget::Texture3D:
            return 1;
        case SurfaceTarget::Texture1DArray:
        case SurfaceTarget::Texture2DArray:
        case SurfaceTarget::TextureCubemap:
        case SurfaceTarget::TextureCubeArray:
            return depth;
        }
        UNREACHABLE();
    }

    bool IsLayered() const {
        switch (target) {
        case SurfaceTarget::Texture1DArray:
        case SurfaceTarget::Texture2DArray:
        case SurfaceTarget::TextureCubeArray:
        case SurfaceTarget::TextureCubemap:
            return true;
        default:
            return false;
        }
    }

    // Auto block resizing algorithm from:
    // https://cgit.freedesktop.org/mesa/mesa/tree/src/gallium/drivers/nouveau/nv50/nv50_miptree.c
    u32 GetMipBlockHeight(u32 level) const {
        if (level == 0)
            return block_height;
        const u32 alt_height = GetMipHeight(level);
        const u32 h = GetDefaultBlockHeight(pixel_format);
        const u32 blocks_in_y = (alt_height + h - 1) / h;
        u32 block_height = 16;
        while (block_height > 1 && blocks_in_y <= block_height * 4) {
            block_height >>= 1;
        }
        return block_height;
    }

    u32 GetMipBlockDepth(u32 level) const {
        if (level == 0)
            return block_depth;
        if (target != SurfaceTarget::Texture3D)
            return 1;

        const u32 depth = GetMipDepth(level);
        u32 block_depth = 32;
        while (block_depth > 1 && depth * 2 <= block_depth) {
            block_depth >>= 1;
        }
        if (block_depth == 32 && GetMipBlockHeight(level) >= 4) {
            return 16;
        }
        return block_depth;
    }

    std::size_t GetGuestMipmapLevelOffset(u32 level) const {
        std::size_t offset = 0;
        for (u32 i = 0; i < level; i++) {
            offset += GetInnerMipmapMemorySize(i, false, IsLayered(), false);
        }
        return offset;
    }

    std::size_t GetHostMipmapLevelOffset(u32 level) const {
        std::size_t offset = 0;
        for (u32 i = 0; i < level; i++) {
            offset += GetInnerMipmapMemorySize(i, true, false, false);
        }
        return offset;
    }

    std::size_t GetGuestLayerMemorySize() const {
        return GetInnerMemorySize(false, true, false);
    }

    std::size_t GetHostLayerSize(u32 level) const {
        return GetInnerMipmapMemorySize(level, true, IsLayered(), false);
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
    u32 pitch;
    u32 unaligned_height;
    u32 levels_count;

    // Cached data
    std::size_t guest_size_in_bytes;
    std::size_t host_size_in_bytes;

private:
    std::size_t GetInnerMipmapMemorySize(u32 level, bool as_host_size, bool layer_only,
                                         bool uncompressed) const;
    std::size_t GetInnerMemorySize(bool as_host_size, bool layer_only, bool uncompressed) const;
};

struct SurfaceReserveKey : Common::HashableStruct<SurfaceParams> {
    static SurfaceReserveKey Create(const SurfaceParams& params) {
        SurfaceReserveKey res;
        res.state = params;
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
        std::size_t hash = 0;
        boost::hash_combine(hash, key.base_layer);
        boost::hash_combine(hash, key.layers);
        boost::hash_combine(hash, key.base_level);
        boost::hash_combine(hash, key.levels);
        return hash;
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

    View TryGetView(VAddr view_address, const SurfaceParams& view_params);

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

    View GetView(VAddr view_address, const SurfaceParams& view_params) {
        const View view = TryGetView(view_address, view_params);
        ASSERT(view != nullptr);
        return view;
    }

    bool IsFamiliar(const SurfaceParams& rhs) const;

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

    vk::BufferImageCopy GetBufferImageCopy(u32 level) const;

    vk::ImageSubresourceRange GetImageSubresourceRange() const;

    static std::map<u64, std::pair<u32, u32>> BuildViewOffsetMap(const SurfaceParams& params);

    const VKDevice& device;
    VKResourceManager& resource_manager;
    VKMemoryManager& memory_manager;
    VKScheduler& sched;
    const SurfaceParams params;
    const std::size_t buffer_size;
    const std::map<u64, std::pair<u32, u32>> view_offset_map;

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
        return params.GetMipWidth(base_level);
    }

    u32 GetHeight() const {
        return params.GetMipHeight(base_level);
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
    UniqueImageView image_view_1d;
    UniqueImageView image_view_1d_array;
    UniqueImageView image_view_2d;
    UniqueImageView image_view_2d_array;
    UniqueImageView image_view_3d;
    UniqueImageView image_view_cube;
    UniqueImageView image_view_cube_array;
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
