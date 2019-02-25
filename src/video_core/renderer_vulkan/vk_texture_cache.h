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
#include "video_core/gpu.h"
#include "video_core/rasterizer_cache.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_image.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"
#include "video_core/surface.h"
#include "video_core/texture_cache.h"
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

using VideoCommon::SurfaceParams;
using VideoCore::Surface::ComponentType;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::SurfaceTarget;
using VideoCore::Surface::SurfaceType;

class CachedSurface;
class CachedView;
using Surface = CachedSurface*;
using View = CachedView*;

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

    bool IsFamiliar(const SurfaceParams& view_params) const;

    bool IsViewValid(const SurfaceParams& view_params, u32 layer, u32 level) const;

    bool IsDimensionValid(const SurfaceParams& view_params, u32 level) const;

    bool IsDepthValid(const SurfaceParams& view_params, u32 level) const;

    bool IsInBounds(const SurfaceParams& view_params, u32 layer, u32 level) const;

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

    vk::ImageView GetHandle(Tegra::Shader::TextureType texture_type,
                            Tegra::Texture::SwizzleSource x_source,
                            Tegra::Texture::SwizzleSource y_source,
                            Tegra::Texture::SwizzleSource z_source,
                            Tegra::Texture::SwizzleSource w_source, bool is_array);

    vk::ImageView GetHandle() {
        return GetHandle(Tegra::Shader::TextureType::Texture2D, Tegra::Texture::SwizzleSource::R,
                         Tegra::Texture::SwizzleSource::G, Tegra::Texture::SwizzleSource::B,
                         Tegra::Texture::SwizzleSource::A, false);
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
    using ViewCache = std::unordered_map<u32, UniqueImageView>;

    std::pair<std::reference_wrapper<ViewCache>, vk::ImageViewType> GetTargetCache(
        Tegra::Shader::TextureType texture_type, bool is_array);

    vk::ImageView GetOrCreateView(ViewCache& view_cache, vk::ImageViewType view_type,
                                  Tegra::Texture::SwizzleSource x_source,
                                  Tegra::Texture::SwizzleSource y_source,
                                  Tegra::Texture::SwizzleSource z_source,
                                  Tegra::Texture::SwizzleSource w_source);

    static u32 GetViewCacheKey(Tegra::Texture::SwizzleSource x_source,
                               Tegra::Texture::SwizzleSource y_source,
                               Tegra::Texture::SwizzleSource z_source,
                               Tegra::Texture::SwizzleSource w_source);

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
    ViewCache image_view_1d;
    ViewCache image_view_1d_array;
    ViewCache image_view_2d;
    ViewCache image_view_2d_array;
    ViewCache image_view_3d;
    ViewCache image_view_cube;
    ViewCache image_view_cube_array;
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
        VKExecutionContext exctx, const Tegra::Texture::FullTextureInfo& config);

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
