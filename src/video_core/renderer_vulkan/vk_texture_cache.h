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
using VideoCommon::ViewKey;
using VideoCore::Surface::ComponentType;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::SurfaceTarget;
using VideoCore::Surface::SurfaceType;

class CachedSurface;
class CachedView;
using Surface = CachedSurface*;
using View = CachedView*;

class CachedSurface : public VKImage,
                      public VideoCommon::SurfaceBase<CachedView, VKExecutionContext> {
public:
    explicit CachedSurface(Core::System& system, const VKDevice& device,
                           VKResourceManager& resource_manager, VKMemoryManager& memory_manager,
                           VKScheduler& sched, const SurfaceParams& params);
    ~CachedSurface();

    // Read/Write data in Switch memory to/from vk_buffer
    void LoadBuffer();

    //
    VKExecutionContext FlushBuffer(VKExecutionContext exctx);

    // Upload data in vk_buffer to this surface's texture
    VKExecutionContext UploadTexture(VKExecutionContext exctx);

    void FullTransition(vk::CommandBuffer cmdbuf, vk::PipelineStageFlags new_stage_mask,
                        vk::AccessFlags new_access, vk::ImageLayout new_layout) {
        Transition(cmdbuf, 0, params.GetLayersCount(), 0, params.levels_count, new_stage_mask,
                   new_access, new_layout);
    }

protected:
    std::unique_ptr<CachedView> CreateView(const ViewKey& view_key);

private:
    vk::BufferImageCopy GetBufferImageCopy(u32 level) const;

    vk::ImageSubresourceRange GetImageSubresourceRange() const;

    const VKDevice& device;
    VKResourceManager& resource_manager;
    VKMemoryManager& memory_manager;
    VKScheduler& sched;

    VKMemoryCommit image_commit;

    UniqueBuffer buffer;
    VKMemoryCommit buffer_commit;
    u8* vk_buffer{};
};

class CachedView {
public:
    CachedView(const VKDevice& device, Surface surface, const ViewKey& key);
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
        surface->Transition(cmdbuf, base_layer, layers, base_level, levels, new_stage_mask,
                            new_access, new_layout);
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

class VKTextureCache
    : public VideoCommon::TextureCache<CachedSurface, CachedView, VKExecutionContext> {
public:
    explicit VKTextureCache(Core::System& system, VideoCore::RasterizerInterface& rasterizer,
                            const VKDevice& device, VKResourceManager& resource_manager,
                            VKMemoryManager& memory_manager, VKScheduler& sched);
    ~VKTextureCache();

private:
    std::tuple<View, VKExecutionContext> TryFastGetSurfaceView(
        VKExecutionContext exctx, VAddr address, const SurfaceParams& params,
        bool preserve_contents, const std::vector<Surface>& overlaps);

    std::unique_ptr<CachedSurface> CreateSurface(const SurfaceParams& params);

    std::tuple<View, VKExecutionContext> FastCopySurface(VKExecutionContext exctx,
                                                         Surface src_surface, VAddr address,
                                                         const SurfaceParams& dst_params);

    const VKDevice& device;
    VKResourceManager& resource_manager;
    VKMemoryManager& memory_manager;
    VKScheduler& sched;
};

} // namespace Vulkan
