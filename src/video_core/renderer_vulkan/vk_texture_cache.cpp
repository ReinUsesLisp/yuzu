// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>

#include "common/alignment.h"
#include "common/assert.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/morton.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/surface.h"
#include "video_core/textures/convert.h"

namespace Vulkan {

using VideoCore::MortonSwizzle;
using VideoCore::MortonSwizzleMode;

using Tegra::Texture::SwizzleSource;

static vk::ImageType SurfaceTargetToImageVK(SurfaceTarget target) {
    switch (target) {
    case SurfaceTarget::Texture1D:
    case SurfaceTarget::Texture1DArray:
        return vk::ImageType::e1D;
    case SurfaceTarget::Texture2D:
    case SurfaceTarget::Texture2DArray:
    case SurfaceTarget::TextureCubemap:
    case SurfaceTarget::TextureCubeArray:
        return vk::ImageType::e2D;
    case SurfaceTarget::Texture3D:
        return vk::ImageType::e3D;
    }
    UNREACHABLE_MSG("Unknown texture target={}", static_cast<u32>(target));
    return {};
}

static vk::ImageAspectFlags PixelFormatToImageAspect(PixelFormat pixel_format) {
    if (pixel_format < PixelFormat::MaxColorFormat) {
        return vk::ImageAspectFlagBits::eColor;
    } else if (pixel_format < PixelFormat::MaxDepthFormat) {
        return vk::ImageAspectFlagBits::eDepth;
    } else if (pixel_format < PixelFormat::MaxDepthStencilFormat) {
        return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    } else {
        UNREACHABLE_MSG("Invalid pixel format={}", static_cast<u32>(pixel_format));
        return vk::ImageAspectFlagBits::eColor;
    }
}

static vk::ImageCreateInfo GenerateImageCreateInfo(const VKDevice& device,
                                                   const SurfaceParams& params) {
    constexpr auto sample_count = vk::SampleCountFlagBits::e1;
    constexpr auto tiling = vk::ImageTiling::eOptimal;

    const auto [format, attachable] = MaxwellToVK::SurfaceFormat(
        device, FormatType::Optimal, params.GetPixelFormat(), params.GetComponentType());

    auto image_usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                       vk::ImageUsageFlagBits::eTransferSrc;
    if (attachable) {
        image_usage |= params.IsPixelFormatZeta() ? vk::ImageUsageFlagBits::eDepthStencilAttachment
                                                  : vk::ImageUsageFlagBits::eColorAttachment;
    }

    vk::ImageCreateFlags flags;
    vk::Extent3D extent;
    switch (params.GetTarget()) {
    case SurfaceTarget::TextureCubemap:
    case SurfaceTarget::TextureCubeArray:
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
        [[fallthrough]];
    case SurfaceTarget::Texture1D:
    case SurfaceTarget::Texture1DArray:
    case SurfaceTarget::Texture2D:
    case SurfaceTarget::Texture2DArray:
        extent = vk::Extent3D(params.GetWidth(), params.GetHeight(), 1);
        break;
    case SurfaceTarget::Texture3D:
        extent = vk::Extent3D(params.GetWidth(), params.GetHeight(), params.GetDepth());
        break;
    default:
        UNREACHABLE_MSG("Unknown surface target={}", static_cast<u32>(params.GetTarget()));
        break;
    }

    return vk::ImageCreateInfo(flags, SurfaceTargetToImageVK(params.GetTarget()), format, extent,
                               params.GetNumLevels(), params.GetNumLayers(), sample_count, tiling,
                               image_usage, vk::SharingMode::eExclusive, 0, nullptr,
                               vk::ImageLayout::eUndefined);
}

static void SwizzleFunc(MortonSwizzleMode mode, u8* memory, const SurfaceParams& params, u8* buffer,
                        u32 level) {
    const u32 width = params.GetMipWidth(level);
    const u32 height = params.GetMipHeight(level);
    const u32 block_height = params.GetMipBlockHeight(level);
    const u32 block_depth = params.GetMipBlockDepth(level);

    std::size_t guest_offset = params.GetGuestMipmapLevelOffset(level);
    if (params.IsLayered()) {
        std::size_t host_offset = 0;
        const std::size_t guest_stride = params.GetGuestLayerSize();
        const std::size_t host_stride = params.GetHostLayerSize(level);
        for (u32 layer = 0; layer < params.GetNumLayers(); layer++) {
            MortonSwizzle(mode, params.GetPixelFormat(), width, block_height, height, block_depth,
                          1, params.GetTileWidthSpacing(), buffer + host_offset,
                          memory + guest_offset);
            guest_offset += guest_stride;
            host_offset += host_stride;
        }
    } else {
        MortonSwizzle(mode, params.GetPixelFormat(), width, block_height, height, block_depth,
                      params.GetMipDepth(level), params.GetTileWidthSpacing(), buffer,
                      memory + guest_offset);
    }
}

CachedSurface::CachedSurface(Core::System& system, const VKDevice& device,
                             VKResourceManager& resource_manager, VKMemoryManager& memory_manager,
                             VKScheduler& scheduler, const SurfaceParams& params)
    : VKImage(device, GenerateImageCreateInfo(device, params),
              PixelFormatToImageAspect(params.GetPixelFormat())),
      SurfaceBase(params), system{system}, device{device}, resource_manager{resource_manager},
      memory_manager{memory_manager}, scheduler{scheduler} {
    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();

    image_commit = memory_manager.Commit(GetHandle(), false);

    const vk::BufferCreateInfo buffer_ci(
        {}, std::max(params.GetGuestSizeInBytes(), params.GetHostSizeInBytes()),
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive, 0, nullptr);
    buffer = dev.createBufferUnique(buffer_ci, nullptr, dld);
    buffer_commit = memory_manager.Commit(*buffer, true);
    vk_buffer = buffer_commit->GetData();
}

CachedSurface::~CachedSurface() = default;

void CachedSurface::LoadBuffer() {
    if (params.IsTiled()) {
        ASSERT_MSG(params.GetBlockWidth() == 1, "Block width is defined as {} on texture target {}",
                   params.GetBlockWidth(), static_cast<u32>(params.GetTarget()));
        for (u32 level = 0; level < params.GetNumLevels(); ++level) {
            u8* buffer = vk_buffer + params.GetHostMipmapLevelOffset(level);
            SwizzleFunc(MortonSwizzleMode::MortonToLinear, GetHostPtr(), params, buffer, level);
        }
    } else {
        ASSERT_MSG(params.GetNumLevels() == 1, "Linear mipmap loading is not implemented");
        const u32 bpp = GetFormatBpp(params.GetPixelFormat()) / CHAR_BIT;
        const u32 copy_size = params.GetWidth() * bpp;
        if (params.GetPitch() == copy_size) {
            std::memcpy(vk_buffer, GetHostPtr(), params.GetHostSizeInBytes());
        } else {
            const u8* start = GetHostPtr();
            u8* write_to = vk_buffer;
            for (u32 h = params.GetHeight(); h > 0; h--) {
                std::memcpy(write_to, start, copy_size);
                start += params.GetPitch();
                write_to += copy_size;
            }
        }
    }

    for (u32 level = 0; level < params.GetNumLevels(); ++level) {
        // Convert ASTC just when the format is not supported
        const bool convert_astc = VideoCore::Surface::IsPixelFormatASTC(params.GetPixelFormat()) &&
                                  GetFormat() == vk::Format::eA8B8G8R8UnormPack32;

        Tegra::Texture::ConvertFromGuestToHost(vk_buffer + params.GetHostMipmapLevelOffset(level),
                                               params.GetPixelFormat(), params.GetMipWidth(level),
                                               params.GetMipHeight(level),
                                               params.GetMipDepth(level), convert_astc, true);
    }
}

VKExecutionContext CachedSurface::FlushBuffer(VKExecutionContext exctx) {
    if (!IsModified()) {
        return exctx;
    }

    const auto cmdbuf = exctx.GetCommandBuffer();
    FullTransition(cmdbuf, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead,
                   vk::ImageLayout::eTransferSrcOptimal);

    const auto& dld = device.GetDispatchLoader();
    // TODO(Rodrigo): Do this in a single copy
    for (u32 level = 0; level < params.GetNumLevels(); ++level) {
        cmdbuf.copyImageToBuffer(GetHandle(), vk::ImageLayout::eTransferSrcOptimal, *buffer,
                                 {GetBufferImageCopy(level)}, dld);
    }
    exctx = scheduler.Finish();

    UNIMPLEMENTED_IF(!params.IsTiled());
    ASSERT_MSG(params.GetBlockWidth() == 1, "Block width is defined as {}", params.GetBlockWidth());
    for (u32 level = 0; level < params.GetNumLevels(); ++level) {
        u8* buffer = vk_buffer + params.GetHostMipmapLevelOffset(level);
        SwizzleFunc(MortonSwizzleMode::LinearToMorton, GetHostPtr(), params, buffer, level);
    }

    return exctx;
}

VKExecutionContext CachedSurface::UploadTexture(VKExecutionContext exctx) {
    const auto cmdbuf = exctx.GetCommandBuffer();
    FullTransition(cmdbuf, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite,
                   vk::ImageLayout::eTransferDstOptimal);

    for (u32 level = 0; level < params.GetNumLevels(); ++level) {
        vk::BufferImageCopy copy = GetBufferImageCopy(level);
        const auto& dld = device.GetDispatchLoader();
        if (GetAspectMask() ==
            (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) {
            vk::BufferImageCopy depth = copy;
            vk::BufferImageCopy stencil = copy;
            depth.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
            stencil.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eStencil;
            cmdbuf.copyBufferToImage(*buffer, GetHandle(), vk::ImageLayout::eTransferDstOptimal,
                                     {depth, stencil}, dld);
        } else {
            cmdbuf.copyBufferToImage(*buffer, GetHandle(), vk::ImageLayout::eTransferDstOptimal,
                                     {copy}, dld);
        }
    }
    return exctx;
}

std::unique_ptr<CachedView> CachedSurface::CreateView(const ViewKey& view_key) {
    return std::make_unique<CachedView>(device, this, view_key);
}

vk::BufferImageCopy CachedSurface::GetBufferImageCopy(u32 level) const {
    const u32 vk_depth =
        params.GetTarget() == SurfaceTarget::Texture3D ? params.GetMipDepth(level) : 1;
    return vk::BufferImageCopy(params.GetHostMipmapLevelOffset(level), 0, 0,
                               {GetAspectMask(), level, 0, params.GetNumLayers()}, {0, 0, 0},
                               {params.GetMipWidth(level), params.GetMipHeight(level), vk_depth});
}

vk::ImageSubresourceRange CachedSurface::GetImageSubresourceRange() const {
    return {GetAspectMask(), 0, params.GetNumLevels(), 0, params.GetNumLayers()};
}

CachedView::CachedView(const VKDevice& device, Surface surface, const ViewKey& key)
    : params{surface->GetSurfaceParams()}, image{surface->GetHandle()},
      aspect_mask{surface->GetAspectMask()}, device{device}, surface{surface},
      base_layer{key.base_layer}, num_layers{key.num_layers}, base_level{key.base_level},
      num_levels{key.num_levels} {};

CachedView::~CachedView() = default;

vk::ImageView CachedView::GetHandle(Tegra::Shader::TextureType texture_type, SwizzleSource x_source,
                                    SwizzleSource y_source, SwizzleSource z_source,
                                    SwizzleSource w_source, bool is_array) {
    const auto [view_cache, image_view_type] = GetTargetCache(texture_type, is_array);
    return GetOrCreateView(view_cache, image_view_type, x_source, y_source, z_source, w_source);
}

bool CachedView::IsOverlapping(const CachedView* rhs) const {
    // TODO(Rodrigo): Also test for layer and mip level overlaps.
    return surface == rhs->surface;
}

std::pair<std::reference_wrapper<CachedView::ViewCache>, vk::ImageViewType>
CachedView::GetTargetCache(Tegra::Shader::TextureType texture_type, bool is_array) {
    if (is_array) {
        switch (texture_type) {
        case Tegra::Shader::TextureType::Texture1D:
            return {image_view_1d_array, vk::ImageViewType::e1DArray};
        case Tegra::Shader::TextureType::Texture2D:
            return {image_view_2d_array, vk::ImageViewType::e2DArray};
        case Tegra::Shader::TextureType::Texture3D:
            UNREACHABLE_MSG("Arrays of 3D textures do not exist on Maxwell");
            break;
        case Tegra::Shader::TextureType::TextureCube:
            return {image_view_cube_array, vk::ImageViewType::eCubeArray};
        }
    } else {
        switch (texture_type) {
        case Tegra::Shader::TextureType::Texture1D:
            return {image_view_1d, vk::ImageViewType::e1D};
        case Tegra::Shader::TextureType::Texture2D:
            return {image_view_2d, vk::ImageViewType::e2D};
        case Tegra::Shader::TextureType::Texture3D:
            return {image_view_3d, vk::ImageViewType::e3D};
        case Tegra::Shader::TextureType::TextureCube:
            return {image_view_cube, vk::ImageViewType::eCube};
        }
    }
    UNREACHABLE();
    return {image_view_2d, vk::ImageViewType::e2D};
}

vk::ImageView CachedView::GetOrCreateView(ViewCache& view_cache, vk::ImageViewType view_type,
                                          SwizzleSource x_source, SwizzleSource y_source,
                                          SwizzleSource z_source, SwizzleSource w_source) {
    const u32 view_key = GetViewCacheKey(x_source, y_source, z_source, w_source);
    const auto [entry, is_cache_miss] = view_cache.try_emplace(view_key);
    auto& image_view = entry->second;
    if (!is_cache_miss) {
        return *image_view;
    }

    const vk::ImageViewCreateInfo image_view_ci(
        {}, surface->GetHandle(), view_type, surface->GetFormat(),
        {MaxwellToVK::SwizzleSource(x_source), MaxwellToVK::SwizzleSource(y_source),
         MaxwellToVK::SwizzleSource(z_source), MaxwellToVK::SwizzleSource(w_source)},
        GetImageSubresourceRange());

    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    return *(image_view = dev.createImageViewUnique(image_view_ci, nullptr, dld));
}

u32 CachedView::GetViewCacheKey(SwizzleSource x_source, SwizzleSource y_source,
                                SwizzleSource z_source, SwizzleSource w_source) {
    return static_cast<u8>(x_source) | static_cast<u8>(y_source) << 8 |
           static_cast<u8>(z_source) << 16 | static_cast<u8>(w_source) << 24;
}

VKTextureCache::VKTextureCache(Core::System& system, VideoCore::RasterizerInterface& rasterizer,
                               const VKDevice& device, VKResourceManager& resource_manager,
                               VKMemoryManager& memory_manager, VKScheduler& scheduler)
    : TextureCache(system, rasterizer), device{device}, resource_manager{resource_manager},
      memory_manager{memory_manager}, scheduler{scheduler} {}

VKTextureCache::~VKTextureCache() = default;

std::tuple<View, VKExecutionContext> VKTextureCache::TryFastGetSurfaceView(
    VKExecutionContext exctx, VAddr cpu_addr, u8* host_ptr, const SurfaceParams& params,
    bool preserve_contents, const std::vector<Surface>& overlaps) {
    if (overlaps.size() > 1) {
        return {{}, exctx};
    }
    const Surface old_surface = overlaps[0];
    const auto& old_params = old_surface->GetSurfaceParams();

    if (old_params.GetTarget() == params.GetTarget() && old_params.GetType() == params.GetType() &&
        old_params.GetDepth() == params.GetDepth() && params.GetDepth() == 1 &&
        GetFormatBpp(old_params.GetPixelFormat()) == GetFormatBpp(params.GetPixelFormat())) {
        return FastCopySurface(exctx, old_surface, cpu_addr, host_ptr, params);
    }

    return {{}, exctx};
}

std::unique_ptr<CachedSurface> VKTextureCache::CreateSurface(const SurfaceParams& params) {
    return std::make_unique<CachedSurface>(system, device, resource_manager, memory_manager,
                                           scheduler, params);
}

std::tuple<View, VKExecutionContext> VKTextureCache::FastCopySurface(
    VKExecutionContext exctx, Surface src_surface, VAddr cpu_addr, u8* host_ptr,
    const SurfaceParams& dst_params) {
    const auto& src_params = src_surface->GetSurfaceParams();
    const u32 width{std::min(src_params.GetWidth(), dst_params.GetWidth())};
    const u32 height{std::min(src_params.GetHeight(), dst_params.GetHeight())};

    const Surface dst_surface = GetUncachedSurface(dst_params);
    Register(dst_surface, cpu_addr, host_ptr);

    const auto cmdbuf = exctx.GetCommandBuffer();
    src_surface->FullTransition(cmdbuf, vk::PipelineStageFlagBits::eTransfer,
                                vk::AccessFlagBits::eTransferRead,
                                vk::ImageLayout::eTransferSrcOptimal);
    dst_surface->FullTransition(cmdbuf, vk::PipelineStageFlagBits::eTransfer,
                                vk::AccessFlagBits::eTransferWrite,
                                vk::ImageLayout::eTransferDstOptimal);

    // TODO(Rodrigo): Copy mipmaps
    const auto& dld = device.GetDispatchLoader();
    const vk::ImageCopy copy({src_surface->GetAspectMask(), 0, 0, 1}, {0, 0, 0},
                             {dst_surface->GetAspectMask(), 0, 0, 1}, {0, 0, 0},
                             {width, height, 1});
    cmdbuf.copyImage(src_surface->GetHandle(), vk::ImageLayout::eTransferSrcOptimal,
                     dst_surface->GetHandle(), vk::ImageLayout::eTransferDstOptimal, {copy}, dld);

    dst_surface->MarkAsModified(true);

    return {dst_surface->GetView(cpu_addr, dst_params), exctx};
}

} // namespace Vulkan