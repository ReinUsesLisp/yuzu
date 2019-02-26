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
using VideoCore::Surface::ComponentTypeFromDepthFormat;
using VideoCore::Surface::ComponentTypeFromRenderTarget;
using VideoCore::Surface::ComponentTypeFromTexture;
using VideoCore::Surface::PixelFormatFromDepthFormat;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;
using VideoCore::Surface::PixelFormatFromTextureFormat;
using VideoCore::Surface::SurfaceTargetFromTextureType;

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

static VAddr GetAddressForTexture(Core::System& system,
                                  const Tegra::Texture::FullTextureInfo& config) {
    auto& memory_manager{system.GPU().MemoryManager()};
    const auto cpu_addr{memory_manager.GpuToCpuAddress(config.tic.Address())};
    ASSERT(cpu_addr);
    return cpu_addr.value_or(0);
}

static VAddr GetAddressForFramebuffer(Core::System& system, std::size_t index) {
    auto& memory_manager{system.GPU().MemoryManager()};
    const auto& config{system.GPU().Maxwell3D().regs.rt[index]};
    const auto cpu_addr{memory_manager.GpuToCpuAddress(
        config.Address() + config.base_layer * config.layer_stride * sizeof(u32))};
    ASSERT(cpu_addr);
    return *cpu_addr;
}

static vk::ImageCreateInfo CreateImageInfo(const VKDevice& device, const SurfaceParams& params) {
    constexpr auto sample_count = vk::SampleCountFlagBits::e1;
    constexpr auto tiling = vk::ImageTiling::eOptimal;

    const auto [format, attachable] = MaxwellToVK::SurfaceFormat(
        device, FormatType::Optimal, params.pixel_format, params.component_type);

    auto image_usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                       vk::ImageUsageFlagBits::eTransferSrc;
    if (attachable) {
        const bool is_zeta = params.pixel_format >= PixelFormat::MaxColorFormat &&
                             params.pixel_format < PixelFormat::MaxDepthStencilFormat;
        image_usage |= is_zeta ? vk::ImageUsageFlagBits::eDepthStencilAttachment
                               : vk::ImageUsageFlagBits::eColorAttachment;
    }

    vk::ImageCreateFlags flags;
    vk::Extent3D extent;
    switch (params.target) {
    case SurfaceTarget::TextureCubemap:
    case SurfaceTarget::TextureCubeArray:
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
        [[fallthrough]];
    case SurfaceTarget::Texture1D:
    case SurfaceTarget::Texture1DArray:
    case SurfaceTarget::Texture2D:
    case SurfaceTarget::Texture2DArray:
        extent = {params.width, params.height, 1};
        break;
    case SurfaceTarget::Texture3D:
        extent = {params.width, params.height, params.depth};
        break;
    default:
        UNREACHABLE_MSG("Unknown surface target={}", static_cast<u32>(params.target));
        break;
    }

    return vk::ImageCreateInfo(flags, SurfaceTargetToImageVK(params.target), format, extent,
                               params.levels_count, params.GetLayersCount(), sample_count, tiling,
                               image_usage, vk::SharingMode::eExclusive, 0, nullptr,
                               vk::ImageLayout::eUndefined);
}

static void SwizzleFunc(MortonSwizzleMode mode, VAddr address, const SurfaceParams& params,
                        u8* buffer, u32 level) {
    const u32 width = params.GetMipWidth(level);
    const u32 height = params.GetMipHeight(level);
    const u32 block_height = params.GetMipBlockHeight(level);
    const u32 block_depth = params.GetMipBlockDepth(level);

    std::size_t guest_offset = params.GetGuestMipmapLevelOffset(level);
    if (params.IsLayered()) {
        std::size_t host_offset = 0;
        const std::size_t guest_stride = params.GetGuestLayerMemorySize();
        const std::size_t host_stride = params.GetHostLayerSize(level);
        for (u32 layer = 0; layer < params.GetLayersCount(); layer++) {
            MortonSwizzle(mode, params.pixel_format, width, block_height, height, block_depth, 1,
                          params.tile_width_spacing, buffer + host_offset, host_stride,
                          address + guest_offset);
            guest_offset += guest_stride;
            host_offset += host_stride;
        }
    } else {
        MortonSwizzle(mode, params.pixel_format, width, block_height, height, block_depth,
                      params.GetMipDepth(level), params.tile_width_spacing, buffer, 0,
                      address + guest_offset);
    }
}

CachedSurface::CachedSurface(Core::System& system, const VKDevice& device,
                             VKResourceManager& resource_manager, VKMemoryManager& memory_manager,
                             VKScheduler& sched, const SurfaceParams& params)
    : VKImage(device, CreateImageInfo(device, params),
              PixelFormatToImageAspect(params.pixel_format)),
      SurfaceBase(params), device{device}, resource_manager{resource_manager},
      memory_manager{memory_manager}, sched{sched} {
    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();

    image_commit = memory_manager.Commit(GetHandle(), false);

    const vk::BufferCreateInfo buffer_ci({}, std::max(params.guest_size_in_bytes, params.host_size_in_bytes),
                                         vk::BufferUsageFlagBits::eTransferDst |
                                             vk::BufferUsageFlagBits::eTransferSrc,
                                         vk::SharingMode::eExclusive, 0, nullptr);
    buffer = dev.createBufferUnique(buffer_ci, nullptr, dld);
    buffer_commit = memory_manager.Commit(*buffer, true);
    vk_buffer = buffer_commit->GetData();
}

CachedSurface::~CachedSurface() = default;

void CachedSurface::LoadBuffer() {
    if (params.is_tiled) {
        ASSERT_MSG(params.block_width == 1, "Block width is defined as {} on texture target {}",
                   params.block_width, static_cast<u32>(params.target));

        for (u32 level = 0; level < params.levels_count; ++level) {
            u8* buffer = vk_buffer + params.GetHostMipmapLevelOffset(level);
            SwizzleFunc(MortonSwizzleMode::MortonToLinear, GetAddress(), params, buffer, level);
        }
    } else {
        ASSERT_MSG(params.levels_count == 1, "Linear mipmap loading is not implemented");
        const u32 bpp = GetFormatBpp(params.pixel_format) / CHAR_BIT;
        const u32 copy_size = params.width * bpp;
        if (params.pitch == copy_size) {
            std::memcpy(vk_buffer, Memory::GetPointer(GetAddress()), params.host_size_in_bytes);
        } else {
            const u8* start = Memory::GetPointer(GetAddress());
            u8* write_to = vk_buffer;
            for (u32 h = params.height; h > 0; h--) {
                std::memcpy(write_to, start, copy_size);
                start += params.pitch;
                write_to += copy_size;
            }
        }
    }

    for (u32 level = 0; level < params.levels_count; ++level) {
        Tegra::Texture::ConvertFromGuestToHost(vk_buffer + params.GetHostMipmapLevelOffset(level),
                                               params.pixel_format, params.GetMipWidth(level),
                                               params.GetMipHeight(level),
                                               params.GetMipDepth(level), false, true);
    }
}

VKExecutionContext CachedSurface::FlushBuffer(VKExecutionContext exctx) {
    if (!IsModified()) {
        return exctx;
    }

    const auto cmdbuf = exctx.GetCommandBuffer();
    Transition(cmdbuf, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer,
               vk::AccessFlagBits::eTransferRead);

    const auto& dld = device.GetDispatchLoader();
    // TODO(Rodrigo): Do this in a single copy
    for (u32 level = 0; level < params.levels_count; ++level) {
        cmdbuf.copyImageToBuffer(GetHandle(), vk::ImageLayout::eTransferSrcOptimal, *buffer,
                                 {GetBufferImageCopy(level)}, dld);
    }
    exctx = sched.Finish();

    UNIMPLEMENTED_IF(!params.is_tiled);
    ASSERT_MSG(params.block_width == 1, "Block width is defined as {}", params.block_width);
    for (u32 level = 0; level < params.levels_count; ++level) {
        u8* buffer = vk_buffer + params.GetHostMipmapLevelOffset(level);
        SwizzleFunc(MortonSwizzleMode::LinearToMorton, GetAddress(), params, buffer, level);
    }

    return exctx;
}

VKExecutionContext CachedSurface::UploadTexture(VKExecutionContext exctx) {
    const auto cmdbuf = exctx.GetCommandBuffer();
    Transition(cmdbuf, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer,
               vk::AccessFlagBits::eTransferWrite);

    for (u32 level = 0; level < params.levels_count; ++level) {
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

void CachedSurface::Transition(vk::CommandBuffer cmdbuf, vk::ImageLayout new_layout,
                               vk::PipelineStageFlags new_stage_mask, vk::AccessFlags new_access) {
    VKImage::Transition(cmdbuf, GetImageSubresourceRange(), new_layout, new_stage_mask, new_access);
}

std::unique_ptr<CachedView> CachedSurface::CreateView(const ViewKey& view_key) {
    return std::make_unique<CachedView>(device, this, view_key);
}

vk::BufferImageCopy CachedSurface::GetBufferImageCopy(u32 level) const {
    const u32 vk_depth = params.target == SurfaceTarget::Texture3D ? params.GetMipDepth(level) : 1;
    return vk::BufferImageCopy(params.GetHostMipmapLevelOffset(level), 0, 0,
                               {GetAspectMask(), level, 0, params.GetLayersCount()}, {0, 0, 0},
                               {params.GetMipWidth(level), params.GetMipHeight(level), vk_depth});
}

vk::ImageSubresourceRange CachedSurface::GetImageSubresourceRange() const {
    return {GetAspectMask(), 0, params.levels_count, 0, params.GetLayersCount()};
}

CachedView::CachedView(const VKDevice& device, Surface surface, const ViewKey& key)
    : params{surface->GetSurfaceParams()}, image{surface->GetHandle()},
      aspect_mask{surface->GetAspectMask()}, device{device}, surface{surface},
      base_layer{key.base_layer}, layers{key.layers},
      base_level{key.base_level}, levels{key.levels} {};

CachedView::~CachedView() = default;

vk::ImageView CachedView::GetHandle(Tegra::Shader::TextureType texture_type,
                                    Tegra::Texture::SwizzleSource x_source,
                                    Tegra::Texture::SwizzleSource y_source,
                                    Tegra::Texture::SwizzleSource z_source,
                                    Tegra::Texture::SwizzleSource w_source, bool is_array) {
    const auto [view_cache, image_view_type] = GetTargetCache(texture_type, is_array);
    return GetOrCreateView(view_cache, image_view_type, x_source, y_source, z_source, w_source);
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
                                          Tegra::Texture::SwizzleSource x_source,
                                          Tegra::Texture::SwizzleSource y_source,
                                          Tegra::Texture::SwizzleSource z_source,
                                          Tegra::Texture::SwizzleSource w_source) {
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

u32 CachedView::GetViewCacheKey(Tegra::Texture::SwizzleSource x_source,
                                Tegra::Texture::SwizzleSource y_source,
                                Tegra::Texture::SwizzleSource z_source,
                                Tegra::Texture::SwizzleSource w_source) {
    return static_cast<u8>(x_source) | static_cast<u8>(y_source) << 8 |
           static_cast<u8>(z_source) << 16 | static_cast<u8>(w_source) << 24;
}

VKTextureCache::VKTextureCache(Core::System& system, VideoCore::RasterizerInterface& rasterizer,
                               const VKDevice& device, VKResourceManager& resource_manager,
                               VKMemoryManager& memory_manager, VKScheduler& sched)
    : TextureCache(system, rasterizer), device{device}, resource_manager{resource_manager},
      memory_manager{memory_manager}, sched{sched} {}

VKTextureCache::~VKTextureCache() = default;

std::tuple<View, VKExecutionContext> VKTextureCache::TryFastGetSurfaceView(
    VKExecutionContext exctx, VAddr address, const SurfaceParams& params, bool preserve_contents,
    const std::vector<Surface>& overlaps) {
    if (overlaps.size() > 1) {
        return {{}, exctx};
    }
    const Surface old_surface = overlaps[0];
    const auto& old_params = old_surface->GetSurfaceParams();

    if (old_params.target == params.target && old_params.type == params.type &&
        old_params.depth == params.depth && params.depth == 1 &&
        GetFormatBpp(old_params.pixel_format) == GetFormatBpp(params.pixel_format)) {
        return FastCopySurface(exctx, old_surface, address, params);
    }

    return {{}, exctx};
}

std::unique_ptr<CachedSurface> VKTextureCache::CreateSurface(const SurfaceParams& params) {
    return std::make_unique<CachedSurface>(system, device, resource_manager, memory_manager, sched,
                                           params);
}

std::tuple<View, VKExecutionContext> VKTextureCache::FastCopySurface(
    VKExecutionContext exctx, Surface src_surface, VAddr address, const SurfaceParams& dst_params) {
    const auto& src_params = src_surface->GetSurfaceParams();
    const u32 width{std::min(src_params.width, dst_params.width)};
    const u32 height{std::min(src_params.height, dst_params.height)};

    const Surface dst_surface = GetUncachedSurface(dst_params);
    Register(dst_surface, address);

    const auto cmdbuf = exctx.GetCommandBuffer();
    src_surface->Transition(cmdbuf, vk::ImageLayout::eTransferSrcOptimal,
                            vk::PipelineStageFlagBits::eTransfer,
                            vk::AccessFlagBits::eTransferRead);
    dst_surface->Transition(cmdbuf, vk::ImageLayout::eTransferDstOptimal,
                            vk::PipelineStageFlagBits::eTransfer,
                            vk::AccessFlagBits::eTransferWrite);

    // TODO(Rodrigo): Copy mipmaps
    const auto& dld = device.GetDispatchLoader();
    const vk::ImageCopy copy({src_surface->GetAspectMask(), 0, 0, 1}, {0, 0, 0},
                             {dst_surface->GetAspectMask(), 0, 0, 1}, {0, 0, 0},
                             {width, height, 1});
    cmdbuf.copyImage(src_surface->GetHandle(), vk::ImageLayout::eTransferSrcOptimal,
                     dst_surface->GetHandle(), vk::ImageLayout::eTransferDstOptimal, {copy}, dld);

    dst_surface->MarkAsModified(true);

    return {dst_surface->GetView(address, dst_params), exctx};
}

} // namespace Vulkan