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
#include "video_core/textures/astc.h"

#pragma optimize("", off)

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
    case SurfaceTarget::Texture2D:
    case SurfaceTarget::Texture2DArray:
    case SurfaceTarget::TextureCubemap:
    case SurfaceTarget::TextureCubeArray:
        return vk::ImageType::e2D;
    default:
        UNIMPLEMENTED_MSG("Unimplemented texture family={}", static_cast<u32>(target));
        return vk::ImageType::e2D;
    }
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
    return *cpu_addr;
}

static VAddr GetAddressForFramebuffer(Core::System& system, std::size_t index) {
    auto& memory_manager{system.GPU().MemoryManager()};
    const auto& config{system.GPU().Maxwell3D().regs.rt[index]};
    const auto cpu_addr{memory_manager.GpuToCpuAddress(
        config.Address() + config.base_layer * config.layer_stride * sizeof(u32))};
    ASSERT(cpu_addr);
    return *cpu_addr;
}

/*static*/ SurfaceParams SurfaceParams::CreateForTexture(
    Core::System& system, const Tegra::Texture::FullTextureInfo& config,
    const VKShader::SamplerEntry& entry) {

    SurfaceParams params{};
    params.is_tiled = config.tic.IsTiled();
    params.block_width = params.is_tiled ? config.tic.BlockWidth() : 0,
    params.block_height = params.is_tiled ? config.tic.BlockHeight() : 0,
    params.block_depth = params.is_tiled ? config.tic.BlockDepth() : 0,
    params.tile_width_spacing = params.is_tiled ? (1 << config.tic.tile_width_spacing.Value()) : 1;
    // params.srgb_conversion = config.tic.IsSrgbConversionEnabled();
    params.pixel_format = PixelFormatFromTextureFormat(config.tic.format, config.tic.r_type.Value(),
                                                       false /*params.srgb_conversion*/);
    params.component_type = ComponentTypeFromTexture(config.tic.r_type.Value());
    params.type = GetFormatType(params.pixel_format);
    params.target = SurfaceTargetFromTextureType(config.tic.texture_type);
    params.width = Common::AlignUp(config.tic.Width(), GetCompressionFactor(params.pixel_format));
    params.height = Common::AlignUp(config.tic.Height(), GetCompressionFactor(params.pixel_format));
    params.depth = config.tic.Depth();
    if (params.target == SurfaceTarget::TextureCubemap ||
        params.target == SurfaceTarget::TextureCubeArray) {
        params.depth *= 6;
    }
    params.unaligned_height = config.tic.Height();
    params.levels_count = config.tic.max_mip_level + 1;

    params.CalculateSizes();
    return params;
}

/*static*/ SurfaceParams SurfaceParams::CreateForDepthBuffer(
    Core::System& system, u32 zeta_width, u32 zeta_height, Tegra::DepthFormat format,
    u32 block_width, u32 block_height, u32 block_depth,
    Tegra::Engines::Maxwell3D::Regs::InvMemoryLayout type) {

    SurfaceParams params{};
    params.is_tiled = type == Tegra::Engines::Maxwell3D::Regs::InvMemoryLayout::BlockLinear;
    params.block_width = 1 << std::min(block_width, 5U);
    params.block_height = 1 << std::min(block_height, 5U);
    params.block_depth = 1 << std::min(block_depth, 5U);
    params.tile_width_spacing = 1;
    params.pixel_format = PixelFormatFromDepthFormat(format);
    params.component_type = ComponentTypeFromDepthFormat(format);
    params.type = GetFormatType(params.pixel_format);
    // params.srgb_conversion = false;
    params.width = zeta_width;
    params.height = zeta_height;
    params.unaligned_height = zeta_height;
    params.target = SurfaceTarget::Texture2D;
    params.depth = 1;
    params.levels_count = 1;

    params.CalculateSizes();
    return params;
}

/*static*/ SurfaceParams SurfaceParams::CreateForFramebuffer(Core::System& system,
                                                             std::size_t index) {
    const auto& config{system.GPU().Maxwell3D().regs.rt[index]};
    SurfaceParams params{};

    params.is_tiled =
        config.memory_layout.type == Tegra::Engines::Maxwell3D::Regs::InvMemoryLayout::BlockLinear;
    params.block_width = 1 << config.memory_layout.block_width;
    params.block_height = 1 << config.memory_layout.block_height;
    params.block_depth = 1 << config.memory_layout.block_depth;
    params.tile_width_spacing = 1;
    params.pixel_format = PixelFormatFromRenderTargetFormat(config.format);
    // params.srgb_conversion = config.format == Tegra::RenderTargetFormat::BGRA8_SRGB ||
    //                         config.format == Tegra::RenderTargetFormat::RGBA8_SRGB;
    params.component_type = ComponentTypeFromRenderTarget(config.format);
    params.type = GetFormatType(params.pixel_format);
    params.width = config.width;
    params.height = config.height;
    params.unaligned_height = config.height;
    params.target = SurfaceTarget::Texture2D;
    params.depth = 1;
    params.levels_count = 1;

    params.CalculateSizes();
    return params;
}

/**
 * Helper function to perform software conversion (as needed) when loading a buffer from Switch
 * memory. This is for Maxwell pixel formats that cannot be represented as-is in Vulkan or with
 * typical desktop GPUs.
 */
static void ConvertFormatAsNeeded_LoadVKBuffer(u8* data, PixelFormat pixel_format, u32 width,
                                               u32 height, u32 depth) {
    switch (pixel_format) {
    case PixelFormat::ASTC_2D_4X4:
    case PixelFormat::ASTC_2D_8X8:
    case PixelFormat::ASTC_2D_8X5:
    case PixelFormat::ASTC_2D_5X4:
    case PixelFormat::ASTC_2D_5X5:
    case PixelFormat::ASTC_2D_4X4_SRGB:
    case PixelFormat::ASTC_2D_8X8_SRGB:
    case PixelFormat::ASTC_2D_8X5_SRGB:
    case PixelFormat::ASTC_2D_5X4_SRGB:
    case PixelFormat::ASTC_2D_5X5_SRGB:
    case PixelFormat::ASTC_2D_10X8:
    case PixelFormat::ASTC_2D_10X8_SRGB: {
        UNIMPLEMENTED();
        break;
    }
    case PixelFormat::S8Z24:
        UNIMPLEMENTED();
        break;
    }
}

void SurfaceParams::CalculateSizes() {
    guest_size_in_bytes = GetInnerMemorySize(false, false, false);

    // ASTC is uncompressed in software, in emulated as RGBA8
    if (IsPixelFormatASTC(pixel_format)) {
        host_size_in_bytes = width * height * depth * 4;
    } else {
        host_size_in_bytes = GetInnerMemorySize(true, false, false);
    }
}

constexpr u32 GetMipmapSize(bool uncompressed, u32 mip_size, u32 tile) {
    return uncompressed ? mip_size : std::max(1U, (mip_size + tile - 1) / tile);
}

std::size_t SurfaceParams::GetInnerMipmapMemorySize(u32 level, bool as_host_size, bool layer_only,
                                                    bool uncompressed) const {
    const bool tiled = as_host_size ? false : is_tiled;
    const u32 tile_x = GetDefaultBlockWidth(pixel_format);
    const u32 tile_y = GetDefaultBlockHeight(pixel_format);
    const u32 m_width = GetMipmapSize(uncompressed, GetMipWidth(level), tile_x);
    const u32 m_height = GetMipmapSize(uncompressed, GetMipHeight(level), tile_y);
    const u32 m_depth = layer_only ? 1U : GetMipDepth(level);
    return Tegra::Texture::CalculateSize(tiled, GetBytesPerPixel(pixel_format), m_width, m_height,
                                         m_depth, GetMipBlockHeight(level),
                                         GetMipBlockDepth(level));
}

std::size_t SurfaceParams::GetInnerMemorySize(bool as_host_size, bool layer_only,
                                              bool uncompressed) const {
    std::size_t size = 0;
    for (u32 level = 0; level < levels_count; ++level) {
        size += GetInnerMipmapMemorySize(level, as_host_size, layer_only, uncompressed);
    }
    if (!as_host_size && is_tiled) {
        size = Common::AlignUp(size, Tegra::Texture::GetGOBSize() * block_height * block_depth);
    }
    return size;
}

vk::ImageCreateInfo SurfaceParams::CreateInfo(const VKDevice& device) const {
    constexpr auto sample_count = vk::SampleCountFlagBits::e1;
    constexpr auto tiling = vk::ImageTiling::eOptimal;

    const auto [format, attachable] =
        MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, pixel_format, component_type);

    auto image_usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                       vk::ImageUsageFlagBits::eTransferSrc;
    if (attachable) {
        const bool is_zeta = pixel_format >= PixelFormat::MaxColorFormat &&
                             pixel_format < PixelFormat::MaxDepthStencilFormat;
        image_usage |= is_zeta ? vk::ImageUsageFlagBits::eDepthStencilAttachment
                               : vk::ImageUsageFlagBits::eColorAttachment;
    }

    vk::ImageCreateFlags flags;
    vk::Extent3D extent;
    u32 array_layers{};

    switch (target) {
    case SurfaceTarget::Texture2D:
        extent = {width, height, 1};
        array_layers = 1;
        break;
    case SurfaceTarget::Texture2DArray:
        extent = {width, height, 1};
        array_layers = depth;
        break;
    case SurfaceTarget::TextureCubemap:
    case SurfaceTarget::TextureCubeArray:
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
        extent = {width, height, 1};
        array_layers = depth;
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented surface target {}", static_cast<u32>(target));
        break;
    }

    return vk::ImageCreateInfo(flags, SurfaceTargetToImageVK(target), format, extent, levels_count,
                               array_layers, sample_count, tiling, image_usage,
                               vk::SharingMode::eExclusive, 0, nullptr,
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
        for (u32 layer = 0; layer < params.depth; layer++) {
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
    : VKImage(device, params.CreateInfo(device), PixelFormatToImageAspect(params.pixel_format)),
      device{device}, resource_manager{resource_manager}, memory_manager{memory_manager},
      sched{sched}, params{params}, cached_size_in_bytes{params.guest_size_in_bytes},
      buffer_size{std::max(params.guest_size_in_bytes, params.host_size_in_bytes)},
      view_offset_map{BuildViewOffsetMap(params)} {
    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();

    image_commit = memory_manager.Commit(GetHandle(), false);

    const vk::BufferCreateInfo buffer_ci({}, buffer_size,
                                         vk::BufferUsageFlagBits::eTransferDst |
                                             vk::BufferUsageFlagBits::eTransferSrc,
                                         vk::SharingMode::eExclusive, 0, nullptr);
    buffer = dev.createBufferUnique(buffer_ci, nullptr, dld);
    buffer_commit = memory_manager.Commit(*buffer, true);
    vk_buffer = buffer_commit->GetData();
}

CachedSurface::~CachedSurface() = default;

void CachedSurface::LoadVKBuffer() {
    UNIMPLEMENTED_IF(!params.is_tiled);
    ASSERT_MSG(params.block_width == 1, "Block width is defined as {} on texture target {}",
               params.block_width, static_cast<u32>(params.target));

    for (u32 level = 0; level < params.levels_count; ++level) {
        u8* buffer = vk_buffer + params.GetHostMipmapLevelOffset(level);
        SwizzleFunc(MortonSwizzleMode::MortonToLinear, address, params, buffer, level);
    }
}

VKExecutionContext CachedSurface::FlushVKBuffer(VKExecutionContext exctx) {
    if (!is_modified) {
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
        SwizzleFunc(MortonSwizzleMode::LinearToMorton, address, params, buffer, level);
    }

    return exctx;
}

VKExecutionContext CachedSurface::UploadVKTexture(VKExecutionContext exctx) {
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

View CachedSurface::TryGetView(VAddr view_address, const SurfaceParams& view_params) {
    if (view_address < address) {
        // It can't be a view if it's in a prior address.
        return {};
    }
    const auto relative_offset = static_cast<u64>(view_address - address);
    const auto it = view_offset_map.find(relative_offset);
    if (it == view_offset_map.end()) {
        // Couldn't find an aligned view.
        return {};
    }
    const auto [layer, level] = it->second;

    // TODO(Rodrigo): Do proper matching
    if (view_params.width != params.GetMipWidth(level) ||
        view_params.height != params.GetMipHeight(level)) {
        return {};
    }

    // TODO(Rodrigo): Do bounds checkings

    return GetView(layer, view_params.depth, level, view_params.levels_count);
}

bool CachedSurface::IsFamiliar(const SurfaceParams& view_params) const {
    if (!(std::tie(params.is_tiled, params.tile_width_spacing, params.pixel_format,
                   params.component_type, params.type) ==
          std::tie(view_params.is_tiled, view_params.tile_width_spacing, view_params.pixel_format,
                   view_params.component_type, view_params.type))) {
        return false;
    }
    const SurfaceTarget view_target = view_params.target;
    if (view_target == params.target) {
        return true;
    }
    switch (params.target) {
    case SurfaceTarget::Texture2D:
        return false;
    case SurfaceTarget::Texture2DArray:
        return view_target == SurfaceTarget::Texture2D;
    case SurfaceTarget::TextureCubemap:
        return view_target == SurfaceTarget::Texture2D ||
               view_target == SurfaceTarget::Texture2DArray;
    case SurfaceTarget::TextureCubeArray:
        return view_target == SurfaceTarget::Texture2D ||
               view_target == SurfaceTarget::Texture2DArray ||
               view_target == SurfaceTarget::TextureCubemap;
    default:
        UNIMPLEMENTED_MSG("Unimplemented texture family={}", static_cast<u32>(params.target));
        return false;
    }
}

View CachedSurface::GetView(u32 base_layer, u32 layers, u32 base_level, u32 levels) {
    ViewKey key;
    key.base_layer = base_layer;
    key.layers = layers;
    key.base_level = base_level;
    key.levels = levels;
    const auto [entry, is_cache_miss] = views.try_emplace(key);
    auto& view = entry->second;
    if (is_cache_miss) {
        view = std::make_unique<CachedView>(device, this, base_layer, layers, base_level, levels);
    }
    return view.get();
}

vk::BufferImageCopy CachedSurface::GetBufferImageCopy(u32 level) const {
    switch (params.target) {
    case SurfaceTarget::Texture2D:
    case SurfaceTarget::Texture2DArray:
    case SurfaceTarget::TextureCubemap:
    case SurfaceTarget::TextureCubeArray:
        return vk::BufferImageCopy(params.GetHostMipmapLevelOffset(level), 0, 0,
                                   {GetAspectMask(), level, 0, params.depth}, {0, 0, 0},
                                   {params.GetMipWidth(level), params.GetMipHeight(level), 1});
    default:
        UNIMPLEMENTED_MSG("Unimplemented surface target {}", static_cast<u32>(params.target));
        return {};
    }
}

vk::ImageSubresourceRange CachedSurface::GetImageSubresourceRange() const {
    switch (params.target) {
    case SurfaceTarget::Texture2D:
    case SurfaceTarget::Texture2DArray:
    case SurfaceTarget::TextureCubemap:
    case SurfaceTarget::TextureCubeArray:
        return {GetAspectMask(), 0, params.levels_count, 0, params.depth};
    default:
        UNIMPLEMENTED_MSG("Unimplemented surface target {}", static_cast<u32>(params.target));
        return {};
    }
}

std::map<u64, std::pair<u32, u32>> CachedSurface::BuildViewOffsetMap(const SurfaceParams& params) {
    std::map<u64, std::pair<u32, u32>> view_offset_map;
    switch (params.target) {
    case SurfaceTarget::Texture2D:
        view_offset_map.insert({0, {0, 0}});
        break;
    case SurfaceTarget::Texture2DArray:
    case SurfaceTarget::TextureCubemap:
    case SurfaceTarget::TextureCubeArray: {
        const std::size_t layer_size = params.GetGuestLayerMemorySize();
        for (u32 level = 0; level < params.levels_count; ++level) {
            const std::size_t level_offset = params.GetGuestMipmapLevelOffset(level);
            for (u32 layer = 0; layer < params.depth; ++layer) {
                const auto layer_offset = static_cast<std::size_t>(layer_size * layer);
                view_offset_map.insert({level_offset + layer_offset, {layer, level}});
            }
        }
        break;
    }
    default:
        UNIMPLEMENTED_MSG("Unimplemented surface target {}", static_cast<u32>(params.target));
    }
    return view_offset_map;
}

CachedView::CachedView(const VKDevice& device, Surface surface, u32 base_layer, u32 layers,
                       u32 base_level, u32 levels)
    : params{surface->GetSurfaceParams()}, image{surface->GetHandle()},
      aspect_mask{surface->GetAspectMask()}, device{device}, surface{surface},
      base_layer{base_layer}, layers{layers}, base_level{base_level}, levels{levels} {};

CachedView::~CachedView() = default;

vk::ImageView CachedView::GetHandle(Tegra::Shader::TextureType texture_type, bool is_array) {
    const auto GetOrCreateView = [&](UniqueImageView& image_view, vk::ImageViewType view_type) {
        if (image_view) {
            return *image_view;
        }
        const vk::ComponentMapping swizzle;
        const vk::ImageViewCreateInfo image_view_ci({}, surface->GetHandle(), view_type,
                                                    surface->GetFormat(), swizzle,
                                                    GetImageSubresourceRange());
        const auto dev = device.GetLogical();
        const auto& dld = device.GetDispatchLoader();
        image_view = dev.createImageViewUnique(image_view_ci, nullptr, dld);
        return *image_view;
    };

    switch (texture_type) {
    case Tegra::Shader::TextureType::Texture2D:
        if (is_array) {
            return GetOrCreateView(image_view_2d_array, vk::ImageViewType::e2DArray);
        } else {
            return GetOrCreateView(image_view_2d, vk::ImageViewType::e2D);
        }
    case Tegra::Shader::TextureType::TextureCube:
        if (is_array) {
            return GetOrCreateView(image_view_cube_array, vk::ImageViewType::eCubeArray);
        } else {
            return GetOrCreateView(image_view_cube, vk::ImageViewType::eCube);
        }
    default:
        UNIMPLEMENTED_MSG("Texture type {} not implemented", static_cast<u32>(texture_type));
        return {};
    }
}

VKTextureCache::VKTextureCache(Core::System& system, VideoCore::RasterizerInterface& rasterizer,
                               const VKDevice& device, VKResourceManager& resource_manager,
                               VKMemoryManager& memory_manager, VKScheduler& sched)
    : system{system}, rasterizer{rasterizer}, device{device}, resource_manager{resource_manager},
      memory_manager{memory_manager}, sched{sched} {}

VKTextureCache::~VKTextureCache() = default;

void VKTextureCache::InvalidateRegion(VAddr address, std::size_t size) {
    for (const Surface surface : GetSurfacesInRegion(address, size)) {
        if (!surface->IsRegistered()) {
            // Skip duplicates
            continue;
        }
        Unregister(surface);
    }
}

std::tuple<View, VKExecutionContext> VKTextureCache::GetTextureSurface(
    VKExecutionContext exctx, const Tegra::Texture::FullTextureInfo& config,
    const VKShader::SamplerEntry& entry) {
    return GetSurfaceView(exctx, GetAddressForTexture(system, config),
                          SurfaceParams::CreateForTexture(system, config, entry), true);
}

std::tuple<View, VKExecutionContext> VKTextureCache::GetDepthBufferSurface(VKExecutionContext exctx,
                                                                           bool preserve_contents) {
    const auto& regs{system.GPU().Maxwell3D().regs};
    if (!regs.zeta.Address() || !regs.zeta_enable) {
        return {{}, exctx};
    }

    auto& memory_manager{system.GPU().MemoryManager()};
    const auto cpu_addr{memory_manager.GpuToCpuAddress(regs.zeta.Address())};
    ASSERT(cpu_addr);

    const SurfaceParams depth_params{SurfaceParams::CreateForDepthBuffer(
        system, regs.zeta_width, regs.zeta_height, regs.zeta.format,
        regs.zeta.memory_layout.block_width, regs.zeta.memory_layout.block_height,
        regs.zeta.memory_layout.block_depth, regs.zeta.memory_layout.type)};

    return GetSurfaceView(exctx, *cpu_addr, depth_params, preserve_contents);
}

std::tuple<View, VKExecutionContext> VKTextureCache::GetColorBufferSurface(VKExecutionContext exctx,
                                                                           std::size_t index,
                                                                           bool preserve_contents) {
    const auto& regs{system.GPU().Maxwell3D().regs};
    ASSERT(index < Tegra::Engines::Maxwell3D::Regs::NumRenderTargets);

    if (index >= regs.rt_control.count) {
        return {{}, exctx};
    }
    if (regs.rt[index].Address() == 0 || regs.rt[index].format == Tegra::RenderTargetFormat::NONE) {
        return {{}, exctx};
    }

    return GetSurfaceView(exctx, GetAddressForFramebuffer(system, index),
                          SurfaceParams::CreateForFramebuffer(system, index), preserve_contents);
}

Surface VKTextureCache::TryFindFramebufferSurface(VAddr address) const {
    const auto it = registered_surfaces.find(address);
    return it != registered_surfaces.end() ? *it->second.begin() : nullptr;
}

VKExecutionContext VKTextureCache::LoadSurface(VKExecutionContext exctx, const Surface& surface) {
    surface->LoadVKBuffer();
    exctx = surface->UploadVKTexture(exctx);
    surface->MarkAsModified(false);
    return exctx;
}

std::tuple<View, VKExecutionContext> VKTextureCache::GetSurfaceView(VKExecutionContext exctx,
                                                                    VAddr address,
                                                                    const SurfaceParams& params,
                                                                    bool preserve_contents) {
    const std::vector<Surface> overlaps = GetSurfacesInRegion(address, params.guest_size_in_bytes);
    if (overlaps.empty()) {
        return LoadSurfaceView(exctx, address, params, preserve_contents);
    }

    if (const Surface overlap = overlaps[0]; overlaps.size() == 1 && overlap->IsFamiliar(params)) {
        if (const View view = overlap->TryGetView(address, params); view)
            return {view, exctx};
    }

    View fast_view;
    std::tie(fast_view, exctx) =
        TryFastGetSurfaceView(exctx, address, params, preserve_contents, overlaps);

    for (const Surface surface : overlaps) {
        if (!fast_view) {
            // Flush even when we don't care about the contents, to preserve memory not written by
            // the new surface.
            LOG_CRITICAL(HW_GPU, "Flushing");
            exctx = surface->FlushVKBuffer(exctx);
        }
        Unregister(surface);
    }

    if (fast_view) {
        return {fast_view, exctx};
    }

    return LoadSurfaceView(exctx, address, params, preserve_contents);
}

std::tuple<View, VKExecutionContext> VKTextureCache::LoadSurfaceView(VKExecutionContext exctx,
                                                                     VAddr address,
                                                                     const SurfaceParams& params,
                                                                     bool preserve_contents) {
    const Surface new_surface = GetUncachedSurface(params);
    Register(new_surface, address);
    if (preserve_contents) {
        exctx = LoadSurface(exctx, new_surface);
    }
    return {new_surface->GetView(address, params), exctx};
}

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

std::vector<Surface> VKTextureCache::GetSurfacesInRegion(VAddr address, std::size_t size) const {
    if (size == 0) {
        return {};
    }
    const boost::icl::interval_map<VAddr, Surface>::interval_type interval{
        address, address + static_cast<VAddr>(size)};

    std::vector<Surface> surfaces;
    for (auto& pair : boost::make_iterator_range(registered_surfaces.equal_range(interval))) {
        surfaces.push_back(*pair.second.begin());
    }
    return surfaces;
}

void VKTextureCache::Register(Surface surface, VAddr address) {
    surface->Register(address);
    registered_surfaces.add(
        {{address, address + static_cast<VAddr>(surface->GetSizeInBytes())}, {surface}});

    rasterizer.UpdatePagesCachedCount(surface->GetAddress(), surface->GetSizeInBytes(), 1);
}

void VKTextureCache::Unregister(Surface surface) {
    surface->Unregister();
    const auto interval = boost::icl::interval_map<VAddr, Surface>::interval_type::right_open(
        surface->GetAddress(), surface->GetAddress() + surface->GetSizeInBytes());
    registered_surfaces.subtract({interval, {surface}});

    rasterizer.UpdatePagesCachedCount(surface->GetAddress(), surface->GetSizeInBytes(), -1);
}

Surface VKTextureCache::GetUncachedSurface(const SurfaceParams& params) {
    if (const Surface surface = TryGetReservedSurface(params); surface)
        return surface;
    // No reserved surface available, create a new one and reserve it
    auto new_surface = std::make_unique<CachedSurface>(system, device, resource_manager,
                                                       memory_manager, sched, params);
    const Surface surface = new_surface.get();
    ReserveSurface(std::move(new_surface));
    return surface;
}

void VKTextureCache::ReserveSurface(std::unique_ptr<CachedSurface> surface) {
    const auto& surface_reserve_key{SurfaceReserveKey::Create(surface->GetSurfaceParams())};
    surface_reserve[surface_reserve_key].push_back(std::move(surface));
}

Surface VKTextureCache::TryGetReservedSurface(const SurfaceParams& params) {
    const auto& surface_reserve_key{SurfaceReserveKey::Create(params)};
    auto search{surface_reserve.find(surface_reserve_key)};
    if (search == surface_reserve.end()) {
        return {};
    }
    for (auto& surface : search->second) {
        if (!surface->IsRegistered()) {
            return surface.get();
        }
    }
    return {};
}

} // namespace Vulkan