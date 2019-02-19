// Copyright 2018 yuzu Emulator Project
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

namespace Vulkan {

using VideoCore::MortonSwizzle;
using VideoCore::MortonSwizzleMode;
using VideoCore::Surface::ComponentTypeFromDepthFormat;
using VideoCore::Surface::ComponentTypeFromRenderTarget;
using VideoCore::Surface::ComponentTypeFromTexture;
using VideoCore::Surface::PixelFormatFromDepthFormat;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;
using VideoCore::Surface::PixelFormatFromTextureFormat;
using VideoCore::Surface::SurfaceFamilyFromTextureType;

static vk::ImageType SurfaceFamilyToImageVK(SurfaceFamily family) {
    switch (family) {
    case SurfaceFamily::Texture2D:
        return vk::ImageType::e2D;
    }
    UNIMPLEMENTED_MSG("Unimplemented texture family={}", static_cast<u32>(family));
    return vk::ImageType::e2D;
}

static vk::ImageViewType SurfaceTargetToImageViewVK(SurfaceTarget target) {
    switch (target) {
    case SurfaceTarget::Texture2D:
        return vk::ImageViewType::e2D;
    }
    UNIMPLEMENTED_MSG("Unimplemented texture target={}", static_cast<u32>(target));
    return vk::ImageViewType::e2D;
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
    const auto cpu_addr{memory_manager.GpuToCpuAddress(config.Address())};
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
    params.width = Common::AlignUp(config.tic.Width(), GetCompressionFactor(params.pixel_format));
    params.height = Common::AlignUp(config.tic.Height(), GetCompressionFactor(params.pixel_format));
    params.unaligned_height = config.tic.Height();
    params.family = SurfaceFamilyFromTextureType(config.tic.texture_type);

    // Unimplemented
    params.depth = 1;

    // params.is_layered = SurfaceTargetIsLayered(params.target);
    // params.max_mip_level = config.tic.max_mip_level + 1;
    // params.rt = {};

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
    params.family = SurfaceFamily::Texture2D;
    params.depth = 1;
    // params.max_mip_level = 1;
    // params.is_layered = false;
    // params.rt = {};

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
    params.family = SurfaceFamily::Texture2D;
    params.depth = 1;
    // params.max_mip_level = 0;
    // params.is_layered = false;

    // Render target specific parameters, not used for caching
    // params.rt.index = static_cast<u32>(index);
    // params.rt.array_mode = config.array_mode;
    // params.rt.layer_stride = config.layer_stride;
    // params.rt.volume = config.volume;
    // params.rt.base_layer = config.base_layer;

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
    size_in_bytes = GetSizeInBytes();

    if (IsPixelFormatASTC(pixel_format)) {
        // ASTC is uncompressed in software, in emulated as RGBA8
        size_in_bytes_vk = width * height * depth * 4;
    } else {
        size_in_bytes_vk = GetSizeInBytesVK();
    }
}

vk::ImageCreateInfo SurfaceParams::CreateInfo(const VKDevice& device) const {
    constexpr u32 mipmaps = 1;
    constexpr u32 array_layers = 1;
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
    return vk::ImageCreateInfo({}, SurfaceFamilyToImageVK(family), format, {width, height, depth},
                               mipmaps, array_layers, sample_count, tiling, image_usage,
                               vk::SharingMode::eExclusive, 0, nullptr,
                               vk::ImageLayout::eUndefined);
}

static void SwizzleFunc(const MortonSwizzleMode& mode, VAddr address, const SurfaceParams& params,
                        u8* vk_buffer, u32 mip_level) {
    UNIMPLEMENTED_IF(params.depth != 1);

    const u64 offset = 0;
    MortonSwizzle(mode, params.pixel_format, params.width, params.block_height, params.height,
                  params.block_depth, params.depth, params.tile_width_spacing, vk_buffer, 0,
                  address + offset);
}

CachedSurface::CachedSurface(Core::System& system, const VKDevice& device,
                             VKResourceManager& resource_manager, VKMemoryManager& memory_manager,
                             VKScheduler& sched, const SurfaceParams& params)
    : VKImage(device, params.CreateInfo(device), PixelFormatToImageAspect(params.pixel_format)),
      device{device}, resource_manager{resource_manager}, memory_manager{memory_manager},
      sched{sched}, params{params}, cached_size_in_bytes{params.size_in_bytes},
      buffer_size{std::max(params.size_in_bytes, params.size_in_bytes_vk)} {
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
    if (params.is_tiled) {
        ASSERT_MSG(params.block_width == 1, "Block width is defined as {} on texture family {}",
                   params.block_width, static_cast<u32>(params.family));
        SwizzleFunc(MortonSwizzleMode::MortonToLinear, address, params, vk_buffer, 0);
    } else {
        std::memcpy(vk_buffer, Memory::GetPointer(address), params.size_in_bytes_vk);
    }
    ConvertFormatAsNeeded_LoadVKBuffer(vk_buffer, params.pixel_format, params.width, params.height,
                                       params.depth);
}

VKExecutionContext CachedSurface::FlushVKBuffer(VKExecutionContext exctx) {
    const auto& dld = device.GetDispatchLoader();
    const auto cmdbuf = exctx.GetCommandBuffer();

    Transition(cmdbuf, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer,
               vk::AccessFlagBits::eTransferRead);

    const vk::BufferImageCopy copy(0, 0, 0, {GetAspectMask(), 0, 0, 1}, {0, 0, 0},
                                   {params.width, params.height, params.depth});
    cmdbuf.copyImageToBuffer(GetHandle(), vk::ImageLayout::eTransferSrcOptimal, *buffer, {copy},
                             dld);
    exctx = sched.Finish();

    UNIMPLEMENTED_IF(!params.is_tiled);
    ASSERT_MSG(params.block_width == 1, "Block width is defined as {}", params.block_width);
    SwizzleFunc(MortonSwizzleMode::LinearToMorton, address, params, vk_buffer, 0);

    return exctx;
}

VKExecutionContext CachedSurface::UploadVKTexture(VKExecutionContext exctx) {
    const auto cmdbuf = exctx.GetCommandBuffer();
    Transition(cmdbuf, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer,
               vk::AccessFlagBits::eTransferWrite);

    const auto& dld = device.GetDispatchLoader();
    const vk::BufferImageCopy copy(0, 0, 0, {GetAspectMask(), 0, 0, 1}, {0, 0, 0},
                                   {params.width, params.height, params.depth});
    if (GetAspectMask() == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) {
        vk::BufferImageCopy depth = copy;
        vk::BufferImageCopy stencil = copy;
        depth.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
        stencil.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eStencil;

        cmdbuf.copyBufferToImage(*buffer, GetHandle(), vk::ImageLayout::eTransferDstOptimal,
                                 {depth, stencil}, dld);
    } else {
        cmdbuf.copyBufferToImage(*buffer, GetHandle(), vk::ImageLayout::eTransferDstOptimal, {copy},
                                 dld);
    }
    return exctx;
}

View CachedSurface::GetView(u32 layer, u32 level) {
    ViewKey key;
    key.layer = layer;
    key.level = level;
    const auto [entry, is_cache_miss] = views.try_emplace(key);
    auto& view = entry->second;
    if (is_cache_miss) {
        view = std::make_unique<CachedView>(device, this, layer, level);
    }
    return view.get();
}

CachedView::CachedView(const VKDevice& device, Surface surface, u32 layer, u32 level)
    : device{device}, surface{surface} {
    UNIMPLEMENTED_IF(layer > 0);
    UNIMPLEMENTED_IF(level > 0);
    const vk::ComponentMapping swizzle;
    const vk::ImageSubresourceRange range(surface->GetAspectMask(), level, 1, layer, 1);
    const vk::ImageViewCreateInfo image_view_ci({}, surface->GetHandle(), vk::ImageViewType::e2D,
                                                surface->GetFormat(), swizzle, range);

    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    image_view = dev.createImageViewUnique(image_view_ci, nullptr, dld);
}

CachedView::~CachedView() = default;

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
    return GetView(exctx, GetAddressForTexture(system, config),
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

    return GetView(exctx, *cpu_addr, depth_params, preserve_contents);
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

    return GetView(exctx, GetAddressForFramebuffer(system, index),
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

std::tuple<View, VKExecutionContext> VKTextureCache::GetView(VKExecutionContext exctx,
                                                             VAddr address,
                                                             const SurfaceParams& params,
                                                             bool preserve_contents) {
    const std::vector<Surface> overlaps = GetSurfacesInRegion(address, params.GetSizeInBytes());
    if (overlaps.empty()) {
        return LoadView(exctx, address, params, preserve_contents);
    }

    if (overlaps.size() == 1) {
        if (const Surface overlap = overlaps[0]; overlap->IsFamiliar(params)) {
            if (const View view = overlap->TryGetView(params); view)
                return {view, exctx};
        }
    }

    for (const Surface overlap : overlaps) {
        exctx = overlap->FlushVKBuffer(exctx);
        Unregister(overlap);
    }

    return LoadView(exctx, address, params, preserve_contents);
}

std::tuple<View, VKExecutionContext> VKTextureCache::LoadView(VKExecutionContext exctx,
                                                              VAddr address,
                                                              const SurfaceParams& params,
                                                              bool preserve_contents) {
    const Surface surface = GetUncachedSurface(params);
    Register(surface, address);

    if (preserve_contents) {
        exctx = LoadSurface(exctx, surface);
    }
    const View superset_view = surface->TryGetView(params);
    ASSERT(superset_view);
    return {superset_view, exctx};
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