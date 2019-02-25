// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common_types.h"
#include "core/core.h"
#include "video_core/surface.h"
#include "video_core/texture_cache.h"
#include "video_core/textures/decoders.h"
#include "video_core/textures/texture.h"

namespace VideoCommon {

using VideoCore::Surface::SurfaceTarget;

using VideoCore::Surface::ComponentTypeFromDepthFormat;
using VideoCore::Surface::ComponentTypeFromRenderTarget;
using VideoCore::Surface::ComponentTypeFromTexture;
using VideoCore::Surface::PixelFormatFromDepthFormat;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;
using VideoCore::Surface::PixelFormatFromTextureFormat;
using VideoCore::Surface::SurfaceTargetFromTextureType;

constexpr u32 GetMipmapSize(bool uncompressed, u32 mip_size, u32 tile) {
    return uncompressed ? mip_size : std::max(1U, (mip_size + tile - 1) / tile);
}

SurfaceParams SurfaceParams::CreateForTexture(Core::System& system,
                                              const Tegra::Texture::FullTextureInfo& config) {
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
    params.pitch = params.is_tiled ? 0 : config.tic.Pitch();
    params.unaligned_height = config.tic.Height();
    params.levels_count = config.tic.max_mip_level + 1;

    params.CalculateSizes();
    return params;
}

SurfaceParams SurfaceParams::CreateForDepthBuffer(
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

SurfaceParams SurfaceParams::CreateForFramebuffer(Core::System& system, std::size_t index) {
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
    if (params.is_tiled) {
        params.width = config.width;
    } else {
        const u32 bpp = GetFormatBpp(params.pixel_format) / CHAR_BIT;
        params.pitch = config.width;
        params.width = params.pitch / bpp;
    }
    params.height = config.height;
    params.depth = 1;
    params.unaligned_height = config.height;
    params.target = SurfaceTarget::Texture2D;
    params.levels_count = 1;

    params.CalculateSizes();
    return params;
}

u32 SurfaceParams::GetMipWidth(u32 level) const {
    return std::max(1U, width >> level);
}

u32 SurfaceParams::GetMipHeight(u32 level) const {
    return std::max(1U, height >> level);
}

u32 SurfaceParams::GetMipDepth(u32 level) const {
    return IsLayered() ? depth : std::max(1U, depth >> level);
}

u32 SurfaceParams::GetLayersCount() const {
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

bool SurfaceParams::IsLayered() const {
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

u32 SurfaceParams::GetMipBlockHeight(u32 level) const {
    // Auto block resizing algorithm from:
    // https://cgit.freedesktop.org/mesa/mesa/tree/src/gallium/drivers/nouveau/nv50/nv50_miptree.c
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

u32 SurfaceParams::GetMipBlockDepth(u32 level) const {
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

std::size_t SurfaceParams::GetGuestMipmapLevelOffset(u32 level) const {
    std::size_t offset = 0;
    for (u32 i = 0; i < level; i++) {
        offset += GetInnerMipmapMemorySize(i, false, IsLayered(), false);
    }
    return offset;
}

std::size_t SurfaceParams::GetHostMipmapLevelOffset(u32 level) const {
    std::size_t offset = 0;
    for (u32 i = 0; i < level; i++) {
        offset += GetInnerMipmapMemorySize(i, true, false, false);
    }
    return offset;
}

std::size_t SurfaceParams::GetGuestLayerMemorySize() const {
    return GetInnerMemorySize(false, true, false);
}

std::size_t SurfaceParams::GetHostLayerSize(u32 level) const {
    return GetInnerMipmapMemorySize(level, true, IsLayered(), false);
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

} // namespace VideoCommon