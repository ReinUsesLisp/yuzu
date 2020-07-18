// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/format_lookup_table.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/samples_helper.h"
#include "video_core/texture_cache/types.h"
#include "video_core/textures/texture.h"

namespace VideoCommon {

using Tegra::Texture::TextureType;
using Tegra::Texture::TICEntry;
using VideoCore::Surface::PixelFormat;

ImageInfo::ImageInfo(const TICEntry& config) {
    format = PixelFormatFromTextureInfo(config.format, config.r_type, config.g_type, config.b_type,
                                        config.a_type, config.srgb_conversion);

    num_samples = NumSamples(config.msaa_mode);
    resources.mipmaps = config.max_mip_level + 1;

    if (config.IsPitchLinear()) {
        pitch = config.Pitch();
    } else {
        block = {
            .width = config.block_width,
            .height = config.block_height,
            .depth = config.block_depth,
        };
    }
    UNIMPLEMENTED_IF(config.tile_width_spacing != 0);

    if (config.texture_type != TextureType::Texture2D &&
        config.texture_type != TextureType::Texture2DNoMipmap) {
        ASSERT(!config.IsPitchLinear());
    }

    switch (config.texture_type) {
    case TextureType::Texture1D:
        ASSERT(config.BaseLayer() == 0);
        type = ImageType::e1D;
        size.width = config.Width();
        break;
    case TextureType::Texture1DArray:
        UNIMPLEMENTED_IF(config.BaseLayer() != 0);
        type = ImageType::e1D;
        size.width = config.Width();
        resources.layers = config.Depth();
        break;
    case TextureType::Texture2D:
    case TextureType::Texture2DNoMipmap:
        ASSERT(config.Depth() == 1);
        type = config.IsPitchLinear() ? ImageType::Linear : ImageType::e2D;
        size.width = config.Width();
        size.height = config.Height();
        resources.layers = config.BaseLayer() + 1;
        break;
    case TextureType::Texture2DArray:
        UNIMPLEMENTED_IF(config.BaseLayer() != 0);
        type = ImageType::e2D;
        size.width = config.Width();
        size.height = config.Height();
        resources.layers = config.Depth();
        break;
    case TextureType::TextureCubemap:
        UNIMPLEMENTED_IF(config.BaseLayer() != 0);
        ASSERT(config.Depth() == 1);
        type = ImageType::e2D;
        size.width = config.Width();
        size.height = config.Height();
        resources.layers = 6;
        break;
    case TextureType::TextureCubeArray:
        UNIMPLEMENTED_IF(config.load_store_hint != 0);
        type = ImageType::e2D;
        size.width = config.Width();
        size.height = config.Height();
        resources.layers = config.Depth() * 6;
        break;
    case TextureType::Texture3D:
        ASSERT(config.BaseLayer() == 0);
        type = ImageType::e3D;
        size.width = config.Width();
        size.height = config.Height();
        size.depth = config.Depth();
        break;
    default:
        UNREACHABLE_MSG("Invalid texture_type={}", static_cast<int>(config.texture_type.Value()));
        break;
    }
}

ImageInfo::ImageInfo(const Tegra::Engines::Maxwell3D::Regs& regs, size_t index) {
    const auto& rt = regs.rt[index];

    format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(rt.format);

    size.width = rt.width;
    size.height = rt.height;

    num_samples = NumSamples(regs.multisample_mode);

    block = {
        .width = rt.tile_mode.block_width,
        .height = rt.tile_mode.block_height,
        .depth = rt.tile_mode.block_depth,
    };

    if (rt.tile_mode.is_pitch_linear) {
        ASSERT(rt.tile_mode.is_3d == 0);
        type = ImageType::Linear;
        pitch = size.width * BytesPerBlock(format);
        return;
    }

    if (rt.tile_mode.is_3d) {
        ASSERT(rt.tile_mode.is_pitch_linear == 0);
        type = ImageType::e3D;
        size.depth = rt.depth;
    } else {
        type = ImageType::e2D;
        resources.layers = rt.depth;
    }
}

ImageInfo::ImageInfo(const Tegra::Engines::Maxwell3D::Regs& regs) {
    format = VideoCore::Surface::PixelFormatFromDepthFormat(regs.zeta.format);

    size.width = regs.zeta_width;
    size.height = regs.zeta_height;

    // TODO: Maybe we can deduce the number of mipmaps from the layer stride
    resources.mipmaps = 1;
    num_samples = NumSamples(regs.multisample_mode);

    block = {
        .width = regs.zeta.tile_mode.block_width,
        .height = regs.zeta.tile_mode.block_height,
        .depth = regs.zeta.tile_mode.block_depth,
    };

    if (regs.zeta.tile_mode.is_pitch_linear) {
        ASSERT(regs.zeta.tile_mode.is_3d == 0);
        type = ImageType::Linear;
        pitch = size.width * BytesPerBlock(format);
    } else if (regs.zeta.tile_mode.is_3d) {
        ASSERT(regs.zeta.tile_mode.is_pitch_linear == 0);
        type = ImageType::e3D;
        size.depth = regs.zeta_depth;
    } else {
        type = ImageType::e2D;
        resources.layers = regs.zeta_depth;
    }
}

ImageInfo::ImageInfo(const Tegra::Engines::Fermi2D::Regs::Surface& config) {
    format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(config.format);

    if (config.linear) {
        type = ImageType::Linear;
        size.width = config.width;
        pitch = config.pitch;
        return;
    }

    block = {
        .width = config.block_width,
        .height = config.block_height,
        .depth = config.block_depth,
    };

    if (block.depth > 0) {
        type = ImageType::e3D;
        size.width = config.width;
        size.height = config.height;
        size.depth = config.depth;
    } else {
        type = ImageType::e2D;
        size.width = config.width;
        size.height = config.height;
        resources.layers = config.depth;
    }
}

} // namespace VideoCommon
