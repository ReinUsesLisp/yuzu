// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "video_core/texture_cache/image_view_info.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/types.h"
#include "video_core/textures/texture.h"

namespace VideoCommon {

namespace {

[[nodiscard]] u8 CastSwizzle(SwizzleSource source) {
    const u8 casted = static_cast<u8>(source);
    ASSERT(static_cast<SwizzleSource>(casted) == source);
    return casted;
}

} // Anonymous namespace

ImageViewInfo::ImageViewInfo(const TICEntry& config) noexcept
    : format{PixelFormatFromTIC(config)}, x_source{CastSwizzle(config.x_source)},
      y_source{CastSwizzle(config.y_source)}, z_source{CastSwizzle(config.z_source)},
      w_source{CastSwizzle(config.w_source)} {
    range.base.mipmap = config.res_min_mip_level;
    range.extent.mipmaps = config.res_max_mip_level - config.res_min_mip_level + 1;

    switch (config.texture_type) {
    case TextureType::Texture1D:
        ASSERT(config.BaseLayer() == 0);
        ASSERT(config.Height() == 1);
        ASSERT(config.Depth() == 1);
        type = ImageViewType::e1D;
        break;
    case TextureType::Texture2D:
    case TextureType::Texture2DNoMipmap:
        ASSERT(config.Depth() == 1);
        type = config.normalized_coords ? ImageViewType::e2D : ImageViewType::Rect;
        range.base.layer = config.BaseLayer();
        break;
    case TextureType::Texture3D:
        ASSERT(config.BaseLayer() == 0);
        type = ImageViewType::e3D;
        break;
    case TextureType::TextureCubemap:
        ASSERT(config.Depth() == 1);
        UNIMPLEMENTED_IF(config.BaseLayer() != 0);
        type = ImageViewType::Cube;
        range.base.layer = 0;
        range.extent.layers = 6;
        break;
    case TextureType::Texture1DArray:
        type = ImageViewType::e1DArray;
        range.base.layer = config.BaseLayer();
        range.extent.layers = config.Depth();
        break;
    case TextureType::Texture2DArray:
        type = ImageViewType::e2DArray;
        range.base.layer = config.BaseLayer();
        range.extent.layers = config.Depth();
        break;
    case TextureType::Texture1DBuffer:
        UNIMPLEMENTED_MSG("Texture buffers are not implemented");
        break;
    case TextureType::TextureCubeArray:
        UNIMPLEMENTED_IF(config.BaseLayer() != 0);
        type = ImageViewType::CubeArray;
        range.base.layer = 0;
        range.extent.layers = config.Depth() * 6;
        break;
    default:
        UNREACHABLE_MSG("Invalid texture_type={}", static_cast<int>(config.texture_type.Value()));
        break;
    }
}

ImageViewInfo::ImageViewInfo(ImageViewType type_, PixelFormat format_,
                             SubresourceRange range_) noexcept
    : type{type_}, format{format_}, range{range_} {}

} // namespace VideoCommon
