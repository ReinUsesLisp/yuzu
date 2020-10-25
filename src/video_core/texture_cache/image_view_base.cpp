// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>

#include "common/assert.h"
#include "video_core/compatible_formats.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/image_view_info.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

ImageViewBase::ImageViewBase(const ImageViewInfo& info, const ImageInfo& image_info,
                             ImageId image_id_)
    : image_id{image_id_}, format{info.format}, type{info.type}, range{info.range},
      size{
          std::max(image_info.size.width >> range.base.mipmap, 1u),
          std::max(image_info.size.height >> range.base.mipmap, 1u),
          std::max(image_info.size.depth >> range.base.mipmap, 1u),
      } {
    ASSERT_MSG(VideoCore::Surface::IsViewCompatible(image_info.format, info.format),
               "Image view format {} is incompatible with image format {}", info.format,
               image_info.format);
}

ImageViewBase::ImageViewBase(const NullImageParams&) {}

} // namespace VideoCommon
