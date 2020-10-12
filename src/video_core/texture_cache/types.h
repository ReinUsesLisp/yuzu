// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_funcs.h"
#include "common/common_types.h"
#include "video_core/texture_cache/slot_vector.h"

namespace VideoCommon {

constexpr size_t NUM_RT = 8;
constexpr size_t MAX_MIPMAP = 14;

using ImageId = SlotId;
using ImageViewId = SlotId;
using ImageAllocId = SlotId;
using SamplerId = SlotId;
using FramebufferId = SlotId;

enum class ImageType : u32 {
    e1D,
    e2D,
    e3D,
    Linear,
    Rect,
};

enum class ImageViewType : u32 {
    e1D,
    e2D,
    Cube,
    e3D,
    e1DArray,
    e2DArray,
    CubeArray,
    Rect,
    Buffer,
};
constexpr size_t NUM_IMAGE_VIEW_TYPES = 8;

enum class RelaxedOptions : u32 {
    Size = 1 << 0,
    Format = 1 << 1,
};
DECLARE_ENUM_FLAG_OPERATORS(RelaxedOptions)

struct Offset3D {
    constexpr auto operator<=>(const Offset3D&) const noexcept = default;

    u32 x;
    u32 y;
    u32 z;
};

struct Extent2D {
    constexpr auto operator<=>(const Extent2D&) const noexcept = default;

    u32 width;
    u32 height;
};

struct Extent3D {
    constexpr auto operator<=>(const Extent3D&) const noexcept = default;

    u32 width;
    u32 height;
    u32 depth;
};

struct SubresourceLayers {
    u32 base_mipmap = 0;
    u32 base_layer = 0;
    u32 num_layers = 1;
};

struct SubresourceBase {
    constexpr auto operator<=>(const SubresourceBase&) const noexcept = default;

    u32 mipmap = 0;
    u32 layer = 0;
};

struct SubresourceExtent {
    constexpr auto operator<=>(const SubresourceExtent&) const noexcept = default;

    u32 mipmaps = 1;
    u32 layers = 1;
};

struct SubresourceRange {
    constexpr auto operator<=>(const SubresourceRange&) const noexcept = default;

    SubresourceBase base;
    SubresourceExtent extent;
};

struct ImageCopy {
    SubresourceLayers src_subresource;
    SubresourceLayers dst_subresource;
    Offset3D src_offset;
    Offset3D dst_offset;
    Extent3D extent;
};

struct BufferImageCopy {
    size_t buffer_offset;
    size_t buffer_size;
    u32 buffer_row_length;
    u32 buffer_image_height;
    SubresourceLayers image_subresource;
    Offset3D image_offset;
    Extent3D image_extent;
};

struct SwizzleParameters {
    Extent3D num_tiles;
    Extent3D block;
    size_t buffer_offset;
    u32 mipmap;
};

} // namespace VideoCommon
