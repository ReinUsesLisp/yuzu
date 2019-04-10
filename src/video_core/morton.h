// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"
#include "video_core/surface.h"

namespace VideoCore {

enum class MortonSwizzleMode { MortonToLinear, LinearToMorton };

void MortonSwizzle(MortonSwizzleMode mode, VideoCore::Surface::PixelFormat format,
                   std::size_t stride, std::size_t block_height, std::size_t height,
                   std::size_t block_depth, std::size_t depth, std::size_t tile_width_spacing,
                   u8* buffer, u8* addr);

void MortonCopyPixels128(MortonSwizzleMode mode, std::size_t width, std::size_t height,
                         std::size_t bytes_per_pixel, std::size_t linear_bytes_per_pixel,
                         u8* morton_data, u8* linear_data);

} // namespace VideoCore
