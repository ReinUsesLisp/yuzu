// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include "common/common_types.h"
#include "video_core/textures/texture.h"

namespace Tegra::Texture {

// GOBSize constant. Calculated by 64 bytes in x multiplied by 8 y coords, represents
// an small rect of (64/bytes_per_pixel)X8.
inline std::size_t GetGOBSize() {
    return 512;
}

/// Unswizzles a swizzled texture without changing its format.
void UnswizzleTexture(u8* unswizzled_data, u8* address, std::size_t tile_size_x,
                      std::size_t tile_size_y, std::size_t bytes_per_pixel, std::size_t width,
                      std::size_t height, std::size_t depth,
                      std::size_t block_height = TICEntry::DefaultBlockHeight,
                      std::size_t block_depth = TICEntry::DefaultBlockHeight,
                      std::size_t width_spacing = 0);

/// Unswizzles a swizzled texture without changing its format.
std::vector<u8> UnswizzleTexture(u8* address, std::size_t tile_size_x, std::size_t tile_size_y,
                                 std::size_t bytes_per_pixel, std::size_t width, std::size_t height,
                                 std::size_t depth,
                                 std::size_t block_height = TICEntry::DefaultBlockHeight,
                                 std::size_t block_depth = TICEntry::DefaultBlockHeight,
                                 std::size_t width_spacing = 0);

/// Copies texture data from a buffer and performs swizzling/unswizzling as necessary.
void CopySwizzledData(std::size_t width, std::size_t height, std::size_t depth,
                      std::size_t bytes_per_pixel, std::size_t out_bytes_per_pixel,
                      u8* swizzled_data, u8* unswizzled_data, bool unswizzle,
                      std::size_t block_height, std::size_t block_depth, std::size_t width_spacing);

/// Decodes an unswizzled texture into a A8R8G8B8 texture.
std::vector<u8> DecodeTexture(const std::vector<u8>& texture_data, TextureFormat format,
                              std::size_t width, std::size_t height);

/// This function calculates the correct size of a texture depending if it's tiled or not.
std::size_t CalculateSize(bool tiled, std::size_t bytes_per_pixel, std::size_t width,
                          std::size_t height, std::size_t depth, std::size_t block_height,
                          std::size_t block_depth);

/// Copies an untiled subrectangle into a tiled surface.
void SwizzleSubrect(std::size_t subrect_width, std::size_t subrect_height, std::size_t source_pitch,
                    std::size_t swizzled_width, std::size_t bytes_per_pixel, u8* swizzled_data,
                    u8* unswizzled_data, std::size_t block_height);

/// Copies a tiled subrectangle into a linear surface.
void UnswizzleSubrect(std::size_t subrect_width, std::size_t subrect_height, std::size_t dest_pitch,
                      std::size_t swizzled_width, std::size_t bytes_per_pixel, u8* swizzled_data,
                      u8* unswizzled_data, std::size_t block_height, std::size_t offset_x,
                      std::size_t offset_y);

} // namespace Tegra::Texture
