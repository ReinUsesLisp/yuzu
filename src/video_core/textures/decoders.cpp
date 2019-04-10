// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cmath>
#include <cstring>
#include "common/alignment.h"
#include "common/assert.h"
#include "video_core/gpu.h"
#include "video_core/textures/decoders.h"
#include "video_core/textures/texture.h"

namespace Tegra::Texture {

/**
 * This table represents the internal swizzle of a gob,
 * in format 16 bytes x 2 sector packing.
 * Calculates the offset of an (x, y) position within a swizzled texture.
 * Taken from the Tegra X1 Technical Reference Manual. pages 1187-1188
 */
template <std::size_t N, std::size_t M, std::size_t Align>
struct alignas(64) SwizzleTable {
    static_assert(M * Align == 64, "Swizzle Table does not align to GOB");
    constexpr SwizzleTable() {
        for (std::size_t y = 0; y < N; ++y) {
            for (std::size_t x = 0; x < M; ++x) {
                const std::size_t x2 = x * Align;
                values[y][x] = ((x2 % 64) / 32) * 256 + ((y % 8) / 2) * 64 + ((x2 % 32) / 16) * 32 +
                               (y % 2) * 16 + (x2 % 16);
            }
        }
    }
    const std::array<std::size_t, M>& operator[](std::size_t index) const {
        return values[index];
    }
    std::array<std::array<std::size_t, M>, N> values{};
};

constexpr std::size_t gob_size_x = 64;
constexpr std::size_t gob_size_y = 8;
constexpr std::size_t gob_size_z = 1;
constexpr std::size_t gob_size = gob_size_x * gob_size_y * gob_size_z;
constexpr std::size_t fast_swizzle_align = 16;

constexpr auto legacy_swizzle_table = SwizzleTable<gob_size_y, gob_size_x, gob_size_z>();
constexpr auto fast_swizzle_table = SwizzleTable<gob_size_y, 4, fast_swizzle_align>();

/**
 * This function manages ALL the GOBs(Group of Bytes) Inside a single block.
 * Instead of going gob by gob, we map the coordinates inside a block and manage from
 * those. Block_Width is assumed to be 1.
 */
void PreciseProcessBlock(u8* const swizzled_data, u8* const unswizzled_data, const bool unswizzle,
                         const std::size_t x_start, const std::size_t y_start,
                         const std::size_t z_start, const std::size_t x_end,
                         const std::size_t y_end, const std::size_t z_end,
                         const std::size_t tile_offset, const std::size_t xy_block_size,
                         const std::size_t layer_z, const std::size_t stride_x,
                         const std::size_t bytes_per_pixel, const std::size_t out_bytes_per_pixel) {
    std::array<u8*, 2> data_ptrs;
    std::size_t z_address = tile_offset;

    for (std::size_t z = z_start; z < z_end; z++) {
        std::size_t y_address = z_address;
        std::size_t pixel_base = layer_z * z + y_start * stride_x;
        for (std::size_t y = y_start; y < y_end; y++) {
            const auto& table = legacy_swizzle_table[y % gob_size_y];
            for (std::size_t x = x_start; x < x_end; x++) {
                const std::size_t swizzle_offset{y_address +
                                                 table[x * bytes_per_pixel % gob_size_x]};
                const std::size_t pixel_index{x * out_bytes_per_pixel + pixel_base};
                data_ptrs[unswizzle] = swizzled_data + swizzle_offset;
                data_ptrs[!unswizzle] = unswizzled_data + pixel_index;
                std::memcpy(data_ptrs[0], data_ptrs[1], bytes_per_pixel);
            }
            pixel_base += stride_x;
            if ((y + 1) % gob_size_y == 0)
                y_address += gob_size;
        }
        z_address += xy_block_size;
    }
}

/**
 * This function manages ALL the GOBs(Group of Bytes) Inside a single block.
 * Instead of going gob by gob, we map the coordinates inside a block and manage from
 * those. Block_Width is assumed to be 1.
 */
void FastProcessBlock(u8* const swizzled_data, u8* const unswizzled_data, const bool unswizzle,
                      const std::size_t x_start, const std::size_t y_start,
                      const std::size_t z_start, const std::size_t x_end, const std::size_t y_end,
                      const std::size_t z_end, const std::size_t tile_offset,
                      const std::size_t xy_block_size, const std::size_t layer_z,
                      const std::size_t stride_x, const std::size_t bytes_per_pixel,
                      const std::size_t out_bytes_per_pixel) {
    std::array<u8*, 2> data_ptrs;
    std::size_t z_address = tile_offset;
    const std::size_t x_startb = x_start * bytes_per_pixel;
    const std::size_t x_endb = x_end * bytes_per_pixel;

    for (std::size_t z = z_start; z < z_end; z++) {
        std::size_t y_address = z_address;
        std::size_t pixel_base = layer_z * z + y_start * stride_x;
        for (std::size_t y = y_start; y < y_end; y++) {
            const auto& table = fast_swizzle_table[y % gob_size_y];
            for (std::size_t xb = x_startb; xb < x_endb; xb += fast_swizzle_align) {
                const std::size_t swizzle_offset{y_address + table[(xb / fast_swizzle_align) % 4]};
                const std::size_t out_x = xb * out_bytes_per_pixel / bytes_per_pixel;
                const std::size_t pixel_index{out_x + pixel_base};
                data_ptrs[unswizzle ? 1 : 0] = swizzled_data + swizzle_offset;
                data_ptrs[unswizzle ? 0 : 1] = unswizzled_data + pixel_index;
                std::memcpy(data_ptrs[0], data_ptrs[1], fast_swizzle_align);
            }
            pixel_base += stride_x;
            if ((y + 1) % gob_size_y == 0)
                y_address += gob_size;
        }
        z_address += xy_block_size;
    }
}

/**
 * This function unswizzles or swizzles a texture by mapping Linear to BlockLinear Textue.
 * The body of this function takes care of splitting the swizzled texture into blocks,
 * and managing the extents of it. Once all the parameters of a single block are obtained,
 * the function calls 'ProcessBlock' to process that particular Block.
 *
 * Documentation for the memory layout and decoding can be found at:
 *  https://envytools.readthedocs.io/en/latest/hw/memory/g80-surface.html#blocklinear-surfaces
 */
template <bool fast>
void SwizzledData(u8* const swizzled_data, u8* const unswizzled_data, const bool unswizzle,
                  const std::size_t width, const std::size_t height, const std::size_t depth,
                  const std::size_t bytes_per_pixel, const std::size_t out_bytes_per_pixel,
                  const std::size_t block_height, const std::size_t block_depth,
                  const std::size_t width_spacing) {
    auto div_ceil = [](const std::size_t x, const std::size_t y) { return ((x + y - 1) / y); };
    const std::size_t stride_x = width * out_bytes_per_pixel;
    const std::size_t layer_z = height * stride_x;
    const std::size_t gob_elements_x = gob_size_x / bytes_per_pixel;
    constexpr std::size_t gob_elements_y = gob_size_y;
    constexpr std::size_t gob_elements_z = gob_size_z;
    const std::size_t block_x_elements = gob_elements_x;
    const std::size_t block_y_elements = gob_elements_y * block_height;
    const std::size_t block_z_elements = gob_elements_z * block_depth;
    const std::size_t aligned_width = Common::AlignUp(width, gob_elements_x * width_spacing);
    const std::size_t blocks_on_x = div_ceil(aligned_width, block_x_elements);
    const std::size_t blocks_on_y = div_ceil(height, block_y_elements);
    const std::size_t blocks_on_z = div_ceil(depth, block_z_elements);
    const std::size_t xy_block_size = gob_size * block_height;
    const std::size_t block_size = xy_block_size * block_depth;
    std::size_t tile_offset = 0;
    for (std::size_t zb = 0; zb < blocks_on_z; zb++) {
        const std::size_t z_start = zb * block_z_elements;
        const std::size_t z_end = std::min(depth, z_start + block_z_elements);
        for (std::size_t yb = 0; yb < blocks_on_y; yb++) {
            const std::size_t y_start = yb * block_y_elements;
            const std::size_t y_end = std::min(height, y_start + block_y_elements);
            for (std::size_t xb = 0; xb < blocks_on_x; xb++) {
                const std::size_t x_start = xb * block_x_elements;
                const std::size_t x_end = std::min(width, x_start + block_x_elements);
                if constexpr (fast) {
                    FastProcessBlock(swizzled_data, unswizzled_data, unswizzle, x_start, y_start,
                                     z_start, x_end, y_end, z_end, tile_offset, xy_block_size,
                                     layer_z, stride_x, bytes_per_pixel, out_bytes_per_pixel);
                } else {
                    PreciseProcessBlock(swizzled_data, unswizzled_data, unswizzle, x_start, y_start,
                                        z_start, x_end, y_end, z_end, tile_offset, xy_block_size,
                                        layer_z, stride_x, bytes_per_pixel, out_bytes_per_pixel);
                }
                tile_offset += block_size;
            }
        }
    }
}

void CopySwizzledData(std::size_t width, std::size_t height, std::size_t depth,
                      std::size_t bytes_per_pixel, std::size_t out_bytes_per_pixel,
                      u8* const swizzled_data, u8* const unswizzled_data, bool unswizzle,
                      std::size_t block_height, std::size_t block_depth,
                      std::size_t width_spacing) {
    if (bytes_per_pixel % 3 != 0 && (width * bytes_per_pixel) % fast_swizzle_align == 0) {
        SwizzledData<true>(swizzled_data, unswizzled_data, unswizzle, width, height, depth,
                           bytes_per_pixel, out_bytes_per_pixel, block_height, block_depth,
                           width_spacing);
    } else {
        SwizzledData<false>(swizzled_data, unswizzled_data, unswizzle, width, height, depth,
                            bytes_per_pixel, out_bytes_per_pixel, block_height, block_depth,
                            width_spacing);
    }
}

std::size_t BytesPerPixel(TextureFormat format) {
    switch (format) {
    case TextureFormat::DXT1:
    case TextureFormat::DXN1:
        // In this case a 'pixel' actually refers to a 4x4 tile.
        return 8;
    case TextureFormat::DXT23:
    case TextureFormat::DXT45:
    case TextureFormat::DXN2:
    case TextureFormat::BC7U:
    case TextureFormat::BC6H_UF16:
    case TextureFormat::BC6H_SF16:
        // In this case a 'pixel' actually refers to a 4x4 tile.
        return 16;
    case TextureFormat::R32_G32_B32:
        return 12;
    case TextureFormat::ASTC_2D_4X4:
    case TextureFormat::ASTC_2D_5X4:
    case TextureFormat::ASTC_2D_8X8:
    case TextureFormat::ASTC_2D_8X5:
    case TextureFormat::ASTC_2D_10X8:
    case TextureFormat::ASTC_2D_5X5:
    case TextureFormat::A8R8G8B8:
    case TextureFormat::A2B10G10R10:
    case TextureFormat::BF10GF11RF11:
    case TextureFormat::R32:
    case TextureFormat::R16_G16:
        return 4;
    case TextureFormat::A1B5G5R5:
    case TextureFormat::B5G6R5:
    case TextureFormat::G8R8:
    case TextureFormat::R16:
        return 2;
    case TextureFormat::R8:
        return 1;
    case TextureFormat::R16_G16_B16_A16:
        return 8;
    case TextureFormat::R32_G32_B32_A32:
        return 16;
    case TextureFormat::R32_G32:
        return 8;
    default:
        UNIMPLEMENTED_MSG("Format not implemented");
        return 1;
    }
}

void UnswizzleTexture(u8* const unswizzled_data, u8* address, std::size_t tile_size_x,
                      std::size_t tile_size_y, std::size_t bytes_per_pixel, std::size_t width,
                      std::size_t height, std::size_t depth, std::size_t block_height,
                      std::size_t block_depth, std::size_t width_spacing) {
    CopySwizzledData((width + tile_size_x - 1) / tile_size_x,
                     (height + tile_size_y - 1) / tile_size_y, depth, bytes_per_pixel,
                     bytes_per_pixel, address, unswizzled_data, true, block_height, block_depth,
                     width_spacing);
}

std::vector<u8> UnswizzleTexture(u8* address, std::size_t tile_size_x, std::size_t tile_size_y,
                                 std::size_t bytes_per_pixel, std::size_t width, std::size_t height,
                                 std::size_t depth, std::size_t block_height,
                                 std::size_t block_depth, std::size_t width_spacing) {
    std::vector<u8> unswizzled_data(width * height * depth * bytes_per_pixel);
    UnswizzleTexture(unswizzled_data.data(), address, tile_size_x, tile_size_y, bytes_per_pixel,
                     width, height, depth, block_height, block_depth, width_spacing);
    return unswizzled_data;
}

void SwizzleSubrect(std::size_t subrect_width, std::size_t subrect_height, std::size_t source_pitch,
                    std::size_t swizzled_width, std::size_t bytes_per_pixel, u8* swizzled_data,
                    u8* unswizzled_data, std::size_t block_height) {
    const std::size_t image_width_in_gobs{(swizzled_width * bytes_per_pixel + (gob_size_x - 1)) /
                                          gob_size_x};
    for (std::size_t line = 0; line < subrect_height; ++line) {
        const std::size_t gob_address_y =
            (line / (gob_size_y * block_height)) * gob_size * block_height * image_width_in_gobs +
            ((line % (gob_size_y * block_height)) / gob_size_y) * gob_size;
        const auto& table = legacy_swizzle_table[line % gob_size_y];
        for (std::size_t x = 0; x < subrect_width; ++x) {
            const std::size_t gob_address =
                gob_address_y + (x * bytes_per_pixel / gob_size_x) * gob_size * block_height;
            const std::size_t swizzled_offset =
                gob_address + table[(x * bytes_per_pixel) % gob_size_x];
            u8* source_line = unswizzled_data + line * source_pitch + x * bytes_per_pixel;
            u8* dest_addr = swizzled_data + swizzled_offset;

            std::memcpy(dest_addr, source_line, bytes_per_pixel);
        }
    }
}

void UnswizzleSubrect(std::size_t subrect_width, std::size_t subrect_height, std::size_t dest_pitch,
                      std::size_t swizzled_width, std::size_t bytes_per_pixel, u8* swizzled_data,
                      u8* unswizzled_data, std::size_t block_height, std::size_t offset_x,
                      std::size_t offset_y) {
    for (std::size_t line = 0; line < subrect_height; ++line) {
        const std::size_t y2 = line + offset_y;
        const std::size_t gob_address_y =
            (y2 / (gob_size_y * block_height)) * gob_size * block_height +
            ((y2 % (gob_size_y * block_height)) / gob_size_y) * gob_size;
        const auto& table = legacy_swizzle_table[y2 % gob_size_y];
        for (std::size_t x = 0; x < subrect_width; ++x) {
            const std::size_t x2 = (x + offset_x) * bytes_per_pixel;
            const std::size_t gob_address =
                gob_address_y + (x2 / gob_size_x) * gob_size * block_height;
            const std::size_t swizzled_offset = gob_address + table[x2 % gob_size_x];
            u8* dest_line = unswizzled_data + line * dest_pitch + x * bytes_per_pixel;
            u8* source_addr = swizzled_data + swizzled_offset;

            std::memcpy(dest_line, source_addr, bytes_per_pixel);
        }
    }
}

std::vector<u8> DecodeTexture(const std::vector<u8>& texture_data, TextureFormat format,
                              std::size_t width, std::size_t height) {
    std::vector<u8> rgba_data;

    // TODO(Subv): Implement.
    switch (format) {
    case TextureFormat::DXT1:
    case TextureFormat::DXT23:
    case TextureFormat::DXT45:
    case TextureFormat::DXN1:
    case TextureFormat::DXN2:
    case TextureFormat::BC7U:
    case TextureFormat::BC6H_UF16:
    case TextureFormat::BC6H_SF16:
    case TextureFormat::ASTC_2D_4X4:
    case TextureFormat::ASTC_2D_8X8:
    case TextureFormat::ASTC_2D_5X5:
    case TextureFormat::ASTC_2D_10X8:
    case TextureFormat::A8R8G8B8:
    case TextureFormat::A2B10G10R10:
    case TextureFormat::A1B5G5R5:
    case TextureFormat::B5G6R5:
    case TextureFormat::R8:
    case TextureFormat::G8R8:
    case TextureFormat::BF10GF11RF11:
    case TextureFormat::R32_G32_B32_A32:
    case TextureFormat::R32_G32:
    case TextureFormat::R32:
    case TextureFormat::R16:
    case TextureFormat::R16_G16:
    case TextureFormat::R32_G32_B32:
        // TODO(Subv): For the time being just forward the same data without any decoding.
        rgba_data = texture_data;
        break;
    default:
        UNIMPLEMENTED_MSG("Format not implemented");
        break;
    }

    return rgba_data;
}

std::size_t CalculateSize(bool tiled, std::size_t bytes_per_pixel, std::size_t width,
                          std::size_t height, std::size_t depth, std::size_t block_height,
                          std::size_t block_depth) {
    if (tiled) {
        const std::size_t aligned_width = Common::AlignUp(width * bytes_per_pixel, gob_size_x);
        const std::size_t aligned_height = Common::AlignUp(height, gob_size_y * block_height);
        const std::size_t aligned_depth = Common::AlignUp(depth, gob_size_z * block_depth);
        return aligned_width * aligned_height * aligned_depth;
    } else {
        return width * height * depth * bytes_per_pixel;
    }
}

} // namespace Tegra::Texture
