// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/common_types.h"
#include "video_core/compatible_formats.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/format_lookup_table.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/texture_cache/util.h"
#include "video_core/textures/astc.h"
#include "video_core/textures/decoders.h"

namespace VideoCommon {

namespace {

using Tegra::Texture::GOB_SIZE;
using Tegra::Texture::GOB_SIZE_SHIFT;
using Tegra::Texture::GOB_SIZE_X;
using Tegra::Texture::GOB_SIZE_X_SHIFT;
using Tegra::Texture::GOB_SIZE_Y;
using Tegra::Texture::GOB_SIZE_Y_SHIFT;
using Tegra::Texture::GOB_SIZE_Z;
using Tegra::Texture::GOB_SIZE_Z_SHIFT;
using Tegra::Texture::MsaaMode;
using Tegra::Texture::SwizzleTexture;
using Tegra::Texture::TextureFormat;
using Tegra::Texture::TextureType;
using Tegra::Texture::TICEntry;
using Tegra::Texture::UnswizzleTexture;
using VideoCore::Surface::BytesPerBlock;
using VideoCore::Surface::DefaultBlockHeight;
using VideoCore::Surface::DefaultBlockWidth;
using VideoCore::Surface::IsCopyCompatible;
using VideoCore::Surface::IsViewCompatible;
using VideoCore::Surface::PixelFormatFromDepthFormat;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;
using VideoCore::Surface::SurfaceType;

constexpr u32 CONVERTED_BYTES_PER_BLOCK = BytesPerBlock(PixelFormat::A8B8G8R8_UNORM);

struct LevelInfo {
    Extent3D size;
    Extent3D block;
    Extent2D tile_size;
    u32 bpp_log2;
    u32 tile_width_spacing;
};

template <typename T>
[[nodiscard]] constexpr T DivCeil(T number, T divisor) {
    return (number + divisor - 1) / divisor;
}

template <typename T>
[[nodiscard]] constexpr T DivCeilLog2(T value, size_t alignment_log2) {
    return (value + (T(1) << alignment_log2) - 1) >> alignment_log2;
}

[[nodiscard]] constexpr u32 AdjustTileSize(u32 shift, u32 unit_factor, u32 dimension) {
    if (shift == 0) {
        return 0;
    }
    u32 x = unit_factor << (shift - 1);
    if (x >= dimension) {
        while (--shift) {
            x >>= 1;
            if (x < dimension) {
                break;
            }
        }
    }
    return shift;
}

[[nodiscard]] constexpr u32 AdjustMipSize(u32 size, u32 level) {
    return std::max<u32>(size >> level, 1);
}

[[nodiscard]] constexpr Extent3D AdjustMipSize(Extent3D size, s32 level) {
    return Extent3D{
        .width = AdjustMipSize(size.width, level),
        .height = AdjustMipSize(size.height, level),
        .depth = AdjustMipSize(size.depth, level),
    };
}

[[nodiscard]] Extent3D AdjustSamplesSize(Extent3D size, s32 num_samples) {
    const auto [samples_x, samples_y] = SamplesLog2(num_samples);
    return Extent3D{
        .width = size.width >> samples_x,
        .height = size.height >> samples_y,
        .depth = size.depth,
    };
}

template <u32 GOB_EXTENT>
[[nodiscard]] constexpr u32 AdjustMipBlockSize(u32 num_tiles, u32 block_size, u32 level) {
    do {
        while (block_size > 0 && num_tiles <= (1U << (block_size - 1)) * GOB_EXTENT) {
            --block_size;
        }
    } while (level--);
    return block_size;
}

[[nodiscard]] constexpr Extent3D AdjustMipBlockSize(Extent3D num_tiles, Extent3D block_size,
                                                    u32 level) {
    return {
        .width = AdjustMipBlockSize<GOB_SIZE_X>(num_tiles.width, block_size.width, level),
        .height = AdjustMipBlockSize<GOB_SIZE_Y>(num_tiles.height, block_size.height, level),
        .depth = AdjustMipBlockSize<GOB_SIZE_Z>(num_tiles.depth, block_size.depth, level),
    };
}

[[nodiscard]] constexpr Extent3D AdjustTileSize(Extent3D size, Extent2D tile_size) {
    return {
        .width = DivCeil(size.width, tile_size.width),
        .height = DivCeil(size.height, tile_size.height),
        .depth = size.depth,
    };
}

[[nodiscard]] constexpr u32 BytesPerBlockLog2(u32 bytes_per_block) {
    return std::countl_zero(bytes_per_block) ^ 0x1F;
}

[[nodiscard]] constexpr u32 BytesPerBlockLog2(PixelFormat format) {
    return BytesPerBlockLog2(BytesPerBlock(format));
}

[[nodiscard]] constexpr u32 NumBlocks(Extent3D size, Extent2D tile_size) {
    const Extent3D num_blocks = AdjustTileSize(size, tile_size);
    return num_blocks.width * num_blocks.height * num_blocks.depth;
}

[[nodiscard]] constexpr u32 AdjustSize(u32 size, u32 level, u32 block_size) {
    return DivCeil(AdjustMipSize(size, level), block_size);
}

[[nodiscard]] constexpr u32 LayerSize(const TICEntry& config, PixelFormat format) {
    return config.Width() * config.Height() * BytesPerBlock(format);
}

[[nodiscard]] constexpr bool HasTwoDimsPerLayer(TextureType type) {
    switch (type) {
    case TextureType::Texture2D:
    case TextureType::Texture2DArray:
    case TextureType::Texture2DNoMipmap:
    case TextureType::Texture3D:
    case TextureType::TextureCubeArray:
    case TextureType::TextureCubemap:
        return true;
    case TextureType::Texture1D:
    case TextureType::Texture1DArray:
    case TextureType::Texture1DBuffer:
        return false;
    }
    return false;
}

[[nodiscard]] constexpr bool HasTwoDimsPerLayer(ImageType type) {
    switch (type) {
    case ImageType::e2D:
    case ImageType::e3D:
    case ImageType::Linear:
    case ImageType::Rect:
        return true;
    case ImageType::e1D:
    case ImageType::Buffer:
        return false;
    }
    UNREACHABLE_MSG("Invalid image type={}", static_cast<int>(type));
}

[[nodiscard]] constexpr std::pair<int, int> Samples(int num_samples) {
    switch (num_samples) {
    case 1:
        return {1, 1};
    case 2:
        return {2, 1};
    case 4:
        return {2, 2};
    case 8:
        return {4, 2};
    case 16:
        return {4, 4};
    }
    UNREACHABLE_MSG("Invalid number of samples={}", num_samples);
    return {1, 1};
}

[[nodiscard]] constexpr Extent2D DefaultBlockSize(PixelFormat format) {
    return {DefaultBlockWidth(format), DefaultBlockHeight(format)};
}

[[nodiscard]] constexpr Extent3D NumLevelBlocks(const LevelInfo& info, u32 level) {
    return Extent3D{
        .width = AdjustSize(info.size.width, level, info.tile_size.width) << info.bpp_log2,
        .height = AdjustSize(info.size.height, level, info.tile_size.height),
        .depth = AdjustMipSize(info.size.depth, level),
    };
}

[[nodiscard]] constexpr Extent3D TileShift(const LevelInfo& info, u32 level) {
    const Extent3D blocks = NumLevelBlocks(info, level);
    return Extent3D{
        .width = AdjustTileSize(info.block.width, GOB_SIZE_X, blocks.width),
        .height = AdjustTileSize(info.block.height, GOB_SIZE_Y, blocks.height),
        .depth = AdjustTileSize(info.block.depth, GOB_SIZE_Z, blocks.depth),
    };
}

[[nodiscard]] constexpr Extent2D GobSize(u32 bpp_log2, u32 block_height, u32 tile_width_spacing) {
    return Extent2D{
        .width = GOB_SIZE_X_SHIFT - bpp_log2 + tile_width_spacing,
        .height = GOB_SIZE_Y_SHIFT + block_height,
    };
}

[[nodiscard]] constexpr bool IsSmallerThanGobSize(Extent3D num_tiles, Extent2D gob,
                                                  u32 block_depth) {
    return num_tiles.width <= (1U << gob.width) || num_tiles.height <= (1U << gob.height) ||
           num_tiles.depth < (1U << block_depth);
}

[[nodiscard]] constexpr u32 StrideAlignment(Extent3D num_tiles, Extent3D block, Extent2D gob,
                                            u32 bpp_log2) {
    if (IsSmallerThanGobSize(num_tiles, gob, block.depth)) {
        return GOB_SIZE_X_SHIFT - bpp_log2;
    } else {
        return gob.width;
    }
}

[[nodiscard]] constexpr u32 StrideAlignment(Extent3D num_tiles, Extent3D block, u32 bpp_log2,
                                            u32 tile_width_spacing) {
    const Extent2D gob = GobSize(bpp_log2, block.height, tile_width_spacing);
    return StrideAlignment(num_tiles, block, gob, bpp_log2);
}

[[nodiscard]] constexpr Extent2D NumGobs(const LevelInfo& info, u32 level) {
    const Extent3D blocks = NumLevelBlocks(info, level);
    const Extent2D gobs{
        .width = DivCeilLog2(blocks.width, GOB_SIZE_X_SHIFT),
        .height = DivCeilLog2(blocks.height, GOB_SIZE_Y_SHIFT),
    };
    const Extent2D gob = GobSize(info.bpp_log2, info.block.height, info.tile_width_spacing);
    const bool is_small = IsSmallerThanGobSize(blocks, gob, info.block.depth);
    const u32 alignment = is_small ? 0 : info.tile_width_spacing;
    return Extent2D{
        .width = Common::AlignBits(gobs.width, alignment),
        .height = gobs.height,
    };
}

[[nodiscard]] constexpr Extent3D LevelTiles(const LevelInfo& info, u32 level) {
    const Extent3D blocks = NumLevelBlocks(info, level);
    const Extent3D tile_shift = TileShift(info, level);
    const Extent2D gobs = NumGobs(info, level);
    return Extent3D{
        .width = DivCeilLog2(gobs.width, tile_shift.width),
        .height = DivCeilLog2(gobs.height, tile_shift.height),
        .depth = DivCeilLog2(blocks.depth, tile_shift.depth),
    };
}

[[nodiscard]] constexpr u32 CalculateLevelSize(const LevelInfo& info, u32 level) {
    const Extent3D tile_shift = TileShift(info, level);
    const Extent3D tiles = LevelTiles(info, level);
    const u32 num_tiles = tiles.width * tiles.height * tiles.depth;
    const u32 shift = GOB_SIZE_SHIFT + tile_shift.width + tile_shift.height + tile_shift.depth;
    return num_tiles << shift;
}

[[nodiscard]] constexpr std::array<u32, MAX_MIPMAP> CalculateLevelSizes(const LevelInfo& info,
                                                                        u32 num_mipmaps) {
    ASSERT(num_mipmaps <= MAX_MIPMAP);
    std::array<u32, MAX_MIPMAP> sizes{};
    for (u32 level = 0; level < num_mipmaps; ++level) {
        sizes[level] = CalculateLevelSize(info, level);
    }
    return sizes;
}

[[nodiscard]] constexpr LevelInfo MakeLevelInfo(PixelFormat format, Extent3D size, Extent3D block,
                                                u32 num_samples, u32 tile_width_spacing) {
    const auto [samples_x, samples_y] = Samples(num_samples);
    const u32 bytes_per_block = BytesPerBlock(format);
    return {
        .size =
            {
                .width = size.width * samples_x,
                .height = size.height * samples_y,
                .depth = size.depth,
            },
        .block = block,
        .tile_size = DefaultBlockSize(format),
        .bpp_log2 = BytesPerBlockLog2(bytes_per_block),
        .tile_width_spacing = tile_width_spacing,
    };
}

[[nodiscard]] constexpr LevelInfo MakeLevelInfo(const ImageInfo& info) {
    return MakeLevelInfo(info.format, info.size, info.block, info.num_samples,
                         info.tile_width_spacing);
}

[[nodiscard]] constexpr u32 CalculateLevelOffset(PixelFormat format, Extent3D size, Extent3D block,
                                                 u32 num_samples, u32 tile_width_spacing,
                                                 u32 level) {
    const LevelInfo info = MakeLevelInfo(format, size, block, num_samples, tile_width_spacing);
    u32 offset = 0;
    for (u32 current_level = 0; current_level < level; ++current_level) {
        offset += CalculateLevelSize(info, current_level);
    }
    return offset;
}

[[nodiscard]] constexpr u32 AlignLayerSize(u32 size_bytes, Extent3D size, Extent3D block,
                                           u32 tile_size_y, u32 tile_width_spacing) {
    // https://github.com/Ryujinx/Ryujinx/blob/1c9aba6de1520aea5480c032e0ff5664ac1bb36f/Ryujinx.Graphics.Texture/SizeCalculator.cs#L134
    if (tile_width_spacing > 0) {
        const u32 alignment_log2 = GOB_SIZE_SHIFT + tile_width_spacing + block.height + block.depth;
        return Common::AlignBits(size_bytes, alignment_log2);
    }
    const u32 aligned_height = Common::AlignUp(size.height, tile_size_y);
    while (block.height != 0 && aligned_height <= (1U << (block.height - 1)) * GOB_SIZE_Y) {
        --block.height;
    }
    while (block.depth != 0 && size.depth <= (1U << (block.depth - 1))) {
        --block.depth;
    }
    const u32 block_shift = GOB_SIZE_SHIFT + block.height + block.depth;
    const u32 num_blocks = size_bytes >> block_shift;
    if (size_bytes != num_blocks << block_shift) {
        return (num_blocks + 1) << block_shift;
    }
    return size_bytes;
}

[[nodiscard]] std::optional<SubresourceExtent> ResolveOverlapEqualAddress(const ImageInfo& new_info,
                                                                          const ImageBase& overlap,
                                                                          bool strict_size) {
    const ImageInfo& info = overlap.info;
    if (!IsBlockLinearSizeCompatible(new_info, info, 0, 0, strict_size)) {
        return std::nullopt;
    }
    if (new_info.block != info.block) {
        return std::nullopt;
    }
    const SubresourceExtent resources = new_info.resources;
    return SubresourceExtent{
        .mipmaps = std::max(resources.mipmaps, info.resources.mipmaps),
        .layers = std::max(resources.layers, info.resources.layers),
    };
}

[[nodiscard]] std::optional<SubresourceExtent> ResolveOverlapRightAddress3D(
    const ImageInfo& new_info, GPUVAddr gpu_addr, const ImageBase& overlap, bool strict_size) {
    const std::vector<u32> slice_offsets = CalculateSliceOffsets(new_info);
    const u32 diff = static_cast<u32>(overlap.gpu_addr - gpu_addr);
    const auto it = std::ranges::find(slice_offsets, diff);
    if (it == slice_offsets.end()) {
        return std::nullopt;
    }
    const std::vector subresources = CalculateSliceSubresources(new_info);
    const SubresourceBase base = subresources[std::distance(slice_offsets.begin(), it)];
    const ImageInfo& info = overlap.info;
    if (!IsBlockLinearSizeCompatible(new_info, info, base.mipmap, 0, strict_size)) {
        return std::nullopt;
    }
    const u32 mip_depth = std::max(1U, new_info.size.depth << base.mipmap);
    if (mip_depth < info.size.depth + base.layer) {
        return std::nullopt;
    }
    if (MipBlockSize(new_info, base.mipmap) != info.block) {
        return std::nullopt;
    }
    return SubresourceExtent{
        .mipmaps = std::max(new_info.resources.mipmaps, info.resources.mipmaps + base.mipmap),
        .layers = 1,
    };
}

[[nodiscard]] std::optional<SubresourceExtent> ResolveOverlapRightAddress2D(
    const ImageInfo& new_info, GPUVAddr gpu_addr, const ImageBase& overlap, bool strict_size) {
    const u32 layer_stride = new_info.layer_stride;
    const s32 new_size = layer_stride * new_info.resources.layers;
    const s32 diff = static_cast<s32>(overlap.gpu_addr - gpu_addr);
    if (diff > new_size) {
        return std::nullopt;
    }
    const s32 base_layer = diff / layer_stride;
    const s32 mip_offset = diff % layer_stride;
    const std::array offsets = CalculateMipmapOffsets(new_info);
    const auto end = offsets.begin() + new_info.resources.mipmaps;
    const auto it = std::find(offsets.begin(), end, mip_offset);
    if (it == end) {
        // Mipmap is not aligned to any valid size
        return std::nullopt;
    }
    const SubresourceBase base{
        .mipmap = static_cast<s32>(std::distance(offsets.begin(), it)),
        .layer = base_layer,
    };
    const ImageInfo& info = overlap.info;
    if (!IsBlockLinearSizeCompatible(new_info, info, base.mipmap, 0, strict_size)) {
        return std::nullopt;
    }
    if (MipBlockSize(new_info, base.mipmap) != info.block) {
        return std::nullopt;
    }
    return SubresourceExtent{
        .mipmaps = std::max(new_info.resources.mipmaps, info.resources.mipmaps + base.mipmap),
        .layers = std::max(new_info.resources.layers, info.resources.layers + base.layer),
    };
}

[[nodiscard]] std::optional<OverlapResult> ResolveOverlapRightAddress(const ImageInfo& new_info,
                                                                      GPUVAddr gpu_addr,
                                                                      VAddr cpu_addr,
                                                                      const ImageBase& overlap,
                                                                      bool strict_size) {
    std::optional<SubresourceExtent> resources;
    if (new_info.type != ImageType::e3D) {
        resources = ResolveOverlapRightAddress2D(new_info, gpu_addr, overlap, strict_size);
    } else {
        resources = ResolveOverlapRightAddress3D(new_info, gpu_addr, overlap, strict_size);
    }
    if (!resources) {
        return std::nullopt;
    }
    return OverlapResult{
        .gpu_addr = gpu_addr,
        .cpu_addr = cpu_addr,
        .resources = *resources,
    };
}

[[nodiscard]] std::optional<OverlapResult> ResolveOverlapLeftAddress(const ImageInfo& new_info,
                                                                     GPUVAddr gpu_addr,
                                                                     VAddr cpu_addr,
                                                                     const ImageBase& overlap,
                                                                     bool strict_size) {
    const std::optional<SubresourceBase> base = overlap.TryFindBase(gpu_addr);
    if (!base) {
        return std::nullopt;
    }
    const ImageInfo& info = overlap.info;
    if (!IsBlockLinearSizeCompatible(new_info, info, base->mipmap, 0, strict_size)) {
        return std::nullopt;
    }
    if (new_info.block != MipBlockSize(info, base->mipmap)) {
        return std::nullopt;
    }
    const SubresourceExtent resources = new_info.resources;
    s32 layers = 1;
    if (info.type != ImageType::e3D) {
        layers = std::max(resources.layers, info.resources.layers + base->layer);
    }
    return OverlapResult{
        .gpu_addr = overlap.gpu_addr,
        .cpu_addr = overlap.cpu_addr,
        .resources =
            {
                .mipmaps = std::max(resources.mipmaps + base->mipmap, info.resources.mipmaps),
                .layers = layers,
            },
    };
}

[[nodiscard]] Extent2D PitchLinearAlignedSize(const ImageInfo& info) {
    // https://github.com/Ryujinx/Ryujinx/blob/1c9aba6de1520aea5480c032e0ff5664ac1bb36f/Ryujinx.Graphics.Texture/SizeCalculator.cs#L212
    static constexpr u32 STRIDE_ALIGNMENT = 32;
    ASSERT(info.type == ImageType::Linear);
    const Extent2D num_tiles{
        .width = DivCeil(info.size.width, DefaultBlockWidth(info.format)),
        .height = DivCeil(info.size.height, DefaultBlockHeight(info.format)),
    };
    const u32 width_alignment = STRIDE_ALIGNMENT / BytesPerBlock(info.format);
    return Extent2D{
        .width = Common::AlignUp(num_tiles.width, width_alignment),
        .height = num_tiles.height,
    };
}

[[nodiscard]] Extent3D BlockLinearAlignedSize(const ImageInfo& info, u32 level) {
    // https://github.com/Ryujinx/Ryujinx/blob/1c9aba6de1520aea5480c032e0ff5664ac1bb36f/Ryujinx.Graphics.Texture/SizeCalculator.cs#L176
    ASSERT(info.type != ImageType::Linear);
    const Extent3D size = AdjustMipSize(info.size, level);
    const Extent3D num_tiles{
        .width = DivCeil(size.width, DefaultBlockWidth(info.format)),
        .height = DivCeil(size.height, DefaultBlockHeight(info.format)),
        .depth = size.depth,
    };
    const u32 bpp_log2 = BytesPerBlockLog2(info.format);
    const u32 alignment = StrideAlignment(num_tiles, info.block, bpp_log2, info.tile_width_spacing);
    const Extent3D mip_block = AdjustMipBlockSize(num_tiles, info.block, 0);
    return Extent3D{
        .width = Common::AlignBits(num_tiles.width, alignment),
        .height = Common::AlignBits(num_tiles.height, GOB_SIZE_Y_SHIFT + mip_block.height),
        .depth = Common::AlignBits(num_tiles.depth, GOB_SIZE_Z_SHIFT + mip_block.depth),
    };
}

[[nodiscard]] constexpr u32 NumBlocksPerLayer(const ImageInfo& info, Extent2D tile_size) noexcept {
    u32 num_blocks = 0;
    for (s32 mipmap = 0; mipmap < info.resources.mipmaps; ++mipmap) {
        const Extent3D mip_size = AdjustMipSize(info.size, mipmap);
        num_blocks += NumBlocks(mip_size, tile_size);
    }
    return num_blocks;
}

[[nodiscard]] u32 NumSlices(const ImageInfo& info) noexcept {
    ASSERT(info.type == ImageType::e3D);
    u32 num_slices = 0;
    for (s32 level = 0; level < info.resources.mipmaps; ++level) {
        num_slices += AdjustMipSize(info.size.depth, level);
    }
    return num_slices;
}

void SwizzlePitchLinearImage(Tegra::MemoryManager& gpu_memory, GPUVAddr gpu_addr,
                             const ImageInfo& info, const BufferImageCopy& copy,
                             std::span<const u8> memory) {
    ASSERT(copy.image_offset.z == 0);
    ASSERT(copy.image_extent.depth == 1);
    ASSERT(copy.image_subresource.base_mipmap == 0);
    ASSERT(copy.image_subresource.base_layer == 0);
    ASSERT(copy.image_subresource.num_layers == 1);

    const u32 bytes_per_block = BytesPerBlock(info.format);
    const u32 row_length = copy.image_extent.width * bytes_per_block;
    const u32 guest_offset_x = copy.image_offset.x * bytes_per_block;

    for (u32 line = 0; line < copy.image_extent.height; ++line) {
        const u32 host_offset_y = line * info.pitch;
        const u32 guest_offset_y = (copy.image_offset.y + line) * info.pitch;
        const u32 guest_offset = guest_offset_x + guest_offset_y;
        gpu_memory.WriteBlockUnsafe(gpu_addr + guest_offset, memory.data() + host_offset_y,
                                    row_length);
    }
}

void SwizzleBlockLinearImage(Tegra::MemoryManager& gpu_memory, GPUVAddr gpu_addr,
                             const ImageInfo& info, const BufferImageCopy& copy,
                             std::span<const u8> input) {
    const Extent3D size = info.size;
    const LevelInfo level_info = MakeLevelInfo(info);
    const Extent2D tile_size = DefaultBlockSize(info.format);
    const u32 bytes_per_block = BytesPerBlock(info.format);

    const u32 mipmap = copy.image_subresource.base_mipmap;
    const Extent3D level_size = AdjustMipSize(size, mipmap);
    const u32 num_blocks_per_layer = NumBlocks(level_size, tile_size);
    const u32 host_bytes_per_layer = num_blocks_per_layer * bytes_per_block;

    UNIMPLEMENTED_IF(info.tile_width_spacing > 0);

    UNIMPLEMENTED_IF(copy.image_offset.x != 0);
    UNIMPLEMENTED_IF(copy.image_offset.y != 0);
    UNIMPLEMENTED_IF(copy.image_offset.z != 0);
    UNIMPLEMENTED_IF(copy.image_extent != level_size);

    const Extent3D num_tiles = AdjustTileSize(level_size, tile_size);
    const Extent3D block = AdjustMipBlockSize(num_tiles, level_info.block, mipmap);

    size_t host_offset = copy.buffer_offset;

    const u32 num_mipmaps = info.resources.mipmaps;
    const std::array sizes = CalculateLevelSizes(level_info, num_mipmaps);
    size_t guest_offset = std::reduce(sizes.begin(), sizes.begin() + mipmap, 0);
    const size_t layer_stride =
        AlignLayerSize(std::reduce(sizes.begin(), sizes.begin() + num_mipmaps, 0), size,
                       level_info.block, tile_size.height, info.tile_width_spacing);
    const size_t subresource_size = sizes[mipmap];

    const auto dst_data = std::make_unique<u8[]>(subresource_size);
    const std::span<u8> dst(dst_data.get(), subresource_size);

    for (s32 layer = 0; layer < info.resources.layers; ++layer) {
        const std::span<const u8> src = input.subspan(host_offset);
        SwizzleTexture(dst, src, bytes_per_block, num_tiles.width, num_tiles.height,
                       num_tiles.depth, block.height, block.depth);

        gpu_memory.WriteBlockUnsafe(gpu_addr + guest_offset, dst.data(), dst.size_bytes());

        host_offset += host_bytes_per_layer;
        guest_offset += layer_stride;
    }
    ASSERT(host_offset - copy.buffer_offset == copy.buffer_size);
}

} // Anonymous namespace

u32 CalculateGuestSizeInBytes(const ImageInfo& info) noexcept {
    if (info.type == ImageType::Buffer) {
        return info.size.width * BytesPerBlock(info.format);
    }
    if (info.type == ImageType::Linear) {
        return info.pitch * info.size.height;
    }
    if (info.resources.layers > 1) {
        ASSERT(info.layer_stride != 0);
        return info.layer_stride * info.resources.layers;
    } else {
        return CalculateLayerSize(info);
    }
}

u32 CalculateUnswizzledSizeBytes(const ImageInfo& info) noexcept {
    if (info.type == ImageType::Buffer) {
        return info.size.width * BytesPerBlock(info.format);
    }
    if (info.num_samples > 1) {
        // Multisample images can't be uploaded or downloaded to the host
        return 0;
    }
    if (info.type == ImageType::Linear) {
        return info.pitch * info.size.height;
    }
    const Extent2D tile_size = DefaultBlockSize(info.format);
    return NumBlocksPerLayer(info, tile_size) * info.resources.layers * BytesPerBlock(info.format);
}

u32 CalculateConvertedSizeBytes(const ImageInfo& info) noexcept {
    if (info.type == ImageType::Buffer) {
        return info.size.width * BytesPerBlock(info.format);
    }
    static constexpr Extent2D TILE_SIZE{1, 1};
    return NumBlocksPerLayer(info, TILE_SIZE) * info.resources.layers * CONVERTED_BYTES_PER_BLOCK;
}

u32 CalculateLayerStride(const ImageInfo& info) noexcept {
    ASSERT(info.type != ImageType::Linear);
    const u32 layer_size = CalculateLayerSize(info);
    const Extent3D size = info.size;
    const Extent3D block = info.block;
    const u32 tile_size_y = DefaultBlockHeight(info.format);
    return AlignLayerSize(layer_size, size, block, tile_size_y, info.tile_width_spacing);
}

u32 CalculateLayerSize(const ImageInfo& info) noexcept {
    ASSERT(info.type != ImageType::Linear);
    return CalculateLevelOffset(info.format, info.size, info.block, info.num_samples,
                                info.tile_width_spacing, info.resources.mipmaps);
}

std::array<u32, MAX_MIPMAP> CalculateMipmapOffsets(const ImageInfo& info) noexcept {
    ASSERT(info.resources.mipmaps <= MAX_MIPMAP);
    const LevelInfo level_info = MakeLevelInfo(info);
    std::array<u32, MAX_MIPMAP> offsets{};
    u32 offset = 0;
    for (s32 level = 0; level < info.resources.mipmaps; ++level) {
        offsets[level] = offset;
        offset += CalculateLevelSize(level_info, level);
    }
    return offsets;
}

std::vector<u32> CalculateSliceOffsets(const ImageInfo& info) {
    ASSERT(info.type == ImageType::e3D);
    std::vector<u32> offsets;
    offsets.reserve(NumSlices(info));

    const LevelInfo level_info = MakeLevelInfo(info);
    u32 mip_offset = 0;
    for (s32 level = 0; level < info.resources.mipmaps; ++level) {
        const Extent3D tile_shift = TileShift(level_info, level);
        const Extent3D tiles = LevelTiles(level_info, level);
        const u32 gob_size_shift = tile_shift.height + GOB_SIZE_SHIFT;
        const u32 slice_size = (tiles.width * tiles.height) << gob_size_shift;
        const u32 z_mask = (1U << tile_shift.depth) - 1;
        const u32 depth = AdjustMipSize(info.size.depth, level);
        for (u32 slice = 0; slice < depth; ++slice) {
            const u32 z_low = slice & z_mask;
            const u32 z_high = slice & ~z_mask;
            offsets.push_back(mip_offset + (z_low << gob_size_shift) + (z_high * slice_size));
        }
        mip_offset += CalculateLevelSize(level_info, level);
    }
    return offsets;
}

std::vector<SubresourceBase> CalculateSliceSubresources(const ImageInfo& info) {
    ASSERT(info.type == ImageType::e3D);
    std::vector<SubresourceBase> subresources;
    subresources.reserve(NumSlices(info));
    for (s32 level = 0; level < info.resources.mipmaps; ++level) {
        const s32 depth = AdjustMipSize(info.size.depth, level);
        for (s32 slice = 0; slice < depth; ++slice) {
            subresources.emplace_back(SubresourceBase{
                .mipmap = level,
                .layer = slice,
            });
        }
    }
    return subresources;
}

u32 CalculateLevelStrideAlignment(const ImageInfo& info, u32 level) {
    const Extent2D tile_size = DefaultBlockSize(info.format);
    const Extent3D level_size = AdjustMipSize(info.size, level);
    const Extent3D num_tiles = AdjustTileSize(level_size, tile_size);
    const Extent3D block = AdjustMipBlockSize(num_tiles, info.block, level);
    const u32 bpp_log2 = BytesPerBlockLog2(info.format);
    return StrideAlignment(num_tiles, block, bpp_log2, info.tile_width_spacing);
}

PixelFormat PixelFormatFromTIC(const TICEntry& config) noexcept {
    return PixelFormatFromTextureInfo(config.format, config.r_type, config.g_type, config.b_type,
                                      config.a_type, config.srgb_conversion);
}

ImageViewType RenderTargetImageViewType(const ImageInfo& info) noexcept {
    switch (info.type) {
    case ImageType::e2D:
        return info.resources.layers > 1 ? ImageViewType::e2DArray : ImageViewType::e2D;
    case ImageType::e3D:
        return ImageViewType::e2DArray;
    case ImageType::Linear:
        return ImageViewType::e2D;
    default:
        UNIMPLEMENTED_MSG("Unimplemented image type={}", static_cast<int>(info.type));
        return ImageViewType{};
    }
}

std::vector<ImageCopy> MakeShrinkImageCopies(const ImageInfo& dst, const ImageInfo& src,
                                             SubresourceBase base) {
    ASSERT(dst.resources.mipmaps >= src.resources.mipmaps);
    ASSERT(dst.num_samples == src.num_samples);

    const bool is_dst_3d = dst.type == ImageType::e3D;
    if (is_dst_3d) {
        ASSERT(src.type == ImageType::e3D);
        ASSERT(src.resources.mipmaps == 1);
    }

    std::vector<ImageCopy> copies;
    copies.reserve(src.resources.mipmaps);
    for (s32 mipmap = 0; mipmap < src.resources.mipmaps; ++mipmap) {
        ImageCopy& copy = copies.emplace_back();
        copy.src_subresource = SubresourceLayers{
            .base_mipmap = mipmap,
            .base_layer = 0,
            .num_layers = src.resources.layers,
        };
        copy.dst_subresource = SubresourceLayers{
            .base_mipmap = base.mipmap + mipmap,
            .base_layer = is_dst_3d ? 0 : base.layer,
            .num_layers = is_dst_3d ? 1 : src.resources.layers,
        };
        copy.src_offset = Offset3D{
            .x = 0,
            .y = 0,
            .z = 0,
        };
        copy.dst_offset = Offset3D{
            .x = 0,
            .y = 0,
            .z = is_dst_3d ? base.layer : 0,
        };
        const Extent3D mip_size = AdjustMipSize(dst.size, base.mipmap + mipmap);
        copy.extent = AdjustSamplesSize(mip_size, dst.num_samples);
        if (is_dst_3d) {
            copy.extent.depth = src.size.depth;
        }
    }
    return copies;
}

bool IsValidAddress(const Tegra::MemoryManager& gpu_memory, const TICEntry& config) {
    if (config.Address() == 0) {
        return false;
    }
    if (config.Address() > (u64(1) << 48)) {
        return false;
    }
    return gpu_memory.GpuToCpuAddress(config.Address()).has_value();
}

std::vector<BufferImageCopy> UnswizzleImage(Tegra::MemoryManager& gpu_memory, GPUVAddr gpu_addr,
                                            const ImageInfo& info, std::span<u8> output) {
    const size_t guest_size_bytes = CalculateGuestSizeInBytes(info);
    const u32 bpp_log2 = BytesPerBlockLog2(info.format);
    const Extent3D size = info.size;

    if (info.type == ImageType::Linear) {
        gpu_memory.ReadBlockUnsafe(gpu_addr, output.data(), guest_size_bytes);

        ASSERT((info.pitch >> bpp_log2) << bpp_log2 == info.pitch);
        return {{
            .buffer_offset = 0,
            .buffer_size = static_cast<size_t>(info.pitch) * size.height,
            .buffer_row_length = info.pitch >> bpp_log2,
            .buffer_image_height = size.height,
            .image_subresource =
                {
                    .base_mipmap = 0,
                    .base_layer = 0,
                    .num_layers = 1,
                },
            .image_offset = {0, 0, 0},
            .image_extent = size,
        }};
    }
    const auto input_data = std::make_unique<u8[]>(guest_size_bytes);
    gpu_memory.ReadBlockUnsafe(gpu_addr, input_data.get(), guest_size_bytes);
    const std::span<const u8> input(input_data.get(), guest_size_bytes);

    const LevelInfo level_info = MakeLevelInfo(info);
    const s32 num_layers = info.resources.layers;
    const s32 num_mipmaps = info.resources.mipmaps;
    const Extent2D tile_size = DefaultBlockSize(info.format);
    const std::array level_sizes = CalculateLevelSizes(level_info, num_mipmaps);
    const Extent2D gob = GobSize(bpp_log2, info.block.height, info.tile_width_spacing);
    const u32 layer_size = std::reduce(level_sizes.begin(), level_sizes.begin() + num_mipmaps, 0);
    const u32 layer_stride = AlignLayerSize(layer_size, size, level_info.block, tile_size.height,
                                            info.tile_width_spacing);
    size_t guest_offset = 0;
    u32 host_offset = 0;
    std::vector<BufferImageCopy> copies(num_mipmaps);

    for (s32 level = 0; level < num_mipmaps; ++level) {
        const Extent3D level_size = AdjustMipSize(size, level);
        const u32 num_blocks_per_layer = NumBlocks(level_size, tile_size);
        const u32 host_bytes_per_layer = num_blocks_per_layer << bpp_log2;
        copies[level] = BufferImageCopy{
            .buffer_offset = host_offset,
            .buffer_size = static_cast<size_t>(host_bytes_per_layer) * num_layers,
            .buffer_row_length = Common::AlignUp(level_size.width, tile_size.width),
            .buffer_image_height = Common::AlignUp(level_size.height, tile_size.height),
            .image_subresource =
                {
                    .base_mipmap = level,
                    .base_layer = 0,
                    .num_layers = info.resources.layers,
                },
            .image_offset = {0, 0, 0},
            .image_extent = level_size,
        };
        const Extent3D num_tiles = AdjustTileSize(level_size, tile_size);
        const Extent3D block = AdjustMipBlockSize(num_tiles, level_info.block, level);
        const u32 stride_alignment = StrideAlignment(num_tiles, info.block, gob, bpp_log2);
        size_t guest_layer_offset = 0;

        for (s32 layer = 0; layer < info.resources.layers; ++layer) {
            const std::span<u8> dst = output.subspan(host_offset);
            const std::span<const u8> src = input.subspan(guest_offset + guest_layer_offset);
            UnswizzleTexture(dst, src, 1U << bpp_log2, num_tiles.width, num_tiles.height,
                             num_tiles.depth, block.height, block.depth, stride_alignment);
            guest_layer_offset += layer_stride;
            host_offset += host_bytes_per_layer;
        }
        guest_offset += level_sizes[level];
    }
    return copies;
}

BufferCopy UploadBufferCopy(Tegra::MemoryManager& gpu_memory, GPUVAddr gpu_addr,
                            const ImageBase& image, std::span<u8> output) {
    gpu_memory.ReadBlockUnsafe(gpu_addr, output.data(), image.guest_size_bytes);
    return BufferCopy{
        .src_offset = 0,
        .dst_offset = 0,
        .size = image.guest_size_bytes,
    };
}

void ConvertImage(std::span<const u8> input, const ImageInfo& info, std::span<u8> output,
                  std::span<BufferImageCopy> copies) {
    u32 output_offset = 0;

    const Extent2D tile_size = DefaultBlockSize(info.format);
    for (BufferImageCopy& copy : copies) {
        const u32 level = copy.image_subresource.base_mipmap;
        const Extent3D mip_size = AdjustMipSize(info.size, level);
        ASSERT(copy.image_offset == Offset3D{});
        ASSERT(copy.image_subresource.base_layer == 0);
        ASSERT(copy.image_extent == mip_size);
        ASSERT(copy.image_extent.depth == 1);
        ASSERT(copy.buffer_row_length == Common::AlignUp(mip_size.width, tile_size.width));
        ASSERT(copy.buffer_image_height == Common::AlignUp(mip_size.height, tile_size.height));

        Tegra::Texture::ASTC::Decompress(input.subspan(copy.buffer_offset), copy.image_extent.width,
                                         copy.image_extent.height,
                                         copy.image_subresource.num_layers, tile_size.width,
                                         tile_size.height, output.subspan(output_offset));
        copy.buffer_offset = output_offset;
        copy.buffer_row_length = mip_size.width;
        copy.buffer_image_height = mip_size.height;

        output_offset += copy.image_extent.width * copy.image_extent.height *
                         copy.image_subresource.num_layers * CONVERTED_BYTES_PER_BLOCK;
    }
}

std::vector<BufferImageCopy> FullDownloadCopies(const ImageInfo& info) {
    const Extent3D size = info.size;
    const u32 bytes_per_block = BytesPerBlock(info.format);
    if (info.type == ImageType::Linear) {
        ASSERT(info.pitch % bytes_per_block == 0);
        return {{
            .buffer_offset = 0,
            .buffer_size = static_cast<size_t>(info.pitch) * size.height,
            .buffer_row_length = info.pitch / bytes_per_block,
            .buffer_image_height = size.height,
            .image_subresource =
                {
                    .base_mipmap = 0,
                    .base_layer = 0,
                    .num_layers = 1,
                },
            .image_offset = {0, 0, 0},
            .image_extent = size,
        }};
    }
    UNIMPLEMENTED_IF(info.tile_width_spacing > 0);

    const s32 num_layers = info.resources.layers;
    const s32 num_mipmaps = info.resources.mipmaps;
    const Extent2D tile_size = DefaultBlockSize(info.format);

    u32 host_offset = 0;

    std::vector<BufferImageCopy> copies(num_mipmaps);
    for (s32 mipmap = 0; mipmap < num_mipmaps; ++mipmap) {
        const Extent3D level_size = AdjustMipSize(size, mipmap);
        const u32 num_blocks_per_layer = NumBlocks(level_size, tile_size);
        const u32 host_bytes_per_mipmap = num_blocks_per_layer * bytes_per_block * num_layers;
        copies[mipmap] = BufferImageCopy{
            .buffer_offset = host_offset,
            .buffer_size = host_bytes_per_mipmap,
            .buffer_row_length = level_size.width,
            .buffer_image_height = level_size.height,
            .image_subresource =
                {
                    .base_mipmap = mipmap,
                    .base_layer = 0,
                    .num_layers = info.resources.layers,
                },
            .image_offset = {0, 0, 0},
            .image_extent = level_size,
        };
        host_offset += host_bytes_per_mipmap;
    }
    return copies;
}

Extent3D MipSize(Extent3D size, u32 mipmap) {
    return AdjustMipSize(size, mipmap);
}

Extent3D MipBlockSize(const ImageInfo& info, u32 mipmap) {
    const LevelInfo level_info = MakeLevelInfo(info);
    const Extent2D tile_size = DefaultBlockSize(info.format);
    const Extent3D level_size = AdjustMipSize(info.size, mipmap);
    const Extent3D num_tiles = AdjustTileSize(level_size, tile_size);
    return AdjustMipBlockSize(num_tiles, level_info.block, mipmap);
}

std::vector<SwizzleParameters> FullUploadSwizzles(const ImageInfo& info) {
    const Extent2D tile_size = DefaultBlockSize(info.format);
    if (info.type == ImageType::Linear) {
        return std::vector{SwizzleParameters{
            .num_tiles = AdjustTileSize(info.size, tile_size),
            .block = {},
            .buffer_offset = 0,
            .mipmap = 0,
        }};
    }
    const LevelInfo level_info = MakeLevelInfo(info);
    const Extent3D size = info.size;
    const s32 num_mipmaps = info.resources.mipmaps;

    u32 guest_offset = 0;
    std::vector<SwizzleParameters> params(num_mipmaps);
    for (s32 mipmap = 0; mipmap < num_mipmaps; ++mipmap) {
        const Extent3D level_size = AdjustMipSize(size, mipmap);
        const Extent3D num_tiles = AdjustTileSize(level_size, tile_size);
        const Extent3D block = AdjustMipBlockSize(num_tiles, level_info.block, mipmap);
        params[mipmap] = SwizzleParameters{
            .num_tiles = num_tiles,
            .block = block,
            .buffer_offset = guest_offset,
            .mipmap = mipmap,
        };
        guest_offset += CalculateLevelSize(level_info, mipmap);
    }
    return params;
}

void SwizzleImage(Tegra::MemoryManager& gpu_memory, GPUVAddr gpu_addr, const ImageInfo& info,
                  std::span<const BufferImageCopy> copies, std::span<const u8> memory) {
    const bool is_pitch_linear = info.type == ImageType::Linear;
    for (const BufferImageCopy& copy : copies) {
        if (is_pitch_linear) {
            SwizzlePitchLinearImage(gpu_memory, gpu_addr, info, copy, memory);
        } else {
            SwizzleBlockLinearImage(gpu_memory, gpu_addr, info, copy, memory);
        }
    }
}

std::string CompareImageInfos(const ImageInfo& lhs, const ImageInfo& rhs) {
    std::string message;
    if (lhs.format != rhs.format) {
        message += fmt::format("- Different formats: {} vs {}\n", lhs.format, rhs.format);
    }
    if (lhs.type != rhs.type) {
        message += fmt::format("- Different types: {} vs {}\n", lhs.type, rhs.type);
    }
    if (lhs.size != rhs.size) {
        message += fmt::format("- Different sizes: {} vs {}\n", lhs.size, rhs.size);
    }
    if (lhs.resources.layers != rhs.resources.layers) {
        message += fmt::format("- Different number of layers: {} vs {}\n", lhs.resources.layers,
                               rhs.resources.layers);
    }
    if (lhs.resources.mipmaps != rhs.resources.mipmaps) {
        message += fmt::format("- Different number of mipmaps: {} vs {}\n", lhs.resources.mipmaps,
                               rhs.resources.mipmaps);
    }
    if (lhs.num_samples != rhs.num_samples) {
        message += fmt::format("- Different number of samples: {} vs {}\n", lhs.num_samples,
                               rhs.num_samples);
    }
    if (lhs.type == rhs.type) {
        if (lhs.type == ImageType::Linear) {
            if (lhs.pitch != rhs.pitch) {
                message += fmt::format("- Different pitch: {} vs {}\n", lhs.pitch, rhs.pitch);
            }
        } else {
            if (lhs.block != rhs.block) {
                message += fmt::format("- Different block size: {} vs {}\n", lhs.block, rhs.block);
            }
        }
    }
    if (message.empty()) {
        return "Identical";
    } else {
        return fmt::format("\n{}", message);
    }
}

bool IsBlockLinearSizeCompatible(const ImageInfo& lhs, const ImageInfo& rhs, u32 lhs_mipmap,
                                 u32 rhs_mipmap, bool strict_size) noexcept {
    ASSERT(lhs.type != ImageType::Linear);
    ASSERT(rhs.type != ImageType::Linear);
    if (strict_size) {
        const Extent3D lhs_size = AdjustMipSize(lhs.size, lhs_mipmap);
        const Extent3D rhs_size = AdjustMipSize(rhs.size, rhs_mipmap);
        return lhs_size.width == rhs_size.width && lhs_size.height == rhs_size.height;
    } else {
        const Extent3D lhs_size = BlockLinearAlignedSize(lhs, lhs_mipmap);
        const Extent3D rhs_size = BlockLinearAlignedSize(rhs, rhs_mipmap);
        return lhs_size.width == rhs_size.width && lhs_size.height == rhs_size.height;
    }
}

bool IsPitchLinearSameSize(const ImageInfo& lhs, const ImageInfo& rhs, bool strict_size) noexcept {
    ASSERT(lhs.type == ImageType::Linear);
    ASSERT(rhs.type == ImageType::Linear);
    if (strict_size) {
        return lhs.size.width == rhs.size.width && lhs.size.height == rhs.size.height;
    } else {
        const Extent2D lhs_size = PitchLinearAlignedSize(lhs);
        const Extent2D rhs_size = PitchLinearAlignedSize(rhs);
        return lhs_size == rhs_size;
    }
}

std::optional<OverlapResult> ResolveOverlap(const ImageInfo& new_info, GPUVAddr gpu_addr,
                                            VAddr cpu_addr, const ImageBase& overlap,
                                            bool strict_size) {
    ASSERT(new_info.type != ImageType::Linear);
    ASSERT(overlap.info.type != ImageType::Linear);
    if (!IsLayerStrideCompatible(new_info, overlap.info)) {
        return std::nullopt;
    }
    if (!IsViewCompatible(overlap.info.format, new_info.format)) {
        return std::nullopt;
    }
    if (gpu_addr == overlap.gpu_addr) {
        const std::optional solution = ResolveOverlapEqualAddress(new_info, overlap, strict_size);
        if (!solution) {
            return std::nullopt;
        }
        return OverlapResult{
            .gpu_addr = gpu_addr,
            .cpu_addr = cpu_addr,
            .resources = *solution,
        };
    }
    if (overlap.gpu_addr > gpu_addr) {
        return ResolveOverlapRightAddress(new_info, gpu_addr, cpu_addr, overlap, strict_size);
    }
    // if overlap.gpu_addr < gpu_addr
    return ResolveOverlapLeftAddress(new_info, gpu_addr, cpu_addr, overlap, strict_size);
}

bool IsLayerStrideCompatible(const ImageInfo& lhs, const ImageInfo& rhs) {
    // If either of the layer strides is zero, we can assume they are compatible
    // These images generally come from rendertargets
    if (lhs.layer_stride == 0) {
        return true;
    }
    if (rhs.layer_stride == 0) {
        return true;
    }
    // It's definitely compatible if the layer stride matches
    if (lhs.layer_stride == rhs.layer_stride) {
        return true;
    }
    // Although we also have to compare for cases where it can be unaligned
    // This can happen if the image doesn't have layers, so the stride is not aligned
    if (lhs.maybe_unaligned_layer_stride == rhs.maybe_unaligned_layer_stride) {
        return true;
    }
    return false;
}

std::optional<SubresourceBase> FindSubresource(const ImageInfo& candidate, const ImageBase& image,
                                               GPUVAddr candidate_addr, RelaxedOptions options) {
    const std::optional<SubresourceBase> base = image.TryFindBase(candidate_addr);
    if (!base) {
        return std::nullopt;
    }
    const ImageInfo& existing = image.info;
    if (False(options & RelaxedOptions::Format)) {
        if (!IsViewCompatible(existing.format, candidate.format)) {
            return std::nullopt;
        }
    }
    if (!IsLayerStrideCompatible(existing, candidate)) {
        return std::nullopt;
    }
    if (existing.type != candidate.type) {
        return std::nullopt;
    }
    if (False(options & RelaxedOptions::Samples)) {
        if (existing.num_samples != candidate.num_samples) {
            return std::nullopt;
        }
    }
    if (existing.resources.mipmaps < candidate.resources.mipmaps + base->mipmap) {
        return std::nullopt;
    }
    if (existing.type == ImageType::e3D) {
        const u32 mip_depth = std::max(1U, existing.size.depth << base->mipmap);
        if (mip_depth < candidate.size.depth + base->layer) {
            return std::nullopt;
        }
    } else {
        if (existing.resources.layers < candidate.resources.layers + base->layer) {
            return std::nullopt;
        }
    }
    const bool strict_size = False(options & RelaxedOptions::Size);
    if (!IsBlockLinearSizeCompatible(existing, candidate, base->mipmap, 0, strict_size)) {
        return std::nullopt;
    }
    // TODO: compare block sizes
    return base;
}

bool IsSubresource(const ImageInfo& candidate, const ImageBase& image, GPUVAddr candidate_addr,
                   RelaxedOptions options) {
    return FindSubresource(candidate, image, candidate_addr, options).has_value();
}

void DeduceBlitImages(ImageInfo& dst_info, ImageInfo& src_info, const ImageBase* dst,
                      const ImageBase* src) {
    if (src && GetFormatType(src->info.format) != SurfaceType::ColorTexture) {
        src_info.format = src->info.format;
    }
    if (dst && GetFormatType(dst->info.format) != SurfaceType::ColorTexture) {
        dst_info.format = dst->info.format;
    }
    if (!dst && src && GetFormatType(src->info.format) != SurfaceType::ColorTexture) {
        dst_info.format = src->info.format;
    }
    if (!src && dst && GetFormatType(dst->info.format) != SurfaceType::ColorTexture) {
        src_info.format = src->info.format;
    }
}

using P = PixelFormat;

static_assert(CalculateLevelSize(LevelInfo{{1920, 1080}, {0, 2, 0}, {1, 1}, 2, 0}, 0) == 0x7f8000);
static_assert(CalculateLevelSize(LevelInfo{{32, 32}, {0, 0, 4}, {1, 1}, 4, 0}, 0) == 0x4000);

static_assert(CalculateLevelOffset(P::R8_SINT, {1920, 1080}, {0, 2}, 1, 0, 7) == 0x2afc00);
static_assert(CalculateLevelOffset(P::ASTC_2D_12X12_UNORM, {8192, 4096}, {0, 2}, 1, 0, 12) ==
              0x50d200);

static_assert(CalculateLevelOffset(P::A8B8G8R8_UNORM, {1024, 1024}, {0, 4}, 1, 0, 0) == 0);
static_assert(CalculateLevelOffset(P::A8B8G8R8_UNORM, {1024, 1024}, {0, 4}, 1, 0, 1) == 0x400000);
static_assert(CalculateLevelOffset(P::A8B8G8R8_UNORM, {1024, 1024}, {0, 4}, 1, 0, 2) == 0x500000);
static_assert(CalculateLevelOffset(P::A8B8G8R8_UNORM, {1024, 1024}, {0, 4}, 1, 0, 3) == 0x540000);
static_assert(CalculateLevelOffset(P::A8B8G8R8_UNORM, {1024, 1024}, {0, 4}, 1, 0, 4) == 0x550000);
static_assert(CalculateLevelOffset(P::A8B8G8R8_UNORM, {1024, 1024}, {0, 4}, 1, 0, 5) == 0x554000);
static_assert(CalculateLevelOffset(P::A8B8G8R8_UNORM, {1024, 1024}, {0, 4}, 1, 0, 6) == 0x555000);
static_assert(CalculateLevelOffset(P::A8B8G8R8_UNORM, {1024, 1024}, {0, 4}, 1, 0, 7) == 0x555400);
static_assert(CalculateLevelOffset(P::A8B8G8R8_UNORM, {1024, 1024}, {0, 4}, 1, 0, 8) == 0x555600);
static_assert(CalculateLevelOffset(P::A8B8G8R8_UNORM, {1024, 1024}, {0, 4}, 1, 0, 9) == 0x555800);

constexpr u32 ValidateLayerSize(PixelFormat format, u32 width, u32 height, u32 block_height,
                                u32 tile_width_spacing, u32 level) {
    const Extent3D size{width, height, 1};
    const Extent3D block{0, block_height, 0};
    const u32 offset = CalculateLevelOffset(format, size, block, 1, tile_width_spacing, level);
    return AlignLayerSize(offset, size, block, DefaultBlockHeight(format), tile_width_spacing);
}

static_assert(ValidateLayerSize(P::ASTC_2D_12X12_UNORM, 8192, 4096, 2, 0, 12) == 0x50d800);
static_assert(ValidateLayerSize(P::A8B8G8R8_UNORM, 1024, 1024, 2, 0, 10) == 0x556000);
static_assert(ValidateLayerSize(P::BC3_UNORM, 128, 128, 2, 0, 8) == 0x6000);

static_assert(ValidateLayerSize(P::A8B8G8R8_UNORM, 518, 572, 4, 3, 1) == 0x190000,
              "Tile width spacing is not working");
static_assert(ValidateLayerSize(P::BC5_UNORM, 1024, 1024, 3, 4, 11) == 0x160000,
              "Compressed tile width spacing is not working");

} // namespace VideoCommon
