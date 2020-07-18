// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <optional>
#include <span>

#include "common/common_types.h"

#include "video_core/engines/maxwell_3d.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/types.h"
#include "video_core/textures/texture.h"

namespace VideoCommon {

using Tegra::Texture::TICEntry;

struct OverlapResult {
    GPUVAddr gpu_addr;
    VAddr cpu_addr;
    SubresourceExtent resources;
};

u32 CalculateGuestSizeInBytes(const ImageInfo& info) noexcept;

u32 CalculateHostSizeInBytes(const ImageInfo& info) noexcept;

u32 CalculateLayerStride(const ImageInfo& info) noexcept;

u32 CalculateLayerSize(const ImageInfo& info) noexcept;

std::array<u32, MAX_MIPMAP> CalculateMipmapOffsets(const ImageInfo& info) noexcept;

GPUVAddr CalculateBaseAddress(const TICEntry& config);

VideoCore::Surface::PixelFormat PixelFormatFromTIC(const Tegra::Texture::TICEntry& config) noexcept;

ImageViewType RenderTargetImageViewType(const ImageInfo& info) noexcept;

bool IsFullyCompatible(const ImageInfo& lhs, const ImageInfo& rhs) noexcept;

/// @note This doesn't check for format compatibilities
bool IsRenderTargetShrinkCompatible(const ImageInfo& dst, const ImageInfo& src, u32 level) noexcept;

std::vector<ImageCopy> MakeShrinkImageCopies(const ImageInfo& dst, const ImageInfo& src,
                                             SubresourceBase base);

bool IsValid(const Tegra::MemoryManager& gpu_memory, const TICEntry& config);

std::vector<BufferImageCopy> UnswizzleImage(Tegra::MemoryManager& gpu_memory, GPUVAddr gpu_addr,
                                            const ImageInfo& info, std::span<u8> memory);

std::vector<BufferImageCopy> FullDownloadCopies(const ImageInfo& info);

Extent3D MipmapSize(const ImageInfo& info, u32 mipmap);

Extent3D MipmapBlockSize(const ImageInfo& info, u32 mipmap);

Extent3D AlignedSize(const ImageInfo& info, u32 mipmap);

bool SizeMatches(const ImageInfo& aligned, const ImageInfo& unaligned, u32 unaligned_mipmap,
                 bool strict_size);

std::vector<SwizzleParameters> FullUploadSwizzles(const ImageInfo& info);

void SwizzleImage(Tegra::MemoryManager& gpu_memory, GPUVAddr gpu_addr, const ImageInfo& info,
                  std::span<const BufferImageCopy> copies, std::span<const u8> memory);

std::string CompareImageInfos(const ImageInfo& lhs, const ImageInfo& rhs);

std::optional<OverlapResult> ResolveOverlap(const ImageInfo& new_info, GPUVAddr gpu_addr,
                                            VAddr cpu_addr, const ImageBase& overlap,
                                            bool strict_size);

} // namespace VideoCommon
