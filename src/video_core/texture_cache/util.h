// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <optional>
#include <span>

#include "common/common_types.h"

#include "video_core/engines/maxwell_3d.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/image_base.h"
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

[[nodiscard]] u32 CalculateGuestSizeInBytes(const ImageInfo& info) noexcept;

[[nodiscard]] u32 CalculateUnswizzledSizeBytes(const ImageInfo& info) noexcept;

[[nodiscard]] u32 CalculateConvertedSizeBytes(const ImageInfo& info) noexcept;

[[nodiscard]] u32 CalculateLayerStride(const ImageInfo& info) noexcept;

[[nodiscard]] u32 CalculateLayerSize(const ImageInfo& info) noexcept;

[[nodiscard]] std::array<u32, MAX_MIPMAP> CalculateMipmapOffsets(const ImageInfo& info) noexcept;

[[nodiscard]] std::vector<u32> CalculateSliceOffsets(const ImageInfo& info);

[[nodiscard]] std::vector<SubresourceBase> CalculateSliceSubresources(const ImageInfo& info);

[[nodiscard]] VideoCore::Surface::PixelFormat PixelFormatFromTIC(
    const Tegra::Texture::TICEntry& config) noexcept;

[[nodiscard]] ImageViewType RenderTargetImageViewType(const ImageInfo& info) noexcept;

[[nodiscard]] std::vector<ImageCopy> MakeShrinkImageCopies(const ImageInfo& dst,
                                                           const ImageInfo& src,
                                                           SubresourceBase base);

[[nodiscard]] bool IsValidAddress(const Tegra::MemoryManager& gpu_memory, const TICEntry& config);

[[nodiscard]] std::vector<BufferImageCopy> UnswizzleImage(Tegra::MemoryManager& gpu_memory,
                                                          GPUVAddr gpu_addr, const ImageInfo& info,
                                                          std::span<u8> memory);

void ConvertImage(std::span<const u8> input, const ImageInfo& info, std::span<u8> output,
                  std::span<BufferImageCopy> copies);

[[nodiscard]] std::vector<BufferImageCopy> FullDownloadCopies(const ImageInfo& info);

[[nodiscard]] Extent3D MipmapSize(const ImageInfo& info, u32 mipmap);

[[nodiscard]] Extent3D MipBlockSize(const ImageInfo& info, u32 mipmap);

[[nodiscard]] std::vector<SwizzleParameters> FullUploadSwizzles(const ImageInfo& info);

void SwizzleImage(Tegra::MemoryManager& gpu_memory, GPUVAddr gpu_addr, const ImageInfo& info,
                  std::span<const BufferImageCopy> copies, std::span<const u8> memory);

[[nodiscard]] std::string CompareImageInfos(const ImageInfo& lhs, const ImageInfo& rhs);

[[nodiscard]] bool IsBlockLinearSizeCompatible(const ImageInfo& new_info,
                                               const ImageInfo& overlap_info, u32 new_mipmap,
                                               u32 overlap_mipmap, bool strict_size) noexcept;

[[nodiscard]] bool IsPitchLinearSameSize(const ImageInfo& lhs, const ImageInfo& rhs,
                                         bool strict_size) noexcept;

[[nodiscard]] std::optional<OverlapResult> ResolveOverlap(const ImageInfo& new_info,
                                                          GPUVAddr gpu_addr, VAddr cpu_addr,
                                                          const ImageBase& overlap,
                                                          bool strict_size);

[[nodiscard]] bool IsLayerStrideCompatible(const ImageInfo& lhs, const ImageInfo& rhs);

[[nodiscard]] std::optional<SubresourceBase> FindSubresource(const ImageInfo& candidate,
                                                             const ImageBase& image,
                                                             GPUVAddr candidate_addr,
                                                             RelaxedOptions options);

[[nodiscard]] bool IsSubresource(const ImageInfo& candidate, const ImageBase& image,
                                 GPUVAddr candidate_addr, RelaxedOptions options);

} // namespace VideoCommon
