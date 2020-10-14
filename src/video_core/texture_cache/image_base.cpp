// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <optional>
#include <vector>

#include "common/common_types.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/image_view_info.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

[[nodiscard]] constexpr std::pair<u32, u32> LayerMipOffset(u32 diff, u32 layer_stride) {
    if (layer_stride == 0) {
        return {0, diff};
    } else {
        return {diff / layer_stride, diff % layer_stride};
    }
}

ImageBase::ImageBase(const ImageInfo& info_, GPUVAddr gpu_addr_, VAddr cpu_addr_)
    : info{info_}, guest_size_bytes{CalculateGuestSizeInBytes(info)},
      unswizzled_size_bytes{CalculateUnswizzledSizeBytes(info)},
      converted_size_bytes{CalculateConvertedSizeBytes(info)}, gpu_addr{gpu_addr_},
      cpu_addr{cpu_addr_}, cpu_addr_end{cpu_addr + guest_size_bytes},
      mipmap_offsets{CalculateMipmapOffsets(info)} {
    if (info.type == ImageType::e3D) {
        slice_offsets = CalculateSliceOffsets(info);
        slice_subresources = CalculateSliceSubresources(info);
    }
}

bool ImageBase::Overlaps(VAddr overlap_cpu_addr, size_t overlap_size) const noexcept {
    const VAddr overlap_end = overlap_cpu_addr + overlap_size;
    return cpu_addr < overlap_end && overlap_cpu_addr < cpu_addr_end;
}

std::optional<SubresourceBase> ImageBase::FindSubresourceFromAddress(
    GPUVAddr rhs_addr) const noexcept {
    if (rhs_addr < gpu_addr) {
        // Subresource address can't be lower than the base
        return std::nullopt;
    }
    const u32 diff = static_cast<u32>(rhs_addr - gpu_addr);
    if (diff > guest_size_bytes) {
        // This can happen when two CPU addresses are used for different GPU addresses
        return std::nullopt;
    }
    if (info.type != ImageType::e3D) {
        const auto [layer, mip_offset] = LayerMipOffset(diff, info.layer_stride);
        const auto end = mipmap_offsets.begin() + info.resources.mipmaps;
        const auto it = std::find(mipmap_offsets.begin(), end, mip_offset);
        if (it == end) {
            return std::nullopt;
        }
        return SubresourceBase{
            .mipmap = static_cast<u32>(std::distance(mipmap_offsets.begin(), it)),
            .layer = layer,
        };
    } else {
        // TODO: Consider using binary_search after a threshold
        const auto it = std::ranges::find(slice_offsets, diff);
        if (it == slice_offsets.cend()) {
            return std::nullopt;
        }
        return slice_subresources[std::distance(slice_offsets.begin(), it)];
    }
}

ImageViewId ImageBase::FindView(const ImageViewInfo& info) const noexcept {
    const auto it = std::ranges::find(image_view_infos, info);
    if (it == image_view_infos.end()) {
        return ImageViewId{};
    }
    return image_view_ids[std::distance(image_view_infos.begin(), it)];
}

void ImageBase::InsertView(const ImageViewInfo& info, ImageViewId image_view_id) {
    image_view_infos.push_back(info);
    image_view_ids.push_back(image_view_id);
}

} // namespace VideoCommon
