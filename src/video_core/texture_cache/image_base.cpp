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
namespace {
/// Returns the base layer and mip level offset
[[nodiscard]] std::pair<u32, u32> LayerMipOffset(u32 diff, u32 layer_stride) {
    if (layer_stride == 0) {
        return {0, diff};
    } else {
        return {diff / layer_stride, diff % layer_stride};
    }
}

[[nodiscard]] bool ValidateLayers(const SubresourceLayers& layers, const ImageInfo& info) {
    return layers.base_mipmap < info.resources.mipmaps &&
           layers.base_layer + layers.num_layers <= info.resources.layers;
}

[[nodiscard]] bool ValidateCopy(const ImageCopy& copy, const ImageInfo& dst, const ImageInfo& src) {
    const Extent3D src_size = MipSize(src.size, copy.src_subresource.base_mipmap);
    const Extent3D dst_size = MipSize(dst.size, copy.dst_subresource.base_mipmap);
    if (!ValidateLayers(copy.src_subresource, src)) {
        return false;
    }
    if (!ValidateLayers(copy.dst_subresource, dst)) {
        return false;
    }
    if (copy.src_offset.x + copy.extent.width > src_size.width ||
        copy.src_offset.y + copy.extent.height > src_size.height ||
        copy.src_offset.z + copy.extent.depth > src_size.depth) {
        return false;
    }
    if (copy.dst_offset.x + copy.extent.width > dst_size.width ||
        copy.dst_offset.y + copy.extent.height > dst_size.height ||
        copy.dst_offset.z + copy.extent.depth > dst_size.depth) {
        return false;
    }
    return true;
}
} // Anonymous namespace

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
    GPUVAddr other_addr) const noexcept {
    if (other_addr < gpu_addr) {
        // Subresource address can't be lower than the base
        return std::nullopt;
    }
    const u32 diff = static_cast<u32>(other_addr - gpu_addr);
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

void AddImageAlias(ImageBase& lhs, ImageBase& rhs, ImageId lhs_id, ImageId rhs_id) {
    static constexpr auto OPTIONS = RelaxedOptions::Size;
    if (lhs.gpu_addr > rhs.gpu_addr) {
        // If lhs is not on the left, flip them
        return AddImageAlias(rhs, lhs, rhs_id, lhs_id);
    }
    UNIMPLEMENTED_IF(rhs.info.resources.mipmaps != 1);

    const SubresourceBase base = FindSubresource(rhs.info, lhs, rhs.gpu_addr, OPTIONS).value();
    const Extent3D lhs_size = MipSize(lhs.info.size, base.mipmap);
    const Extent3D rhs_size = rhs.info.size;
    const Extent3D size{
        .width = std::min(lhs_size.width, rhs_size.width),
        .height = std::min(lhs_size.height, rhs_size.height),
        .depth = std::min(lhs_size.depth, rhs_size.depth),
    };
    const bool is_lhs_3d = lhs.info.type == ImageType::e3D;
    const bool is_rhs_3d = rhs.info.type == ImageType::e3D;
    const Offset3D lhs_offset{0, 0, 0};
    const Offset3D rhs_offset{0, 0, is_rhs_3d ? base.layer : 0};
    const u32 lhs_layers = is_lhs_3d ? 1 : lhs.info.resources.layers - base.layer;
    const u32 rhs_layers = is_rhs_3d ? 1 : rhs.info.resources.layers;
    const u32 num_layers = std::min(lhs_layers, rhs_layers);
    const SubresourceLayers lhs_subresource{
        .base_mipmap = 0,
        .base_layer = 0,
        .num_layers = num_layers,
    };
    const SubresourceLayers rhs_subresource{
        .base_mipmap = base.mipmap,
        .base_layer = is_rhs_3d ? 0 : base.layer,
        .num_layers = num_layers,
    };
    const ImageCopy to_lhs_copy{
        .src_subresource = lhs_subresource,
        .dst_subresource = rhs_subresource,
        .src_offset = lhs_offset,
        .dst_offset = rhs_offset,
        .extent = size,
    };
    const ImageCopy to_rhs_copy{
        .src_subresource = rhs_subresource,
        .dst_subresource = lhs_subresource,
        .src_offset = rhs_offset,
        .dst_offset = lhs_offset,
        .extent = size,
    };
    const AliasedImage lhs_alias{
        .copy = to_lhs_copy,
        .id = rhs_id,
    };
    const AliasedImage rhs_alias{
        .copy = to_rhs_copy,
        .id = lhs_id,
    };
    ASSERT_MSG(ValidateCopy(to_lhs_copy, lhs.info, rhs.info), "Invalid RHS to LHS copy");
    ASSERT_MSG(ValidateCopy(to_rhs_copy, rhs.info, lhs.info), "Invalid LHS to RHS copy");

    lhs.aliased_images.push_back(lhs_alias);
    rhs.aliased_images.push_back(rhs_alias);
}

} // namespace VideoCommon
