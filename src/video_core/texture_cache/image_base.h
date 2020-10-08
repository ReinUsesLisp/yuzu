// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <optional>
#include <vector>

#include "common/common_funcs.h"
#include "common/common_types.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

enum class ImageFlagBits : u32 {
    AcceleratedUpload = 1 << 0, ///< Upload can be accelerated in the GPU
    Converted = 1 << 1,   ///< Guest format is not supported natively and it has to be converted
    CpuModified = 1 << 2, ///< Contents have been modified from the CPU
    GpuModified = 1 << 3, ///< Contents have been modified from the GPU
    Tracked = 1 << 4,     ///< Writes and reads are being hooked from the CPU JIT
    Strong = 1 << 5,      ///< Exists in the image table, the dimensions are can be trusted
    Picked = 1 << 6,      ///< Temporary flag to mark the image as picked
};
DECLARE_ENUM_FLAG_OPERATORS(ImageFlagBits)

struct ImageViewInfo;

struct ImageBase {
    explicit ImageBase(const ImageInfo& info_, GPUVAddr gpu_addr_, VAddr cpu_addr_);

    [[nodiscard]] bool Overlaps(VAddr overlap_cpu_addr, size_t overlap_size) const noexcept;

    [[nodiscard]] std::optional<SubresourceBase> FindSubresourceFromAddress(
        GPUVAddr rhs_addr) const noexcept;

    [[nodiscard]] ImageViewId FindView(const ImageViewInfo& info) const noexcept;

    void InsertView(const ImageViewInfo& info, ImageViewId image_view_id);

    ImageInfo info;

    u32 guest_size_bytes = 0;
    u32 unswizzled_size_bytes = 0;
    u32 converted_size_bytes = 0;

    GPUVAddr gpu_addr = 0;
    VAddr cpu_addr = 0;
    VAddr cpu_addr_end = 0;

    ImageFlagBits flags = ImageFlagBits::CpuModified;
    u64 invalidation_tick = 0;
    u64 modification_tick = 0;

    std::array<u32, MAX_MIPMAP> mipmap_offsets{};

    std::vector<ImageViewInfo> image_view_infos;
    std::vector<ImageViewId> image_view_ids;
};

struct ImageAllocBase {
    std::vector<ImageId> images;
};

} // namespace VideoCommon
