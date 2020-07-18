// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <vector>

#include "common/assert.h"
#include "common/common_types.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

class ImageAllocPageTable {
    struct Entry {
        GPUVAddr gpu_addr;
        u32 next;
        ImageAllocId value;
    };

public:
    static constexpr size_t ADDRESS_SPACE_BITS = 40;
    static constexpr size_t PAGE_BITS = 20;
    static constexpr size_t NUM_PAGES = size_t(1) << (ADDRESS_SPACE_BITS - PAGE_BITS);

    explicit ImageAllocPageTable();
    ~ImageAllocPageTable();

    [[nodiscard]] ImageAllocId Find(GPUVAddr gpu_addr) const;

    void PushFront(GPUVAddr gpu_addr, ImageAllocId alloc_id);

    void Erase(GPUVAddr gpu_addr);

private:
    [[nodiscard]] u32 FreeId();

    std::vector<Entry> entries;
    std::vector<u32> free_list;
    std::array<u32, NUM_PAGES> page_table;
};

} // namespace VideoCommon
