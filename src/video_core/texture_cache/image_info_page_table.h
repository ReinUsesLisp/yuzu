// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <limits>
#include <memory>

#include "common/common_types.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

class ImageInfoPageTable {
    static constexpr size_t ADDRESS_SPACE_BITS = 40;
    static constexpr size_t PAGE_BITS = 20;
    static constexpr size_t NUM_PAGES = size_t(1) << (ADDRESS_SPACE_BITS - PAGE_BITS);

    static constexpr u32 INVALID_ID = std::numeric_limits<u32>::max();

public:
    struct Entry {
        ImageInfo info;
        u32 next;
        GPUVAddr gpu_addr;
    };

    class Iterator {
        friend ImageInfoPageTable;

    public:
        [[nodiscard]] bool Valid() const noexcept;

        [[nodiscard]] ImageInfo& Info() const noexcept;

        void Next() noexcept;

    private:
        ImageInfoPageTable* table;
        GPUVAddr gpu_addr;
        u32 id;
    };

    explicit ImageInfoPageTable();

    void Push(GPUVAddr gpu_addr, const ImageInfo& info);

    Iterator AddressLinkedList(GPUVAddr gpu_addr);

    void Prepare(size_t max_entries);

    void Restore(const Entry& entry) noexcept;

    [[nodiscard]] Entry* begin() noexcept;

    [[nodiscard]] Entry* end() noexcept;

private:
    size_t entries_capacity = 0;
    std::unique_ptr<Entry[]> entries;

    u32 free_id = INVALID_ID;
    std::array<u32, NUM_PAGES> id_page_table;
};

} // namespace VideoCommon
