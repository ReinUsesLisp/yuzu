// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <limits>
#include <utility>
#include <vector>

#include "common/assert.h"
#include "common/common_types.h"
#include "video_core/texture_cache/image_alloc_page_table.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

namespace {

constexpr u32 INVALID_ID = std::numeric_limits<u32>::max();

constexpr size_t Page(GPUVAddr gpu_addr) {
    return static_cast<size_t>(gpu_addr >> ImageAllocPageTable::PAGE_BITS);
}

} // Anonymous namespace

ImageAllocPageTable::ImageAllocPageTable() {
    page_table.fill(INVALID_ID);
}

ImageAllocPageTable::~ImageAllocPageTable() = default;

ImageAllocId ImageAllocPageTable::Find(GPUVAddr gpu_addr) const {
    u32 id = page_table[Page(gpu_addr)];
    while (id != INVALID_ID) {
        const Entry& entry = entries[id];
        if (entry.gpu_addr == gpu_addr) {
            return entry.value;
        }
        id = entry.next;
    }
    return ImageAllocId{};
}

void ImageAllocPageTable::PushFront(GPUVAddr gpu_addr, ImageAllocId alloc_id) {
    const u32 free_id = FreeId();
    const u32 next_id = std::exchange(page_table[Page(gpu_addr)], free_id);
    entries[free_id] = Entry{
        .gpu_addr = gpu_addr,
        .next = next_id,
        .value = alloc_id,
    };
}

void ImageAllocPageTable::Erase(GPUVAddr gpu_addr) {
    u32 id = page_table[Page(gpu_addr)];
    ASSERT(id != INVALID_ID);

    if (entries[id].gpu_addr == gpu_addr) {
        page_table[Page(gpu_addr)] = entries[id].next;
        free_list.push_back(id);
        return;
    }

    u32 prev = id;
    id = entries[id].next;
    while (id != INVALID_ID) {
        if (entries[id].gpu_addr == gpu_addr) {
            entries[prev].next = entries[id].next;
            free_list.push_back(id);
            return;
        }
        prev = id;
        id = entries[id].next;
    }

    UNREACHABLE();
}

u32 ImageAllocPageTable::FreeId() {
    if (free_list.empty()) {
        const u32 id = static_cast<u32>(entries.size());
        entries.emplace_back();
        return id;
    } else {
        const u32 id = free_list.back();
        free_list.pop_back();
        return id;
    }
}

} // namespace VideoCommon
