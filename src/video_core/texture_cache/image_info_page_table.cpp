// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <memory>
#include <utility>

#include "common/common_types.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_info_page_table.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

bool ImageInfoPageTable::Iterator::Valid() const noexcept {
    return id != INVALID_ID;
}

ImageInfo& ImageInfoPageTable::Iterator::Info() const noexcept {
    return table->entries[id].info;
}

void ImageInfoPageTable::Iterator::Next() noexcept {
    const Entry* const entries = table->entries.get();
    do {
        id = entries[id].next;
    } while (id != INVALID_ID && entries[id].gpu_addr != gpu_addr);
}

ImageInfoPageTable::ImageInfoPageTable() {
    id_page_table.fill(INVALID_ID);
}

void ImageInfoPageTable::Push(GPUVAddr gpu_addr, const ImageInfo& info) {
    ++free_id;

    const u32 next = std::exchange(id_page_table[gpu_addr >> PAGE_BITS], free_id);
    entries[free_id] = Entry{
        .info = info,
        .next = next,
        .gpu_addr = gpu_addr,
    };
}

ImageInfoPageTable::Iterator ImageInfoPageTable::AddressLinkedList(GPUVAddr gpu_addr) {
    u32 id = id_page_table[gpu_addr >> PAGE_BITS];
    while (id != INVALID_ID && entries[id].gpu_addr != gpu_addr) {
        id = entries[id].next;
    }
    Iterator it;
    it.table = this;
    it.gpu_addr = gpu_addr;
    it.id = id;
    return it;
}

void ImageInfoPageTable::Prepare(size_t max_entries) {
    free_id = INVALID_ID;
    if (entries_capacity < max_entries) {
        entries_capacity = max_entries;
        entries = std::make_unique<Entry[]>(max_entries);
    }
}

void ImageInfoPageTable::Restore(const Entry& entry) noexcept {
    id_page_table[entry.gpu_addr >> PAGE_BITS] = INVALID_ID;
}

ImageInfoPageTable::Entry* ImageInfoPageTable::begin() noexcept {
    return entries.get();
}

ImageInfoPageTable::Entry* ImageInfoPageTable::end() noexcept {
    // Unsigned integer overflow here is intentional
    const u32 num_entries = free_id + 1U;
    return entries.get() + num_entries;
}

} // namespace VideoCommon
