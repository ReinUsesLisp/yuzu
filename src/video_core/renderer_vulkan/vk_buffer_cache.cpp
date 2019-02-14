// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstring>
#include <memory>
#include <optional>
#include <tuple>
#include "common/alignment.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"

namespace Vulkan {

VKBufferCache::VKBufferCache(Core::System& system, RasterizerVulkan& rasterizer,
                             VKResourceManager& resource_manager, const VKDevice& device,
                             VKMemoryManager& memory_manager, VKScheduler& sched, u64 size)
    : RasterizerCache{rasterizer}, system{system} {
    const auto usage = vk::BufferUsageFlagBits::eVertexBuffer |
                       vk::BufferUsageFlagBits::eIndexBuffer |
                       vk::BufferUsageFlagBits::eUniformBuffer;
    const auto access = vk::AccessFlagBits::eVertexAttributeRead | vk::AccessFlagBits::eIndexRead |
                        vk::AccessFlagBits::eUniformRead;
    stream_buffer =
        std::make_unique<VKStreamBuffer>(resource_manager, device, memory_manager, sched, size,
                                         usage, access, vk::PipelineStageFlagBits::eAllCommands);
}

VKBufferCache::~VKBufferCache() = default;

std::tuple<u64, vk::Buffer> VKBufferCache::UploadMemory(Tegra::GPUVAddr gpu_addr, std::size_t size,
                                                        u64 alignment, bool cache) {
    const auto cpu_addr{system.GPU().MemoryManager().GpuToCpuAddress(gpu_addr)};
    ASSERT(cpu_addr);

    // Cache management is a big overhead, so only cache entries with a given size.
    // TODO: Figure out which size is the best for given games.
    cache &= size >= 2048;

    if (cache) {
        if (auto entry = TryGet(*cpu_addr); entry) {
            if (entry->size >= size && entry->alignment == alignment) {
                return {entry->offset, entry->buffer};
            }
            Unregister(entry);
        }
    }

    AlignBuffer(alignment);
    const u64 uploaded_offset = buffer_offset;

    Memory::ReadBlock(*cpu_addr, buffer_ptr, size);

    buffer_ptr += size;
    buffer_offset += size;

    if (cache) {
        auto entry = std::make_shared<CachedBufferEntry>();
        entry->offset = uploaded_offset;
        entry->buffer = buffer_handle;
        entry->size = size;
        entry->alignment = alignment;
        entry->addr = *cpu_addr;
        Register(entry);
    }

    return {uploaded_offset, buffer_handle};
}

std::tuple<u64, vk::Buffer> VKBufferCache::UploadHostMemory(const u8* raw_pointer, std::size_t size,
                                                            u64 alignment) {
    AlignBuffer(alignment);
    std::memcpy(buffer_ptr, raw_pointer, size);
    const u64 uploaded_offset = buffer_offset;

    buffer_ptr += size;
    buffer_offset += size;
    return {uploaded_offset, buffer_handle};
}

std::tuple<u8*, u64, vk::Buffer> VKBufferCache::ReserveMemory(std::size_t size, u64 alignment) {
    AlignBuffer(alignment);
    u8* const uploaded_ptr = buffer_ptr;
    const u64 uploaded_offset = buffer_offset;

    buffer_ptr += size;
    buffer_offset += size;
    return {uploaded_ptr, uploaded_offset, buffer_handle};
}

void VKBufferCache::Reserve(std::size_t max_size) {
    bool invalidate;
    std::tie(buffer_ptr, buffer_offset_base, buffer_handle, invalidate) =
        stream_buffer->Reserve(max_size, false);
    buffer_offset = buffer_offset_base;

    if (invalidate) {
        InvalidateAll();
    }
}

VKExecutionContext VKBufferCache::Send(VKExecutionContext exctx) {
    return stream_buffer->Send(exctx, buffer_offset - buffer_offset_base);
}

void VKBufferCache::AlignBuffer(std::size_t alignment) {
    // Align the offset, not the mapped pointer
    const u64 offset_aligned = Common::AlignUp(buffer_offset, alignment);
    buffer_ptr += offset_aligned - buffer_offset;
    buffer_offset = offset_aligned;
}

} // namespace Vulkan