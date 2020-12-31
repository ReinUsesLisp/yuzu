// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <climits>
#include <vector>

#include "common/common_types.h"

#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class VKScheduler;

struct StagingBufferRef {
    VkBuffer buffer;
    std::span<u8> mapped_span;
};

class StagingBufferPool {
public:
    explicit StagingBufferPool(const Device& device, VKMemoryManager& memory_manager,
                               VKScheduler& scheduler);
    ~StagingBufferPool();

    StagingBufferRef Request(size_t size, bool host_visible);

    void TickFrame();

private:
    struct StagingBuffer {
        vk::Buffer buffer;
        MemoryCommit commit;
        std::span<u8> mapped_span;
        u64 tick = 0;

        StagingBufferRef Ref() const noexcept {
            return StagingBufferRef{
                .buffer = *buffer,
                .mapped_span = mapped_span,
            };
        }
    };

    struct StagingBuffers {
        std::vector<StagingBuffer> entries;
        size_t delete_index = 0;
        size_t iterate_index = 0;
    };

    static constexpr size_t NUM_LEVELS = sizeof(size_t) * CHAR_BIT;
    using StagingBuffersCache = std::array<StagingBuffers, NUM_LEVELS>;

    std::optional<StagingBufferRef> TryGetReservedBuffer(size_t size, bool host_visible);

    StagingBufferRef CreateStagingBuffer(size_t size, bool host_visible);

    StagingBuffersCache& GetCache(bool host_visible);

    void ReleaseCache(bool host_visible);

    void ReleaseLevel(StagingBuffersCache& cache, size_t log2);

    const Device& device;
    VKMemoryManager& memory_manager;
    VKScheduler& scheduler;

    StagingBuffersCache host_staging_buffers;
    StagingBuffersCache device_staging_buffers;

    size_t current_delete_level = 0;
    u64 buffer_index = 0;
};

} // namespace Vulkan
