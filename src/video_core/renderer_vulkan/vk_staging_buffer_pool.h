// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <climits>
#include <vector>

#include "common/common_types.h"

#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/wrapper.h"
#include "video_core/host_buffer_type.h"

namespace Vulkan {

class VKDevice;
class VKFenceWatch;
class VKScheduler;

struct Buffer {
    vk::Buffer handle;
    VKMemoryCommit commit;

    VkBuffer Handle() const noexcept {
        return *handle;
    }

    MemoryMap Map(u64 size, u64 offset = 0) const {
        return commit->Map(size, offset);
    }
};

class VKStagingBufferPool final {
public:
    using Buffer = ::Vulkan::Buffer;

    explicit VKStagingBufferPool(const VKDevice& device, VKMemoryManager& memory_manager,
                                 VKScheduler& scheduler);
    ~VKStagingBufferPool();

    Buffer& GetUnusedBuffer(size_t size, VideoCommon::HostBufferType type);

    void TickFrame();

private:
    struct StagingBuffer {
        Buffer buffer;
        VKFenceWatch watch;
        u64 last_epoch = 0;
    };

    struct StagingBuffers {
        std::vector<StagingBuffer> entries;
        size_t delete_index = 0;
    };

    static constexpr size_t NUM_LEVELS = sizeof(size_t) * CHAR_BIT;
    using StagingBuffersCache = std::array<StagingBuffers, NUM_LEVELS>;

    Buffer* TryGetReservedBuffer(size_t size, VideoCommon::HostBufferType type);

    Buffer& CreateStagingBuffer(size_t size, VideoCommon::HostBufferType type);

    StagingBuffersCache& Cache(VideoCommon::HostBufferType type);

    void ReleaseCache(VideoCommon::HostBufferType type);

    u64 ReleaseLevel(StagingBuffersCache& cache, size_t log2);

    const VKDevice& device;
    VKMemoryManager& memory_manager;
    VKScheduler& scheduler;

    std::array<StagingBuffersCache, VideoCommon::NUM_HOST_BUFFER_TYPES> caches;

    u64 epoch = 0;

    size_t current_delete_level = 0;
};

} // namespace Vulkan
