// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <tuple>
#include <vector>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

class VKDevice;
class VKFence;
class VKResourceManager;

class StreamBufferResource;

class VKStreamBuffer {
public:
    explicit VKStreamBuffer(VKResourceManager& resource_manager, const VKDevice& device,
                            VKMemoryManager& memory_manager, VKScheduler& sched, u64 size,
                            vk::BufferUsageFlags usage, vk::AccessFlags access,
                            vk::PipelineStageFlags pipeline_stage);
    ~VKStreamBuffer();

    /**
     * Reserves a region of memory from the stream buffer.
     * @param size Size to reserve.
     * @param keep_in_host Mapped buffer will be in host memory, skipping the copy to device local.
     * @returns A tuple in the following order: Raw memory pointer (with offset added), buffer
     * offset, Vulkan buffer handle, buffer has been invalited.
     */
    std::tuple<u8*, u64, vk::Buffer, bool> Reserve(u64 size, bool keep_in_host);

    [[nodiscard]] VKExecutionContext Send(VKExecutionContext exctx, u64 size);

private:
    void CreateBuffers(VKMemoryManager& memory_manager, vk::BufferUsageFlags usage);

    void GrowResources(std::size_t grow_size);

    VKResourceManager& resource_manager;
    VKMemoryManager& memory_manager;
    VKScheduler& sched;
    const VKDevice& device;
    const u64 buffer_size;
    const bool has_device_memory;
    const vk::AccessFlags access;
    const vk::PipelineStageFlags pipeline_stage;

    VKMemoryCommit mappable_commit;
    VKMemoryCommit device_commit;

    UniqueBuffer mappable_buffer;
    UniqueBuffer device_buffer;

    u8* mapped_ptr{};
    u64 buffer_pos{};
    u64 mapped_size{};
    bool use_device{};

    std::vector<std::unique_ptr<StreamBufferResource>> resources;
    u32 used_resources{};
    bool mark_invalidation{};
};

} // namespace Vulkan
