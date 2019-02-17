// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include "common/assert.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"

namespace Vulkan {

constexpr u64 RESOURCE_RESERVE = 0x4000;
constexpr u64 RESOURCE_CHUNK = 0x1000;

class StreamBufferResource final : public VKResource {
public:
    explicit StreamBufferResource() = default;
    virtual ~StreamBufferResource() {
        if (fence) {
            fence->Unprotect(this);
        }
    }

    void Setup(VKFence& new_fence) {
        if (fence) {
            fence->Unprotect(this);
        }

        fence = &new_fence;
        fence->Protect(this);
    }

    void Wait() {
        if (fence) {
            fence->Wait();
        }
    }

protected:
    virtual void OnFenceRemoval(VKFence* signaling_fence) {
        ASSERT(signaling_fence == fence);
        fence = nullptr;
    }

private:
    VKFence* fence{};
    bool is_signaled{};
};

VKStreamBuffer::VKStreamBuffer(VKResourceManager& resource_manager, const VKDevice& device,
                               VKMemoryManager& memory_manager, VKScheduler& sched, u64 size,
                               vk::BufferUsageFlags usage, vk::AccessFlags access,
                               vk::PipelineStageFlags pipeline_stage)
    : resource_manager{resource_manager}, device{device}, memory_manager{memory_manager},
      sched{sched}, has_device_memory{!memory_manager.IsMemoryUnified()},
      buffer_size{size}, access{access}, pipeline_stage{pipeline_stage} {

    CreateBuffers(memory_manager, usage);
    GrowResources(RESOURCE_RESERVE);
}

VKStreamBuffer::~VKStreamBuffer() = default;

std::tuple<u8*, u64, vk::Buffer, bool> VKStreamBuffer::Reserve(u64 size, bool keep_in_host) {
    ASSERT(size <= buffer_size);
    mapped_size = size;

    mark_invalidation = false;
    if (buffer_pos + size > buffer_size) {
        used_resources = 0;
        buffer_pos = 0;
        mark_invalidation = true;
    }

    use_device = has_device_memory && !keep_in_host;
    return {mapped_ptr + buffer_pos, buffer_pos, use_device ? *device_buffer : *mappable_buffer,
            mark_invalidation};
}

#pragma optimize("", off)

VKExecutionContext VKStreamBuffer::Send(VKExecutionContext exctx, u64 size) {
    ASSERT(size <= mapped_size);

    if (mark_invalidation) {
        // TODO(Rodrigo): Find a better way to invalidate than waiting for all resources to finish.
        exctx = sched.Flush();
        std::for_each(resources.begin(), resources.begin() + used_resources,
                      [&](const auto& resource) { resource->Wait(); });
        mark_invalidation = false;
    }

    if (use_device) {
        const auto& dld = device.GetDispatchLoader();
        const u32 graphics_family = device.GetGraphicsFamily();
        const auto cmdbuf = exctx.GetCommandBuffer();

        // Buffers are mirrored.
        const vk::BufferCopy copy_region(buffer_pos, buffer_pos, size);
        cmdbuf.copyBuffer(*mappable_buffer, *device_buffer, {copy_region}, dld);

        const vk::BufferMemoryBarrier barrier(vk::AccessFlagBits::eTransferWrite, access,
                                              graphics_family, graphics_family, *device_buffer,
                                              buffer_pos, size);
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, pipeline_stage, {}, {},
                               {barrier}, {}, dld);
    }

    if (used_resources + 1 >= resources.size()) {
        GrowResources(RESOURCE_CHUNK);
    }
    auto& resource = resources[used_resources++];
    resource->Setup(exctx.GetFence());

    buffer_pos += size;

    return exctx;
}

void VKStreamBuffer::CreateBuffers(VKMemoryManager& memory_manager, vk::BufferUsageFlags usage) {
    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();

    vk::BufferUsageFlags mappable_usage = usage;
    if (has_device_memory) {
        mappable_usage |= vk::BufferUsageFlagBits::eTransferSrc;
    }
    const vk::BufferCreateInfo buffer_ci({}, buffer_size, mappable_usage,
                                         vk::SharingMode::eExclusive, 0, nullptr);

    mappable_buffer = dev.createBufferUnique(buffer_ci, nullptr, dld);
    mappable_commit = memory_manager.Commit(*mappable_buffer, true);
    mapped_ptr = mappable_commit->GetData();

    if (has_device_memory) {
        const vk::BufferCreateInfo buffer_ci({}, buffer_size,
                                             usage | vk::BufferUsageFlagBits::eTransferDst,
                                             vk::SharingMode::eExclusive, 0, nullptr);
        device_buffer = dev.createBufferUnique(buffer_ci, nullptr, dld);

        const vk::MemoryRequirements reqs = dev.getBufferMemoryRequirements(*device_buffer, dld);
        device_commit = memory_manager.Commit(*device_buffer, false);
    }
}

void VKStreamBuffer::GrowResources(std::size_t grow_size) {
    const std::size_t previous_size = resources.size();
    resources.resize(previous_size + grow_size);
    std::generate(resources.begin() + previous_size, resources.end(),
                  [&]() { return std::make_unique<StreamBufferResource>(); });
}

} // namespace Vulkan