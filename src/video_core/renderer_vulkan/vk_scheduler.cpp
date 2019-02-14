// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <vector>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

VKScheduler::VKScheduler(VKResourceManager& resource_manager, const VKDevice& device)
    : resource_manager{resource_manager}, device{device} {
    next_fence = &resource_manager.CommitFence();
    AllocateNewContext();
}

VKScheduler::~VKScheduler() = default;

VKExecutionContext VKScheduler::GetExecutionContext() const {
    return VKExecutionContext(current_fence, current_cmdbuf);
}

VKExecutionContext VKScheduler::Flush(vk::Semaphore semaphore) {
    SubmitExecution(semaphore);
    current_fence->Release();
    AllocateNewContext();
    return GetExecutionContext();
}

VKExecutionContext VKScheduler::Finish(vk::Semaphore semaphore) {
    SubmitExecution(semaphore);
    current_fence->Wait();
    current_fence->Release();
    AllocateNewContext();
    return GetExecutionContext();
}

void VKScheduler::SubmitExecution(vk::Semaphore semaphore) {
    const auto& dld = device.GetDispatchLoader();
    current_cmdbuf.end(dld);

    const auto queue = device.GetGraphicsQueue();
    const vk::SubmitInfo submit_info(0, nullptr, nullptr, 1, &current_cmdbuf, semaphore ? 1u : 0u,
                                     &semaphore);
    queue.submit({submit_info}, *current_fence, dld);
}

void VKScheduler::AllocateNewContext() {
    current_fence = next_fence;
    current_cmdbuf = resource_manager.CommitCommandBuffer(*current_fence);

    // Allocate the semaphore the current call will use in a future fence, to avoid it being
    // freed while it's still being waited on.
    next_fence = &resource_manager.CommitFence();

    const auto& dld = device.GetDispatchLoader();
    current_cmdbuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit}, dld);
}

} // namespace Vulkan
