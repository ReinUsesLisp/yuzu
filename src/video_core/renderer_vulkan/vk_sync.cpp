// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/threadsafe_queue.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_sync.h"

namespace Vulkan {

VulkanScheduler::VulkanScheduler(VulkanResourceManager& resource_manager, const VulkanDevice& device_handler)
    : resource_manager(resource_manager), device(device_handler.GetLogical()),
      queue(device_handler.GetGraphicsQueue()) {

    next_fence = &resource_manager.CommitFence();

    worker_thread = std::thread([&]() {
        while (executing) {
            std::unique_lock work_lock(work_mutex);
            work_cv.wait(work_lock);

            std::unique_ptr<Call> pass;
            while (scheduled_passes.Pop(pass)) {
                std::unique_lock fence_lock = pass->fence->Acquire();
                queue.submit(static_cast<u32>(pass->submit_infos.size()), pass->submit_infos.data(),
                             *pass->fence);
                if (pass->take_fence_ownership) {
                    pass->fence->Release();
                }

                if (scheduled_passes.Empty()) {
                    work_done = true;
                }
                flush_signal.notify_one();
            }
        }
    });
}

VulkanScheduler::~VulkanScheduler() {
    executing = false;
    worker_thread.join();
}

VulkanFence& VulkanScheduler::BeginPass(bool take_fence_ownership) {
    recording_submit = true;

    VulkanFence& now_fence = *next_fence;
    pass = std::make_unique<Call>();
    pass->fence = &now_fence;
    pass->take_fence_ownership = take_fence_ownership;

    // Allocate the semaphore the current call will use in a future fence, to avoid it being freed
    // while it's still being waited on.
    next_fence = &resource_manager.CommitFence();
    pass->semaphore = resource_manager.CommitSemaphore(*next_fence);

    return now_fence;
}

void VulkanScheduler::AddDependency(vk::CommandBuffer cmdbuf, vk::Semaphore semaphore,
                               vk::PipelineStageFlags pipeline_stage) {
    ASSERT(recording_submit);

    const std::size_t index = pass->cmdbufs.size();
    pass->cmdbufs.push_back(cmdbuf);
    pass->signal_semaphores.push_back(semaphore);
    pass->wait_semaphores.push_back(semaphore);
    pass->pipeline_stages.push_back(pipeline_stage);

    const vk::SubmitInfo si({}, nullptr, nullptr, 1, &pass->cmdbufs[index], 1,
                            &pass->signal_semaphores[index]);
    pass->submit_infos.push_back(si);
}

void VulkanScheduler::EndPass() {
    ASSERT(recording_submit);

    if (previous_semaphore) {
        // TODO(Rodrigo): Pipeline wait can be optimized with an extra argument.
        pass->wait_semaphores.push_back(previous_semaphore);
        pass->pipeline_stages.push_back(vk::PipelineStageFlagBits::eAllCommands);
    }
    ASSERT_MSG(pass->wait_semaphores.size() == pass->pipeline_stages.size(),
               "Dependency size mismatch");

    previous_semaphore = pass->semaphore;
    pass->submit_infos.push_back({static_cast<u32>(pass->wait_semaphores.size()),
                                  pass->wait_semaphores.data(), pass->pipeline_stages.data(),
                                  static_cast<u32>(pass->commands.size()), pass->commands.data(), 1,
                                  &pass->semaphore});

    work_done = false;
    scheduled_passes.Push(std::move(pass));
    work_cv.notify_one();

    recording_submit = false;
}

void VulkanScheduler::Flush() {
    if (work_done)
        return;
    std::unique_lock flush_lock(work_mutex);
    work_cv.notify_one();
    flush_signal.wait(flush_lock, [&]() -> bool { return work_done; });
}

vk::CommandBuffer VulkanScheduler::BeginRecord() {
    ASSERT(recording_submit);

    const vk::CommandBuffer cmdbuf = resource_manager.CommitCommandBuffer(*pass->fence);
    cmdbuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return cmdbuf;
}

void VulkanScheduler::EndRecord(vk::CommandBuffer cmdbuf) {
    ASSERT(recording_submit);

    cmdbuf.end();
    pass->commands.push_back(cmdbuf);
}

vk::Semaphore VulkanScheduler::QuerySemaphore() {
    vk::Semaphore semaphore = previous_semaphore;
    previous_semaphore = nullptr;
    return semaphore;
}

} // namespace Vulkan
