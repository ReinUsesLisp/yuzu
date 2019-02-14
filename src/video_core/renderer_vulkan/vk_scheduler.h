// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <utility>
#include <vector>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/declarations.h"

namespace Vulkan {

class VKDevice;
class VKExecutionContext;
class VKFence;
class VKResourceManager;

class VKScheduler {
public:
    explicit VKScheduler(VKResourceManager& resource_manager, const VKDevice& device);
    ~VKScheduler();

    [[nodiscard]] VKExecutionContext GetExecutionContext() const;

    VKExecutionContext Flush(vk::Semaphore semaphore = nullptr);

    VKExecutionContext Finish(vk::Semaphore semaphore = nullptr);

private:
    void SubmitExecution(vk::Semaphore semaphore);

    void AllocateNewContext();

    VKResourceManager& resource_manager;
    const VKDevice& device;

    vk::CommandBuffer current_cmdbuf;
    VKFence* current_fence = nullptr;

    VKFence* next_fence = nullptr;
};

class VKExecutionContext {
    friend class VKScheduler;

public:
    VKExecutionContext() = default;

    VKFence& GetFence() const {
        return *fence;
    }

    vk::CommandBuffer GetCommandBuffer() const {
        return cmdbuf;
    }

private:
    explicit VKExecutionContext(VKFence* fence, vk::CommandBuffer cmdbuf)
        : fence{fence}, cmdbuf{cmdbuf} {}

    VKFence* fence;
    vk::CommandBuffer cmdbuf;
};

} // namespace Vulkan