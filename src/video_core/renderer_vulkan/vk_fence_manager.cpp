// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <thread>

#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_fence_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"

namespace Vulkan {

InnerFence::InnerFence(const VKDevice& device, VKScheduler& scheduler, u32 payload, bool is_stubbed)
    : VideoCommon::FenceBase(payload, is_stubbed), device{device}, scheduler{scheduler} {}

InnerFence::InnerFence(const VKDevice& device, VKScheduler& scheduler, GPUVAddr address,
                       u32 payload, bool is_stubbed)
    : VideoCommon::FenceBase(address, payload, is_stubbed), device{device}, scheduler{scheduler} {}

InnerFence::~InnerFence() = default;

void InnerFence::Queue() {
    if (is_stubbed) {
        return;
    }
    ASSERT(!event);

    const auto& dev = device.GetLogical();
    event = dev.createEventUnique({}, nullptr, device.GetDispatchLoader());
    ticks = scheduler.Ticks();

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([event = *event](vk::CommandBuffer cmdbuf, auto& dld) {
        cmdbuf.setEvent(event, vk::PipelineStageFlagBits::eAllCommands, dld);
    });
}

bool InnerFence::IsSignaled() const {
    if (is_stubbed) {
        return true;
    }
    ASSERT(event);
    return IsEventSignalled();
}

void InnerFence::Wait() {
    if (is_stubbed) {
        return;
    }
    ASSERT(event);

    if (ticks >= scheduler.Ticks()) {
        scheduler.Flush();
    }
    while (!IsEventSignalled()) {
        std::this_thread::yield();
    }
}

bool InnerFence::IsEventSignalled() const {
    const auto& dev = device.GetLogical();
    return dev.getEventStatus(*event, device.GetDispatchLoader()) == vk::Result::eEventSet;
}

VKFenceManager::VKFenceManager(Core::System& system, VideoCore::RasterizerInterface& rasterizer,
                               const VKDevice& device, VKScheduler& scheduler,
                               VKTextureCache& texture_cache, VKBufferCache& buffer_cache)
    : GenericFenceManager(system, rasterizer, texture_cache, buffer_cache), device{device},
      scheduler{scheduler} {}

Fence VKFenceManager::CreateFence(u32 value, bool is_stubbed) {
    return std::make_shared<InnerFence>(device, scheduler, value, is_stubbed);
}

Fence VKFenceManager::CreateFence(GPUVAddr addr, u32 value, bool is_stubbed) {
    return std::make_shared<InnerFence>(device, scheduler, addr, value, is_stubbed);
}

void VKFenceManager::QueueFence(Fence& fence) {
    fence->Queue();
}

bool VKFenceManager::IsFenceSignaled(Fence& fence) {
    return fence->IsSignaled();
}

void VKFenceManager::WaitFence(Fence& fence) {
    fence->Wait();
}

} // namespace Vulkan
