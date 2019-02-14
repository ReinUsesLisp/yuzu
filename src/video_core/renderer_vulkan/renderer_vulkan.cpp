// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <set>
#include <vector>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/frontend/emu_window.h"
#include "core/memory.h"
#include "core/perf_stats.h"
#include "core/settings.h"
#include "video_core/gpu.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/utils.h"

namespace Vulkan {

RendererVulkan::RendererVulkan(Core::Frontend::EmuWindow& window, Core::System& system)
    : RendererBase(window), system{system} {}

RendererVulkan::~RendererVulkan() {
    ShutDown();
}

void RendererVulkan::SwapBuffers(
    std::optional<std::reference_wrapper<const Tegra::FramebufferConfig>> framebuffer) {

    system.GetPerfStats().EndSystemFrame();

    const auto& layout = render_window.GetFramebufferLayout();
    if (framebuffer && layout.width > 0 && layout.height > 0 && render_window.IsShown()) {
        if (swapchain->HasFramebufferChanged(layout)) {
            swapchain->Create(layout.width, layout.height);
            blit_screen->Recreate();
        }

        swapchain->AcquireNextImage();
        const auto [fence, render_semaphore] = blit_screen->Draw(*framebuffer);

        sched->Flush();

        swapchain->Present(render_semaphore, fence);

        render_window.SwapBuffers();
    }

    render_window.PollEvents();

    system.FrameLimiter().DoFrameLimiting(Core::Timing::GetGlobalTimeUs());
    system.GetPerfStats().BeginSystemFrame();
}

bool RendererVulkan::Init() {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr{};
    render_window.RetrieveVulkanHandlers(reinterpret_cast<void**>(&vkGetInstanceProcAddr),
                                         reinterpret_cast<void**>(&instance),
                                         reinterpret_cast<void**>(&surface));
    const vk::DispatchLoaderDynamic dldi(instance, vkGetInstanceProcAddr);

    if (!PickDevices(dldi)) {
        return false;
    }

    memory_manager = std::make_unique<VKMemoryManager>(*device);

    resource_manager = std::make_unique<VKResourceManager>(*device);

    const auto& framebuffer = render_window.GetFramebufferLayout();
    swapchain = std::make_unique<VKSwapchain>(surface, *device);
    swapchain->Create(framebuffer.width, framebuffer.height);

    sched = std::make_unique<VKScheduler>(*resource_manager, *device);

    rasterizer = std::make_unique<RasterizerVulkan>(system, render_window, screen_info, *device,
                                                    *resource_manager, *memory_manager, *sched);

    blit_screen =
        std::make_unique<VKBlitScreen>(render_window, *rasterizer, *device, *resource_manager,
                                       *memory_manager, *swapchain, *sched, screen_info);

    return true;
}

void RendererVulkan::ShutDown() {
    if (!device) {
        return;
    }
    const auto dev = device->GetLogical();
    const auto& dld = device->GetDispatchLoader();
    dev.waitIdle(dld);

    rasterizer.reset();
    blit_screen.reset();
    sched.reset();
    swapchain.reset();
    memory_manager.reset();
    resource_manager.reset();
    device.reset();
}

bool RendererVulkan::PickDevices(const vk::DispatchLoaderDynamic& dldi) {
    const auto devices = instance.enumeratePhysicalDevices(dldi);

    // TODO(Rodrigo): Choose device from config file
    const s32 device_index = Settings::values.vulkan_device;
    if (device_index < 0 || device_index >= static_cast<s32>(devices.size())) {
        LOG_ERROR(Render_Vulkan, "Invalid device index {}!", device_index);
        return false;
    }
    const vk::PhysicalDevice physical_device = devices[device_index];

    if (!VKDevice::IsSuitable(dldi, physical_device, surface)) {
        LOG_ERROR(Render_Vulkan, "Device is not suitable!");
        return false;
    }

    device = std::make_unique<VKDevice>(dldi, physical_device, surface);
    return device->Create(dldi, instance);
}

} // namespace Vulkan