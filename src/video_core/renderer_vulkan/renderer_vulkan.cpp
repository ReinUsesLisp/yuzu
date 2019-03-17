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

enum class IgnoreMode { Head, Tail };

struct IgnoreEntry {
    IgnoreMode mode;
    std::size_t length;
    const char* message;
};

static constexpr IgnoreEntry BuildIgnore(IgnoreMode mode, const char* message) {
    // constexpr std::strlen reimplementation
    std::size_t length = 0;
    while (message[length] != 0) {
        ++length;
    }
    return {mode, length, message};
}

static constexpr std::array<IgnoreEntry, 2> ignored_messages = {
    BuildIgnore(IgnoreMode::Head, "Device Extension VK_EXT_scalar_block_layout "),
    BuildIgnore(IgnoreMode::Tail, " is an array with stride 4 not satisfying alignment to 16")};

static bool IsMessageIgnored(std::string message) {
    const auto divisor = message.find('|');
    const auto end_of_line = message.find('\n');
    for (const auto& ignore : ignored_messages) {
        switch (ignore.mode) {
        case IgnoreMode::Head:
            if (divisor != std::string::npos &&
                message.compare(divisor + 2, ignore.length, ignore.message) == 0) {
                return true;
            }
            break;
        case IgnoreMode::Tail:
            if (end_of_line != std::string::npos && end_of_line >= ignore.length &&
                message.compare(end_of_line - ignore.length, ignore.length, ignore.message) == 0) {
                return true;
            }
            break;
        }
    }
    return false;
}

static VkBool32 DebugCallback(VkDebugReportFlagsEXT flags_, VkDebugReportObjectTypeEXT object_type,
                              u64 object, std::size_t location, s32 message_code,
                              const char* layer_prefix, const char* message, void* user_data) {
    const vk::DebugReportFlagsEXT flags{flags_};
    if (flags & vk::DebugReportFlagBitsEXT::eError && !IsMessageIgnored(message)) {
        LOG_ERROR(Render_Vulkan, "{}", message);
        UNREACHABLE();
    } else if (flags & (vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning |
                        vk::DebugReportFlagBitsEXT::ePerformanceWarning)) {
        LOG_WARNING(Render_Vulkan, "{}", message);
    } else if (flags & vk::DebugReportFlagBitsEXT::eDebug) {
        LOG_DEBUG(Render_Vulkan, "{}", message);
    } else if (flags & vk::DebugReportFlagBitsEXT::eInformation) {
        LOG_TRACE(Render_Vulkan, "{}", message);
    }
    return VK_FALSE;
}

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

        scheduler->Flush(false, render_semaphore);

        if (swapchain->Present(render_semaphore, fence)) {
            blit_screen->Recreate();
        }

        render_window.SwapBuffers();
    }

    render_window.PollEvents();

    system.FrameLimiter().DoFrameLimiting(system.CoreTiming().GetGlobalTimeUs());
    system.GetPerfStats().BeginSystemFrame();
}

bool RendererVulkan::Init() {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr{};
    render_window.RetrieveVulkanHandlers(reinterpret_cast<void**>(&vkGetInstanceProcAddr),
                                         reinterpret_cast<void**>(&instance),
                                         reinterpret_cast<void**>(&surface));
    const vk::DispatchLoaderDynamic dldi(instance, vkGetInstanceProcAddr);

    std::optional<vk::DebugReportCallbackEXT> debug_callback_opt = std::nullopt;
    if (Settings::values.renderer_debug && dldi.vkCreateDebugReportCallbackEXT) {
        debug_callback_opt = CreateDebugCallback(dldi);
        if (!debug_callback_opt) {
            return false;
        }
    }

    if (!PickDevices(dldi)) {
        if (debug_callback_opt) {
            instance.destroy(*debug_callback_opt, nullptr, dldi);
        }
        return false;
    }
    debug_callback = UniqueDebugReportCallbackEXT(
        *debug_callback_opt, vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderDynamic>(
                                 instance, nullptr, device->GetDispatchLoader()));

    memory_manager = std::make_unique<VKMemoryManager>(*device);

    resource_manager = std::make_unique<VKResourceManager>(*device);

    const auto& framebuffer = render_window.GetFramebufferLayout();
    swapchain = std::make_unique<VKSwapchain>(surface, *device);
    swapchain->Create(framebuffer.width, framebuffer.height);

    scheduler = std::make_unique<VKScheduler>(*device, *resource_manager);

    rasterizer = std::make_unique<RasterizerVulkan>(system, render_window, screen_info, *device,
                                                    *resource_manager, *memory_manager, *scheduler);

    blit_screen =
        std::make_unique<VKBlitScreen>(render_window, *rasterizer, *device, *resource_manager,
                                       *memory_manager, *swapchain, *scheduler, screen_info);

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
    scheduler.reset();
    swapchain.reset();
    memory_manager.reset();
    resource_manager.reset();
    device.reset();
}

std::optional<vk::DebugReportCallbackEXT> RendererVulkan::CreateDebugCallback(
    const vk::DispatchLoaderDynamic& dldi) {
    const vk::DebugReportCallbackCreateInfoEXT callback_ci(
        vk::DebugReportFlagBitsEXT::eInformation | vk::DebugReportFlagBitsEXT::eWarning |
            vk::DebugReportFlagBitsEXT::ePerformanceWarning | vk::DebugReportFlagBitsEXT::eError |
            vk::DebugReportFlagBitsEXT::eDebug,
        reinterpret_cast<PFN_vkDebugReportCallbackEXT>(DebugCallback));
    vk::DebugReportCallbackEXT debug_callback;
    if (instance.createDebugReportCallbackEXT(&callback_ci, nullptr, &debug_callback, dldi) !=
        vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Failed to create debug callback");
        return {};
    }
    return debug_callback;
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