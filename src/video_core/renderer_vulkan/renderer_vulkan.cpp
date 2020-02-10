// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#ifdef __linux__
#define VK_USE_PLATFORM_XLIB_KHR
#endif

#include "video_core/renderer_vulkan/declarations.h"

#ifdef __linux__
#undef Always
#undef None
#undef Status
#undef Success
#endif

#include <memory>
#include <optional>
#include <vector>

#include <fmt/format.h>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/telemetry.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/frontend/emu_window.h"
#include "core/memory.h"
#include "core/perf_stats.h"
#include "core/settings.h"
#include "core/telemetry_session.h"
#include "video_core/gpu.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

namespace Vulkan {

namespace {

VkBool32 DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity_,
                       VkDebugUtilsMessageTypeFlagsEXT type,
                       const VkDebugUtilsMessengerCallbackDataEXT* data,
                       [[maybe_unused]] void* user_data) {
    const vk::DebugUtilsMessageSeverityFlagBitsEXT severity{severity_};
    const char* message{data->pMessage};

    if (severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
        LOG_CRITICAL(Render_Vulkan, "{}", message);
    } else if (severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        LOG_WARNING(Render_Vulkan, "{}", message);
    } else if (severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo) {
        LOG_INFO(Render_Vulkan, "{}", message);
    } else if (severity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose) {
        LOG_DEBUG(Render_Vulkan, "{}", message);
    }
    return VK_FALSE;
}

Common::DynamicLibrary OpenVulkanLibrary() {
    Common::DynamicLibrary dl;
#ifdef __APPLE__
    // Check if a path to a specific Vulkan library has been specified.
    char* libvulkan_env = getenv("LIBVULKAN_PATH");
    if (!libvulkan_env || !dl.Open(libvulkan_env)) {
        // Use the libvulkan.dylib from the application bundle.
        std::string filename = File::GetBundleDirectory() + "/Contents/Frameworks/libvulkan.dylib";
        dl.Open(filename.c_str());
    }
#else
    std::string filename = Common::DynamicLibrary::GetVersionedFilename("vulkan", 1);
    if (!dl.Open(filename.c_str())) {
        // Android devices may not have libvulkan.so.1, only libvulkan.so.
        filename = Common::DynamicLibrary::GetVersionedFilename("vulkan");
        dl.Open(filename.c_str());
    }
#endif
    return dl;
}

std::string GetReadableVersion(u32 version) {
    return fmt::format("{}.{}.{}", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
                       VK_VERSION_PATCH(version));
}

std::string GetDriverVersion(const VKDevice& device) {
    // Extracted from
    // https://github.com/SaschaWillems/vulkan.gpuinfo.org/blob/5dddea46ea1120b0df14eef8f15ff8e318e35462/functions.php#L308-L314
    const u32 version = device.GetDriverVersion();

    if (device.GetDriverID() == vk::DriverIdKHR::eNvidiaProprietary) {
        const u32 major = (version >> 22) & 0x3ff;
        const u32 minor = (version >> 14) & 0x0ff;
        const u32 secondary = (version >> 6) & 0x0ff;
        const u32 tertiary = version & 0x003f;
        return fmt::format("{}.{}.{}.{}", major, minor, secondary, tertiary);
    }
    if (device.GetDriverID() == vk::DriverIdKHR::eIntelProprietaryWindows) {
        const u32 major = version >> 14;
        const u32 minor = version & 0x3fff;
        return fmt::format("{}.{}", major, minor);
    }

    return GetReadableVersion(version);
}

std::string BuildCommaSeparatedExtensions(std::vector<std::string> available_extensions) {
    std::sort(std::begin(available_extensions), std::end(available_extensions));

    static constexpr std::size_t AverageExtensionSize = 64;
    std::string separated_extensions;
    separated_extensions.reserve(available_extensions.size() * AverageExtensionSize);

    const auto end = std::end(available_extensions);
    for (auto extension = std::begin(available_extensions); extension != end; ++extension) {
        if (const bool is_last = extension + 1 == end; is_last) {
            separated_extensions += *extension;
        } else {
            separated_extensions += fmt::format("{},", *extension);
        }
    }
    return separated_extensions;
}

std::vector<const char*> SelectInstanceExtensions(const vk::DispatchLoaderDynamic& dld,
                                                  const Core::Frontend::WindowSystemType& wstype,
                                                  bool enable_debug_report) {
    const auto available_extensions = vk::enumerateInstanceExtensionProperties(nullptr, dld);
    std::vector<const char*> extensions;
    extensions.reserve(6);
    const auto add_extension = [&](const char* name, bool required) {
        const auto it = std::find_if(available_extensions.begin(), available_extensions.end(),
                                     [name](const auto& properties) {
                                         return !std::strcmp(name, properties.extensionName);
                                     });
        if (it != available_extensions.end()) {
            extensions.push_back(name);
            return true;
        }
        if (required) {
            LOG_ERROR(Render_Vulkan, "Missing required extension {}", name);
        }
        return false;
    };

#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (wstype == Core::Frontend::WindowSystemType::Windows &&
        !add_extension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME, true)) {
        return {};
    }
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
    if (wstype == Core::Frontend::WindowSystemType::X11 &&
        !add_extension(VK_KHR_XLIB_SURFACE_EXTENSION_NAME, true)) {
        return {};
    }
#endif

    if (!add_extension(VK_KHR_SURFACE_EXTENSION_NAME, true)) {
        return {};
    }

    // VK_EXT_debug_report
    if (enable_debug_report && !add_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, false)) {
        LOG_WARNING(Render_Vulkan, "Debug report requested, but the extension is not available");
    }

    add_extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, false);
    add_extension(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME, false);
    return extensions;
}

UniqueInstance CreateVulkanInstance(const vk::DispatchLoaderDynamic& dld,
                                    Core::Frontend::WindowSystemType wstype, bool enable_debug) {
    const std::vector enabled_extensions = SelectInstanceExtensions(dld, wstype, enable_debug);
    if (enabled_extensions.empty()) {
        return {};
    }

    if (vk::enumerateInstanceVersion(dld) < VK_MAKE_VERSION(1, 1, 0)) {
        LOG_ERROR(Render_Vulkan, "Vulkan 1.1 is not susupported! Try updating your drivers");
        return {};
    }

    vk::ApplicationInfo app_info;
    app_info.pApplicationName = "yuzu Emulator";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "yuzu Emulator";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_MAKE_VERSION(1, 1, 0);

    vk::InstanceCreateInfo instance_create_info;
    instance_create_info.pApplicationInfo = &app_info;
    instance_create_info.enabledExtensionCount = static_cast<u32>(enabled_extensions.size());
    instance_create_info.ppEnabledExtensionNames = enabled_extensions.data();

    // Enable debug layers when requested
    if (enable_debug) {
        static constexpr std::array layer_names = {"VK_LAYER_LUNARG_standard_validation"};
        instance_create_info.enabledLayerCount = static_cast<u32>(layer_names.size());
        instance_create_info.ppEnabledLayerNames = layer_names.data();
    }

    vk::Instance instance;
    const vk::Result res = vk::createInstance(&instance_create_info, nullptr, &instance, dld);
    if (res != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "vkCreateInstance failed: {}", vk::to_string(res));
        return {};
    }

    return UniqueInstance(instance,
                          vk::ObjectDestroy<vk::NoParent, vk::DispatchLoaderDynamic>(nullptr, dld));
}

UniqueSurfaceKHR CreateVulkanSurface(
    vk::Instance instance, const vk::DispatchLoaderDynamic& dld,
    const Core::Frontend::EmuWindow::WindowSystemInfo& window_info) {
    vk::SurfaceKHR surface;
    vk::Result result;
    bool tried = false;

#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (window_info.type == Core::Frontend::WindowSystemType::Windows) {
        const HWND hWnd = static_cast<HWND>(window_info.render_surface);
        const vk::Win32SurfaceCreateInfoKHR win32_ci({}, nullptr, hWnd);
        result = instance.createWin32SurfaceKHR(&win32_ci, nullptr, &surface, dld);
        tried = true;
    }
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
    if (window_info.type == Core::Frontend::WindowSystemType::X11) {
        const auto display = static_cast<Display*>(window_info.display_connection);
        const auto window = reinterpret_cast<Window>(window_info.render_surface);
        const vk::XlibSurfaceCreateInfoKHR xlib_ci({}, display, window);
        result = instance.createXlibSurfaceKHR(&xlib_ci, nullptr, &surface, dld);
        tried = true;
    }
#endif

    if (!tried) {
        LOG_ERROR(Render_Vulkan, "Surface creation is not supported in this platform");
        return {};
    }

    if (result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Failed to create Vulkan surface: {}", vk::to_string(result));
        return {};
    }
    return UniqueSurfaceKHR(surface, vk::ObjectDestroy(instance, nullptr, dld));
}

} // Anonymous namespace

RendererVulkan::RendererVulkan(Core::Frontend::EmuWindow& window, Core::System& system)
    : RendererBase(window), system{system} {}

RendererVulkan::~RendererVulkan() {
    ShutDown();
}

void RendererVulkan::SwapBuffers(const Tegra::FramebufferConfig* framebuffer) {
    render_window.PollEvents();

    if (!framebuffer) {
        return;
    }

    const auto& layout = render_window.GetFramebufferLayout();
    if (layout.width > 0 && layout.height > 0 && render_window.IsShown()) {
        const VAddr framebuffer_addr = framebuffer->address + framebuffer->offset;
        const bool use_accelerated =
            rasterizer->AccelerateDisplay(*framebuffer, framebuffer_addr, framebuffer->stride);
        const bool is_srgb = use_accelerated && screen_info.is_srgb;
        if (swapchain->HasFramebufferChanged(layout) || swapchain->GetSrgbState() != is_srgb) {
            swapchain->Create(layout.width, layout.height, is_srgb);
            blit_screen->Recreate();
        }

        scheduler->WaitWorker();

        swapchain->AcquireNextImage();
        const auto [fence, render_semaphore] = blit_screen->Draw(*framebuffer, use_accelerated);

        scheduler->Flush(false, render_semaphore);

        if (swapchain->Present(render_semaphore, fence)) {
            blit_screen->Recreate();
        }

        rasterizer->TickFrame();
    }

    render_window.PollEvents();
}

void RendererVulkan::TryPresent(int /*timeout_ms*/){
    // TODO (bunnei): ImplementMe
}

std::optional<Core::Frontend::BackendInfo> RendererVulkan::MakeBackendInfo() {
    Core::Frontend::BackendInfo info;
    info.name = "Vulkan";
    info.api_type = Core::Frontend::APIType::Vulkan;
    info.dl = OpenVulkanLibrary();
    if (!info.dl.IsOpen()) {
        return {};
    }

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    if (!info.dl.GetSymbol("vkGetInstanceProcAddr", &vkGetInstanceProcAddr)) {
        return {};
    }

    vk::DispatchLoaderDynamic dld(vkGetInstanceProcAddr);
    const auto instance =
        CreateVulkanInstance(dld, Core::Frontend::WindowSystemType::Uninitialized, false);
    if (!instance) {
        return {};
    }
    dld.init(*instance);

    for (const auto physical_device : instance->enumeratePhysicalDevices(dld)) {
        info.adapters.push_back(physical_device.getProperties(dld).deviceName);
    }
    return info;
}

bool RendererVulkan::Init() {
    vulkan_library = OpenVulkanLibrary();
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    if (!vulkan_library.GetSymbol("vkGetInstanceProcAddr", &vkGetInstanceProcAddr)) {
        return false;
    }
    dld.init(vkGetInstanceProcAddr);

    instance = CreateVulkanInstance(dld, render_window.GetWindowInfo().type,
                                    Settings::values.renderer_debug);
    if (!instance) {
        return false;
    }
    dld.init(*instance);

    std::optional<vk::DebugUtilsMessengerEXT> callback;
    if (Settings::values.renderer_debug && dld.vkCreateDebugUtilsMessengerEXT) {
        callback = CreateDebugCallback();
        if (!callback) {
            return false;
        }
    }
    debug_callback =
        UniqueDebugUtilsMessengerEXT(*callback, vk::ObjectDestroy(*instance, nullptr, dld));

    surface = CreateVulkanSurface(*instance, dld, render_window.GetWindowInfo());
    if (!surface) {
        return false;
    }

    if (!PickDevices()) {
        return false;
    }

    Report();

    memory_manager = std::make_unique<VKMemoryManager>(*device);

    resource_manager = std::make_unique<VKResourceManager>(*device);

    const auto& framebuffer = render_window.GetFramebufferLayout();
    swapchain = std::make_unique<VKSwapchain>(*surface, *device);
    swapchain->Create(framebuffer.width, framebuffer.height, false);

    scheduler = std::make_unique<VKScheduler>(*device, *resource_manager);

    rasterizer = std::make_unique<RasterizerVulkan>(system, render_window, screen_info, *device,
                                                    *resource_manager, *memory_manager, *scheduler);

    blit_screen = std::make_unique<VKBlitScreen>(system, render_window, *rasterizer, *device,
                                                 *resource_manager, *memory_manager, *swapchain,
                                                 *scheduler, screen_info);

    return true;
}

void RendererVulkan::ShutDown() {
    if (device) {
        const auto dev = device->GetLogical();
        const auto& dld = device->GetDispatchLoader();
        if (dev && dld.vkDeviceWaitIdle) {
            dev.waitIdle(dld);
        }
    }

    rasterizer.reset();
    blit_screen.reset();
    scheduler.reset();
    swapchain.reset();
    memory_manager.reset();
    resource_manager.reset();
    device.reset();
    surface.reset();
    debug_callback.reset();
    instance.reset();
}

std::optional<vk::DebugUtilsMessengerEXT> RendererVulkan::CreateDebugCallback() {
    const vk::DebugUtilsMessengerCreateInfoEXT callback_ci(
        {},
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        &DebugCallback, nullptr);
    vk::DebugUtilsMessengerEXT callback;
    if (instance->createDebugUtilsMessengerEXT(&callback_ci, nullptr, &callback, dld) !=
        vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Failed to create debug callback");
        return {};
    }
    return callback;
}

bool RendererVulkan::PickDevices() {
    const auto devices = instance->enumeratePhysicalDevices(dld);

    // TODO(Rodrigo): Choose device from config file
    const s32 device_index = Settings::values.vulkan_device;
    if (device_index < 0 || device_index >= static_cast<s32>(devices.size())) {
        LOG_ERROR(Render_Vulkan, "Invalid device index {}!", device_index);
        return false;
    }
    const vk::PhysicalDevice physical_device = devices[device_index];

    if (!VKDevice::IsSuitable(dld, physical_device, *surface)) {
        return false;
    }

    device = std::make_unique<VKDevice>(dld, physical_device, *surface);
    return device->Create(*instance);
}

void RendererVulkan::Report() const {
    const std::string vendor_name{device->GetVendorName()};
    const std::string model_name{device->GetModelName()};
    const std::string driver_version = GetDriverVersion(*device);
    const std::string driver_name = fmt::format("{} {}", vendor_name, driver_version);

    const std::string api_version = GetReadableVersion(device->GetApiVersion());

    const std::string extensions = BuildCommaSeparatedExtensions(device->GetAvailableExtensions());

    LOG_INFO(Render_Vulkan, "Driver: {}", driver_name);
    LOG_INFO(Render_Vulkan, "Device: {}", model_name);
    LOG_INFO(Render_Vulkan, "Vulkan: {}", api_version);

    auto& telemetry_session = system.TelemetrySession();
    constexpr auto field = Telemetry::FieldType::UserSystem;
    telemetry_session.AddField(field, "GPU_Vendor", vendor_name);
    telemetry_session.AddField(field, "GPU_Model", model_name);
    telemetry_session.AddField(field, "GPU_Vulkan_Driver", driver_name);
    telemetry_session.AddField(field, "GPU_Vulkan_Version", api_version);
    telemetry_session.AddField(field, "GPU_Vulkan_Extensions", extensions);
}

} // namespace Vulkan
