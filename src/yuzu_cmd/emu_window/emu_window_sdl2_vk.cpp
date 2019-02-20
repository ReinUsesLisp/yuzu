// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <string>
#include <vector>
#include <SDL.h>
#include <SDL_vulkan.h>
#include <fmt/format.h>
#include <vulkan/vulkan.h>
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/scm_rev.h"
#include "yuzu_cmd/emu_window/emu_window_sdl2_vk.h"

#pragma optimize("", off)

static VkBool32 DebugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT object_type,
                              u64 object, size_t location, s32 message_code,
                              const char* layer_prefix, const char* message, void* user_data) {
    if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
        LOG_CRITICAL(Render_Vulkan, "{}", message);
        UNREACHABLE();
    } else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
        LOG_WARNING(Render_Vulkan, "{}", message);
    } else if (flags &
               (VK_DEBUG_REPORT_DEBUG_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)) {
        LOG_DEBUG(Render_Vulkan, "{}", message);
    } else if (flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) {
        LOG_TRACE(Render_Vulkan, "{}", message);
    }
    return VK_FALSE;
}

EmuWindow_SDL2_VK::EmuWindow_SDL2_VK(bool fullscreen) : EmuWindow_SDL2(fullscreen) {
    if (SDL_Vulkan_LoadLibrary(nullptr) != 0) {
        LOG_CRITICAL(Frontend, "SDL failed to load the Vulkan library: {}", SDL_GetError());
        exit(EXIT_FAILURE);
    }

    vkGetInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
    if (vkGetInstanceProcAddr == nullptr) {
        LOG_CRITICAL(Frontend, "Failed to retrieve Vulkan function pointer!");
        exit(EXIT_FAILURE);
    }

    const std::string window_title = fmt::format("yuzu {} | {}-{} (Vulkan)", Common::g_build_name,
                                                 Common::g_scm_branch, Common::g_scm_desc);
    render_window =
        SDL_CreateWindow(window_title.c_str(),
                         SDL_WINDOWPOS_UNDEFINED, // x position
                         SDL_WINDOWPOS_UNDEFINED, // y position
                         Layout::ScreenUndocked::Width, Layout::ScreenUndocked::Height,
                         SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_VULKAN);

    const bool use_standard_layers = UseStandardLayers(vkGetInstanceProcAddr);

    u32 extra_ext_count{};
    if (!SDL_Vulkan_GetInstanceExtensions(render_window, &extra_ext_count, NULL)) {
        LOG_CRITICAL(Frontend, "Failed to query Vulkan extensions count from SDL! {}",
                     SDL_GetError());
        exit(1);
    }

    auto extra_ext_names = std::make_unique<const char* []>(extra_ext_count);
    if (!SDL_Vulkan_GetInstanceExtensions(render_window, &extra_ext_count, extra_ext_names.get())) {
        LOG_CRITICAL(Frontend, "Failed to query Vulkan extensions from SDL! {}", SDL_GetError());
        exit(1);
    }
    std::vector<const char*> enabled_extensions;
    enabled_extensions.insert(enabled_extensions.begin(), extra_ext_names.get(),
                              extra_ext_names.get() + extra_ext_count);

    std::vector<const char*> enabled_layers;
    if (use_standard_layers) {
        enabled_extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        enabled_layers.push_back("VK_LAYER_LUNARG_standard_validation");
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_1;
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pApplicationName = "yuzu-emu";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "yuzu-emu";

    VkInstanceCreateInfo instance_ci{};
    instance_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_ci.pApplicationInfo = &app_info;
    instance_ci.enabledExtensionCount = static_cast<u32>(enabled_extensions.size());
    instance_ci.ppEnabledExtensionNames = enabled_extensions.data();
    if (enable_layers) {
        instance_ci.enabledLayerCount = static_cast<u32>(enabled_layers.size());
        instance_ci.ppEnabledLayerNames = enabled_layers.data();
    }

    const auto vkCreateInstance =
        reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
    if (vkCreateInstance == nullptr ||
        vkCreateInstance(&instance_ci, nullptr, &instance) != VK_SUCCESS) {
        LOG_CRITICAL(Frontend, "Failed to create Vulkan instance!");
        exit(EXIT_FAILURE);
    }

    vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
        vkGetInstanceProcAddr(instance, "vkDestroyInstance"));
    if (vkDestroyInstance == nullptr) {
        LOG_CRITICAL(Frontend, "Failed to retrieve Vulkan function pointer!");
        exit(EXIT_FAILURE);
    }

    if (use_standard_layers) {
        VkDebugReportCallbackCreateInfoEXT callback_ci{};
        callback_ci.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        callback_ci.pfnCallback = DebugCallback;
        callback_ci.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT |
                            VK_DEBUG_REPORT_DEBUG_BIT_EXT | VK_DEBUG_REPORT_INFORMATION_BIT_EXT |
                            VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;

        vkCreateDebugReportCallbackEXT = reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT"));
        vkDestroyDebugReportCallbackEXT = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));

        if (vkCreateDebugReportCallbackEXT == nullptr ||
            vkDestroyDebugReportCallbackEXT == nullptr ||
            vkCreateDebugReportCallbackEXT(instance, &callback_ci, nullptr, &debug_report) !=
                VK_SUCCESS) {
            LOG_CRITICAL(Frontend, "Failed to setup Vulkan debug callback!");
        }
    }

    if (!SDL_Vulkan_CreateSurface(render_window, instance, &surface)) {
        LOG_CRITICAL(Frontend, "Failed to create Vulkan surface! {}", SDL_GetError());
        exit(EXIT_FAILURE);
    }

    OnResize();
    OnMinimalClientAreaChangeRequest(GetActiveConfig().min_client_area_size);
    SDL_PumpEvents();
    LOG_INFO(Frontend, "yuzu Version: {} | {}-{} (Vulkan)", Common::g_build_name,
             Common::g_scm_branch, Common::g_scm_desc);
}

EmuWindow_SDL2_VK::~EmuWindow_SDL2_VK() {
    if (debug_report) {
        vkDestroyDebugReportCallbackEXT(instance, debug_report, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
}

void EmuWindow_SDL2_VK::SwapBuffers() {}

void EmuWindow_SDL2_VK::MakeCurrent() {
    // Unused on Vulkan
}

void EmuWindow_SDL2_VK::DoneCurrent() {
    // Unused on Vulkan
}

void EmuWindow_SDL2_VK::RetrieveVulkanHandlers(void** get_instance_proc_addr, void** instance,
                                               void** surface) const {
    *get_instance_proc_addr = reinterpret_cast<void*>(vkGetInstanceProcAddr);
    *instance = this->instance;
    *surface = this->surface;
}

bool EmuWindow_SDL2_VK::UseStandardLayers(PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr) const {
    if (!enable_layers) {
        return false;
    }

    const auto vkEnumerateInstanceLayerProperties =
        reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
            vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceLayerProperties"));
    if (vkEnumerateInstanceLayerProperties == nullptr) {
        LOG_CRITICAL(Frontend, "Failed to retrieve Vulkan function pointer!");
        return false;
    }

    u32 available_layers_count{};
    if (vkEnumerateInstanceLayerProperties(&available_layers_count, nullptr) != VK_SUCCESS) {
        LOG_CRITICAL(Frontend, "Failed to enumerate Vulkan validation layers!");
        return false;
    }
    std::vector<VkLayerProperties> layers(available_layers_count);
    if (vkEnumerateInstanceLayerProperties(&available_layers_count, layers.data()) != VK_SUCCESS) {
        LOG_CRITICAL(Frontend, "Failed to enumerate Vulkan validation layers!");
        return false;
    }

    return std::find_if(layers.begin(), layers.end(), [&](const auto& layer) {
               return layer.layerName == std::string("VK_LAYER_LUNARG_standard_validation");
           }) != layers.end();
}
