// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <vector>
#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

UniqueShaderModule BuildShader(const VKDevice& device, std::size_t code_size, const u8* code_data) {
    LOG_DEBUG(Render_OpenGL, "Creating shader...");

    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    const vk::ShaderModuleCreateInfo shader_ci({}, code_size,
                                               reinterpret_cast<const u32*>(code_data));
    vk::ShaderModule shader_module;
    if (dev.createShaderModule(&shader_ci, nullptr, &shader_module, dld) != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "Shader module failed to build!");
        UNREACHABLE();
    }

    return UniqueShaderModule(shader_module, vk::ObjectDestroy(dev, nullptr, dld));
}

} // namespace Vulkan