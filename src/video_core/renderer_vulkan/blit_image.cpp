// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>

#include "video_core/host_shaders/vulkan_blit_color_float_frag_spv.h"
#include "video_core/host_shaders/vulkan_blit_color_float_vert_spv.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/renderer_vulkan/wrapper.h"
#include "video_core/surface.h"

namespace Vulkan {
namespace {
struct PushConstants {
    std::array<float, 2> tex_scale;
    std::array<float, 2> tex_offset;
};

constexpr VkDescriptorSetLayoutBinding DESCRIPTOR_SET_LAYOUT_BINDING{
    .binding = 0,
    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    .descriptorCount = 1,
    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    .pImmutableSamplers = nullptr,
};
constexpr VkDescriptorSetLayoutCreateInfo DESCRIPTOR_SET_LAYOUT_CREATE_INFO{
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .bindingCount = 1,
    .pBindings = &DESCRIPTOR_SET_LAYOUT_BINDING,
};
constexpr VkPushConstantRange PUSH_CONSTANT_RANGE{
    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    .offset = 0,
    .size = sizeof(PushConstants),
};
constexpr VkPipelineVertexInputStateCreateInfo PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .vertexBindingDescriptionCount = 0,
    .pVertexBindingDescriptions = nullptr,
    .vertexAttributeDescriptionCount = 0,
    .pVertexAttributeDescriptions = nullptr,
};
constexpr VkPipelineInputAssemblyStateCreateInfo PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    .primitiveRestartEnable = VK_FALSE,
};
constexpr VkPipelineViewportStateCreateInfo PIPELINE_VIEWPORT_STATE_CREATE_INFO{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .viewportCount = 1,
    .pViewports = nullptr,
    .scissorCount = 1,
    .pScissors = nullptr,
};
constexpr VkPipelineRasterizationStateCreateInfo PIPELINE_RASTERIZATION_STATE_CREATE_INFO{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .depthClampEnable = VK_FALSE,
    .rasterizerDiscardEnable = VK_FALSE,
    .polygonMode = VK_POLYGON_MODE_FILL,
    .cullMode = VK_CULL_MODE_BACK_BIT,
    .frontFace = VK_FRONT_FACE_CLOCKWISE,
    .depthBiasEnable = VK_FALSE,
    .depthBiasConstantFactor = 0.0f,
    .depthBiasClamp = 0.0f,
    .depthBiasSlopeFactor = 0.0f,
    .lineWidth = 1.0f,
};
constexpr VkPipelineMultisampleStateCreateInfo PIPELINE_MULTISAMPLE_STATE_CREATE_INFO{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    .sampleShadingEnable = VK_FALSE,
    .minSampleShading = 0.0f,
    .pSampleMask = nullptr,
    .alphaToCoverageEnable = VK_FALSE,
    .alphaToOneEnable = VK_FALSE,
};
constexpr std::array DYNAMIC_STATES{
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR,
};
constexpr VkPipelineDynamicStateCreateInfo PIPELINE_DYNAMIC_STATE_CREATE_INFO{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .dynamicStateCount = static_cast<u32>(DYNAMIC_STATES.size()),
    .pDynamicStates = DYNAMIC_STATES.data(),
};

constexpr VkSamplerCreateInfo SamplerCreateInfo(VkFilter filter) {
    return VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = filter,
        .minFilter = filter,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 0.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
        .unnormalizedCoordinates = VK_TRUE,
    };
};

constexpr VkPipelineLayoutCreateInfo PipelineLayoutCreateInfo(
    const VkDescriptorSetLayout* set_layout) {
    return VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &PUSH_CONSTANT_RANGE,
    };
}

} // Anonymous namespace

BlitImageHelper::BlitImageHelper(const VKDevice& device_, VKScheduler& scheduler_,
                                 StateTracker& state_tracker_, VKDescriptorPool& descriptor_pool)
    : device{device_}, scheduler{scheduler_}, state_tracker{state_tracker_},
      set_layout(device.GetLogical().CreateDescriptorSetLayout(DESCRIPTOR_SET_LAYOUT_CREATE_INFO)),
      descriptor_allocator(descriptor_pool, *set_layout),
      vertex_shader(BuildShader(device, VULKAN_BLIT_COLOR_FLOAT_VERT_SPV)),
      fragment_shader(BuildShader(device, VULKAN_BLIT_COLOR_FLOAT_FRAG_SPV)),
      linear_sampler(device.GetLogical().CreateSampler(SamplerCreateInfo(VK_FILTER_LINEAR))),
      nearest_sampler(device.GetLogical().CreateSampler(SamplerCreateInfo(VK_FILTER_NEAREST))),
      pipeline_layout(device.GetLogical().CreatePipelineLayout(
          PipelineLayoutCreateInfo(set_layout.address()))) {}

BlitImageHelper::~BlitImageHelper() = default;

void BlitImageHelper::Invoke(const Framebuffer* dst_framebuffer, const ImageView& src_image_view,
                             const Tegra::Engines::Fermi2D::Config& config) {
    const bool is_linear = config.filter == Tegra::Engines::Fermi2D::Filter::Bilinear;
    const BlitImagePipelineKey key{
        .renderpass = dst_framebuffer->RenderPass(),
        .operation = config.operation,
    };
    const VkPipelineLayout layout = *pipeline_layout;
    const VkImageView src_view = src_image_view.RenderTarget();
    const VkSampler sampler = is_linear ? *linear_sampler : *nearest_sampler;
    const VkPipeline pipeline = FindOrEmplacePipeline(key);
    const VkDescriptorSet descriptor_set = descriptor_allocator.Commit();
    scheduler.RequestRenderpass(dst_framebuffer);
    scheduler.Record([pipeline, layout, sampler, config, src_view, descriptor_set,
                      &device = device](vk::CommandBuffer cmdbuf) {
        const VkOffset2D offset{
            .x = std::min(config.dst_x0, config.dst_x1),
            .y = std::min(config.dst_y0, config.dst_y1),
        };
        const VkExtent2D extent{
            .width = static_cast<u32>(std::abs(config.dst_x1 - config.dst_x0)),
            .height = static_cast<u32>(std::abs(config.dst_y1 - config.dst_y0)),
        };
        const VkViewport viewport{
            .x = static_cast<float>(offset.x),
            .y = static_cast<float>(offset.y),
            .width = static_cast<float>(extent.width),
            .height = static_cast<float>(extent.height),
            .minDepth = 0.0f,
            .maxDepth = 0.0f,
        };
        // TODO: Support scissored blits
        const VkRect2D scissor{
            .offset = offset,
            .extent = extent,
        };
        const float scale_x = static_cast<float>(config.src_x1 - config.src_x0);
        const float scale_y = static_cast<float>(config.src_y1 - config.src_y0);
        const PushConstants push_constants{
            .tex_scale = {scale_x, scale_y},
            .tex_offset = {static_cast<float>(config.src_x0), static_cast<float>(config.src_y0)},
        };
        const VkDescriptorImageInfo image_info{
            .sampler = sampler,
            .imageView = src_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkWriteDescriptorSet write_descriptor_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        device.GetLogical().UpdateDescriptorSets(write_descriptor_set, nullptr);

        // TODO: Barriers
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, descriptor_set,
                                  nullptr);
        cmdbuf.SetViewport(0, viewport);
        cmdbuf.SetScissor(0, scissor);
        cmdbuf.PushConstants(layout, VK_SHADER_STAGE_VERTEX_BIT, push_constants);
        cmdbuf.Draw(3, 1, 0, 0);
    });
    state_tracker.InvalidateViewports();
    state_tracker.InvalidateScissors();
}

VkPipeline BlitImageHelper::FindOrEmplacePipeline(const BlitImagePipelineKey& key) {
    const auto it = std::ranges::find(keys, key);
    if (it != keys.end()) {
        return *pipelines[std::distance(keys.begin(), it)];
    }
    keys.push_back(key);

    const std::array shader_create_infos{
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = *vertex_shader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = *fragment_shader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
    };
    const VkPipelineColorBlendAttachmentState blend_attachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo color_blend_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_CLEAR,
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
        .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
    };
    pipelines.push_back(device.GetLogical().CreateGraphicsPipeline(VkGraphicsPipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = static_cast<u32>(shader_create_infos.size()),
        .pStages = shader_create_infos.data(),
        .pVertexInputState = &PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pInputAssemblyState = &PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pTessellationState = nullptr,
        .pViewportState = &PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pRasterizationState = &PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pMultisampleState = &PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &color_blend_create_info,
        .pDynamicState = &PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .layout = *pipeline_layout,
        .renderPass = key.renderpass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    }));
    return *pipelines.back();
}

} // namespace Vulkan
