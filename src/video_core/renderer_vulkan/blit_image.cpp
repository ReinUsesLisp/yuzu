// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>

#include "video_core/host_shaders/convert_depth_to_float_frag_spv.h"
#include "video_core/host_shaders/convert_float_to_depth_frag_spv.h"
#include "video_core/host_shaders/full_screen_triangle_vert_spv.h"
#include "video_core/host_shaders/vulkan_blit_color_float_frag_spv.h"
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

using VideoCommon::ImageViewType;

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
constexpr VkPipelineColorBlendStateCreateInfo PIPELINE_COLOR_BLEND_STATE_EMPTY_CREATE_INFO{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .logicOpEnable = VK_FALSE,
    .logicOp = VK_LOGIC_OP_CLEAR,
    .attachmentCount = 0,
    .pAttachments = nullptr,
    .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
};
constexpr VkPipelineColorBlendAttachmentState PIPELINE_COLOR_BLEND_ATTACHMENT_STATE{
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
constexpr VkPipelineColorBlendStateCreateInfo PIPELINE_COLOR_BLEND_STATE_GENERIC_CREATE_INFO{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .logicOpEnable = VK_FALSE,
    .logicOp = VK_LOGIC_OP_CLEAR,
    .attachmentCount = 1,
    .pAttachments = &PIPELINE_COLOR_BLEND_ATTACHMENT_STATE,
    .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
};
constexpr VkPipelineDepthStencilStateCreateInfo PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .depthTestEnable = VK_FALSE,
    .depthWriteEnable = VK_TRUE,
    .depthCompareOp = VK_COMPARE_OP_ALWAYS,
    .depthBoundsTestEnable = VK_FALSE,
    .stencilTestEnable = VK_FALSE,
    .front = VkStencilOpState{},
    .back = VkStencilOpState{},
    .minDepthBounds = 0.0f,
    .maxDepthBounds = 0.0f,
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

constexpr VkPipelineShaderStageCreateInfo PipelineShaderStageCreateInfo(VkShaderStageFlagBits stage,
                                                                        VkShaderModule shader) {
    return VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = stage,
        .module = shader,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };
}

constexpr std::array<VkPipelineShaderStageCreateInfo, 2> PipelineShaderStageCreateInfos(
    VkShaderModule vertex_shader, VkShaderModule fragment_shader) {
    return std::array{
        PipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertex_shader),
        PipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader),
    };
}

void UpdateDescriptorSet(const VKDevice& device, VkDescriptorSet descriptor_set, VkSampler sampler,
                         VkImageView image_view) {
    const VkDescriptorImageInfo image_info{
        .sampler = sampler,
        .imageView = image_view,
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
}

} // Anonymous namespace

BlitImageHelper::BlitImageHelper(const VKDevice& device_, VKScheduler& scheduler_,
                                 StateTracker& state_tracker_, VKDescriptorPool& descriptor_pool)
    : device{device_}, scheduler{scheduler_}, state_tracker{state_tracker_},
      set_layout(device.GetLogical().CreateDescriptorSetLayout(DESCRIPTOR_SET_LAYOUT_CREATE_INFO)),
      descriptor_allocator(descriptor_pool, *set_layout),
      full_screen_vert(BuildShader(device, FULL_SCREEN_TRIANGLE_VERT_SPV)),
      blit_color_to_color_frag(BuildShader(device, VULKAN_BLIT_COLOR_FLOAT_FRAG_SPV)),
      convert_depth_to_float_frag(BuildShader(device, CONVERT_DEPTH_TO_FLOAT_FRAG_SPV)),
      convert_float_to_depth_frag(BuildShader(device, CONVERT_FLOAT_TO_DEPTH_FRAG_SPV)),
      linear_sampler(device.GetLogical().CreateSampler(SamplerCreateInfo(VK_FILTER_LINEAR))),
      nearest_sampler(device.GetLogical().CreateSampler(SamplerCreateInfo(VK_FILTER_NEAREST))),
      pipeline_layout(device.GetLogical().CreatePipelineLayout(
          PipelineLayoutCreateInfo(set_layout.address()))) {}

BlitImageHelper::~BlitImageHelper() = default;

void BlitImageHelper::BlitColor(const Framebuffer* dst_framebuffer, const ImageView& src_image_view,
                                const std::array<Offset2D, 2>& dst_region,
                                const std::array<Offset2D, 2>& src_region,
                                Tegra::Engines::Fermi2D::Filter filter,
                                Tegra::Engines::Fermi2D::Operation operation) {
    const bool is_linear = filter == Tegra::Engines::Fermi2D::Filter::Bilinear;
    const BlitImagePipelineKey key{
        .renderpass = dst_framebuffer->RenderPass(),
        .operation = operation,
    };
    const VkPipelineLayout layout = *pipeline_layout;
    const VkImageView src_view = src_image_view.Handle(ImageViewType::e2D);
    const VkSampler sampler = is_linear ? *linear_sampler : *nearest_sampler;
    const VkPipeline pipeline = FindOrEmplacePipeline(key);
    const VkDescriptorSet descriptor_set = descriptor_allocator.Commit();
    scheduler.RequestRenderpass(dst_framebuffer);
    scheduler.Record([dst_region, src_region, pipeline, layout, sampler, src_view, descriptor_set,
                      &device = device](vk::CommandBuffer cmdbuf) {
        const VkOffset2D offset{
            .x = std::min(dst_region[0].x, dst_region[1].x),
            .y = std::min(dst_region[0].y, dst_region[1].y),
        };
        const VkExtent2D extent{
            .width = static_cast<u32>(std::abs(dst_region[1].x - dst_region[0].x)),
            .height = static_cast<u32>(std::abs(dst_region[1].y - dst_region[0].y)),
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
        const float scale_x = static_cast<float>(src_region[1].x - src_region[0].x);
        const float scale_y = static_cast<float>(src_region[1].y - src_region[0].y);
        const PushConstants push_constants{
            .tex_scale = {scale_x, scale_y},
            .tex_offset = {static_cast<float>(src_region[0].x),
                           static_cast<float>(src_region[0].y)},
        };
        UpdateDescriptorSet(device, descriptor_set, sampler, src_view);

        // TODO: Barriers
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, descriptor_set,
                                  nullptr);
        cmdbuf.SetViewport(0, viewport);
        cmdbuf.SetScissor(0, scissor);
        cmdbuf.PushConstants(layout, VK_SHADER_STAGE_VERTEX_BIT, push_constants);
        cmdbuf.Draw(3, 1, 0, 0);
    });
    scheduler.InvalidateState();
}

void BlitImageHelper::ConvertD32ToR32(const Framebuffer* dst_framebuffer,
                                      const ImageView& src_image_view) {
    ConvertDepthToColorPipeline(convert_d32_to_r32_pipeline, dst_framebuffer->RenderPass());
    Convert(*convert_d32_to_r32_pipeline, dst_framebuffer, src_image_view);
}

void BlitImageHelper::ConvertR32ToD32(const Framebuffer* dst_framebuffer,
                                      const ImageView& src_image_view) {

    ConvertColorToDepthPipeline(convert_r32_to_d32_pipeline, dst_framebuffer->RenderPass());
    Convert(*convert_r32_to_d32_pipeline, dst_framebuffer, src_image_view);
}

void BlitImageHelper::ConvertD16ToR16(const Framebuffer* dst_framebuffer,
                                      const ImageView& src_image_view) {
    ConvertDepthToColorPipeline(convert_d16_to_r16_pipeline, dst_framebuffer->RenderPass());
    Convert(*convert_d16_to_r16_pipeline, dst_framebuffer, src_image_view);
}

void BlitImageHelper::ConvertR16ToD16(const Framebuffer* dst_framebuffer,
                                      const ImageView& src_image_view) {
    ConvertColorToDepthPipeline(convert_r16_to_d16_pipeline, dst_framebuffer->RenderPass());
    Convert(*convert_r16_to_d16_pipeline, dst_framebuffer, src_image_view);
}

void BlitImageHelper::Convert(VkPipeline pipeline, const Framebuffer* dst_framebuffer,
                              const ImageView& src_image_view) {
    const VkPipelineLayout layout = *pipeline_layout;
    const VkImageView src_view = src_image_view.Handle(ImageViewType::e2D);
    const VkSampler sampler = *nearest_sampler;
    const VkDescriptorSet descriptor_set = descriptor_allocator.Commit();
    const VkExtent2D extent{
        .width = src_image_view.size.width,
        .height = src_image_view.size.height,
    };
    scheduler.RequestRenderpass(dst_framebuffer);
    scheduler.Record([pipeline, layout, sampler, src_view, descriptor_set, extent,
                      &device = device](vk::CommandBuffer cmdbuf) {
        const VkOffset2D offset{
            .x = 0,
            .y = 0,
        };
        const VkViewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(extent.width),
            .height = static_cast<float>(extent.height),
            .minDepth = 0.0f,
            .maxDepth = 0.0f,
        };
        const VkRect2D scissor{
            .offset = offset,
            .extent = extent,
        };
        const PushConstants push_constants{
            .tex_scale = {viewport.width, viewport.height},
            .tex_offset = {0.0f, 0.0f},
        };
        UpdateDescriptorSet(device, descriptor_set, sampler, src_view);

        // TODO: Barriers
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, descriptor_set,
                                  nullptr);
        cmdbuf.SetViewport(0, viewport);
        cmdbuf.SetScissor(0, scissor);
        cmdbuf.PushConstants(layout, VK_SHADER_STAGE_VERTEX_BIT, push_constants);
        cmdbuf.Draw(3, 1, 0, 0);
    });
    scheduler.InvalidateState();
}

VkPipeline BlitImageHelper::FindOrEmplacePipeline(const BlitImagePipelineKey& key) {
    const auto it = std::ranges::find(keys, key);
    if (it != keys.end()) {
        return *pipelines[std::distance(keys.begin(), it)];
    }
    keys.push_back(key);

    const std::array shader_create_infos =
        PipelineShaderStageCreateInfos(*full_screen_vert, *blit_color_to_color_frag);
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
    // TODO: programmable blending
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

void BlitImageHelper::ConvertDepthToColorPipeline(vk::Pipeline& pipeline, VkRenderPass renderpass) {
    if (pipeline) {
        return;
    }
    const std::array shader_create_infos =
        PipelineShaderStageCreateInfos(*full_screen_vert, *convert_depth_to_float_frag);
    pipeline = device.GetLogical().CreateGraphicsPipeline({
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
        .pColorBlendState = &PIPELINE_COLOR_BLEND_STATE_GENERIC_CREATE_INFO,
        .pDynamicState = &PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .layout = *pipeline_layout,
        .renderPass = renderpass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    });
}

void BlitImageHelper::ConvertColorToDepthPipeline(vk::Pipeline& pipeline, VkRenderPass renderpass) {
    if (pipeline) {
        return;
    }
    const std::array shader_create_infos =
        PipelineShaderStageCreateInfos(*full_screen_vert, *convert_float_to_depth_frag);
    pipeline = device.GetLogical().CreateGraphicsPipeline({
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
        .pDepthStencilState = &PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pColorBlendState = &PIPELINE_COLOR_BLEND_STATE_EMPTY_CREATE_INFO,
        .pDynamicState = &PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .layout = *pipeline_layout,
        .renderPass = renderpass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    });
}

} // namespace Vulkan
