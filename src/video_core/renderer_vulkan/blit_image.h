// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <compare>

#include "video_core/engines/fermi_2d.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/wrapper.h"
#include "video_core/texture_cache/types.h"

namespace Vulkan {

using VideoCommon::Offset2D;

class VKDevice;
class VKScheduler;
class StateTracker;

class Framebuffer;
class ImageView;

struct BlitImagePipelineKey {
    constexpr auto operator<=>(const BlitImagePipelineKey&) const noexcept = default;

    VkRenderPass renderpass;
    Tegra::Engines::Fermi2D::Operation operation;
};

class BlitImageHelper {
public:
    explicit BlitImageHelper(const VKDevice& device, VKScheduler& scheduler,
                             StateTracker& state_tracker, VKDescriptorPool& descriptor_pool);
    ~BlitImageHelper();

    void BlitColor(const Framebuffer* dst_framebuffer, const ImageView& src_image_view,
                   const std::array<Offset2D, 2>& dst_region,
                   const std::array<Offset2D, 2>& src_region,
                   Tegra::Engines::Fermi2D::Filter filter,
                   Tegra::Engines::Fermi2D::Operation operation);

    void ConvertD32ToR32(const Framebuffer* dst_framebuffer, const ImageView& src_image_view);

    void ConvertR32ToD32(const Framebuffer* dst_framebuffer, const ImageView& src_image_view);

    void ConvertD16ToR16(const Framebuffer* dst_framebuffer, const ImageView& src_image_view);

    void ConvertR16ToD16(const Framebuffer* dst_framebuffer, const ImageView& src_image_view);

private:
    void Convert(VkPipeline pipeline, const Framebuffer* dst_framebuffer,
                 const ImageView& src_image_view);

    [[nodiscard]] VkPipeline FindOrEmplacePipeline(const BlitImagePipelineKey& key);

    void ConvertDepthToColorPipeline(vk::Pipeline& pipeline, VkRenderPass renderpass);

    void ConvertColorToDepthPipeline(vk::Pipeline& pipeline, VkRenderPass renderpass);

    const VKDevice& device;
    VKScheduler& scheduler;
    StateTracker& state_tracker;

    vk::DescriptorSetLayout set_layout;
    DescriptorAllocator descriptor_allocator;
    vk::ShaderModule full_screen_vert;
    vk::ShaderModule blit_color_to_color_frag;
    vk::ShaderModule convert_depth_to_float_frag;
    vk::ShaderModule convert_float_to_depth_frag;
    vk::Sampler linear_sampler;
    vk::Sampler nearest_sampler;
    vk::PipelineLayout pipeline_layout;

    std::vector<BlitImagePipelineKey> keys;
    std::vector<vk::Pipeline> pipelines;
    vk::Pipeline convert_d32_to_r32_pipeline;
    vk::Pipeline convert_r32_to_d32_pipeline;
    vk::Pipeline convert_d16_to_r16_pipeline;
    vk::Pipeline convert_r16_to_d16_pipeline;
};

} // namespace Vulkan
