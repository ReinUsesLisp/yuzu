// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <compare>

#include "video_core/engines/fermi_2d.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/wrapper.h"

namespace Vulkan {

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

class BlitImage {
public:
    explicit BlitImage(const VKDevice& device, VKScheduler& scheduler, StateTracker& state_tracker,
                       VKDescriptorPool& descriptor_pool);
    ~BlitImage();

    void Invoke(const Framebuffer* dst_framebuffer, const ImageView& src_image_view,
                const Tegra::Engines::Fermi2D::Config& config);

private:
    [[nodiscard]] VkPipeline FindOrEmplacePipeline(const BlitImagePipelineKey& key);

    const VKDevice& device;
    VKScheduler& scheduler;
    StateTracker& state_tracker;

    vk::DescriptorSetLayout set_layout;
    DescriptorAllocator descriptor_allocator;
    vk::ShaderModule vertex_shader;
    vk::ShaderModule fragment_shader;
    vk::Sampler linear_sampler;
    vk::Sampler nearest_sampler;
    vk::PipelineLayout pipeline_layout;

    std::vector<BlitImagePipelineKey> keys;
    std::vector<vk::Pipeline> pipelines;
};

} // namespace Vulkan
