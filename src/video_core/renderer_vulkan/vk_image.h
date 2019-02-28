// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/declarations.h"

namespace Vulkan {

class VKDevice;

class VKImage {
public:
    VKImage(const VKDevice& device, const vk::ImageCreateInfo& image_ci,
            vk::ImageAspectFlags aspect_mask);
    ~VKImage();

    void Transition(vk::CommandBuffer cmdbuf, u32 base_layer, u32 layers, u32 base_level,
                    u32 levels, vk::PipelineStageFlags new_stage_mask, vk::AccessFlags new_access,
                    vk::ImageLayout new_layout, u32 new_family = VK_QUEUE_FAMILY_IGNORED);

    vk::ImageView GetPresentView() {
        if (!present_view)
            CreatePresentView();
        return *present_view;
    }

    vk::Image GetHandle() const {
        return *image;
    }

    vk::Format GetFormat() const {
        return format;
    }

    vk::ImageAspectFlags GetAspectMask() const {
        return aspect_mask;
    }

private:
    struct SubrangeState {
        vk::AccessFlags access = {};
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
        u32 family = VK_QUEUE_FAMILY_IGNORED;
    };

    void CreatePresentView();

    SubrangeState& GetState(u32 layer, u32 level);

    const VKDevice& device;
    const vk::Format format;
    const vk::ImageAspectFlags aspect_mask;
    const u32 layers;
    const u32 levels;

    UniqueImage image;
    UniqueImageView present_view;

    vk::PipelineStageFlags current_stage_mask = vk::PipelineStageFlagBits::eTopOfPipe;

    std::vector<vk::ImageMemoryBarrier> barriers;
    std::vector<SubrangeState> states;
};

} // namespace Vulkan
