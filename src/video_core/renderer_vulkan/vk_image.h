// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/declarations.h"

namespace Vulkan {

class VKDevice;

class VKImage {
public:
    VKImage(const VKDevice& device, const vk::ImageCreateInfo& image_ci,
            vk::ImageAspectFlags aspect_mask);
    ~VKImage();

    void Transition(vk::CommandBuffer cmdbuf, vk::ImageSubresourceRange subresource_range,
                    vk::ImageLayout new_layout, vk::PipelineStageFlags new_stage_mask,
                    vk::AccessFlags new_access, u32 new_family = VK_QUEUE_FAMILY_IGNORED);

    vk::ImageView GetPresentView() {
        if (!present_view)
            CreatePresentView();
        return *present_view;
    }

    void UpdateLayout(vk::ImageLayout new_layout, vk::PipelineStageFlags new_stage_mask,
                      vk::AccessFlags new_access) {
        current_layout = new_layout;
        current_stage_mask = new_stage_mask;
        current_access = new_access;
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
    void CreatePresentView();

    const VKDevice& device;
    const vk::Format format;
    const vk::ImageAspectFlags aspect_mask;

    UniqueImage image;
    UniqueImageView present_view;

    vk::ImageLayout current_layout;
    // Note(Rodrigo): Using eTransferWrite and eTopOfPipe here is a hack to have a valid value for
    // the initial transition.
    vk::PipelineStageFlags current_stage_mask = vk::PipelineStageFlagBits::eTopOfPipe;
    vk::AccessFlags current_access{};
    u32 current_family = VK_QUEUE_FAMILY_IGNORED;
};

} // namespace Vulkan
