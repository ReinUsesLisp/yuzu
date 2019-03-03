// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <vector>

#include "common/assert.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_image.h"

namespace Vulkan {

VKImage::VKImage(const VKDevice& device, const vk::ImageCreateInfo& image_ci,
                 vk::ImageAspectFlags aspect_mask)
    : device{device}, format{image_ci.format}, aspect_mask{aspect_mask},
      num_layers{image_ci.arrayLayers}, num_levels{image_ci.mipLevels} {
    UNIMPLEMENTED_IF_MSG(image_ci.queueFamilyIndexCount != 0,
                         "Queue family tracking is not implemented");

    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    image = dev.createImageUnique(image_ci, nullptr, dld);

    const u32 num_ranges = num_layers * num_levels;
    barriers.resize(num_ranges);
    subrange_states.resize(num_ranges, {{}, image_ci.initialLayout, VK_QUEUE_FAMILY_IGNORED});
}

VKImage::~VKImage() = default;

void VKImage::Transition(vk::CommandBuffer cmdbuf, u32 base_layer, u32 num_layers, u32 base_level,
                         u32 num_levels, vk::PipelineStageFlags new_stage_mask,
                         vk::AccessFlags new_access, vk::ImageLayout new_layout, u32 new_family) {
    std::size_t i = 0;
    for (u32 layer_it = 0; layer_it < num_layers; ++layer_it) {
        for (u32 level_it = 0; level_it < num_levels; ++level_it, ++i) {
            const u32 layer = base_layer + layer_it;
            const u32 level = base_level + level_it;
            auto& state = GetSubrangeState(layer, level);
            barriers[i] = vk::ImageMemoryBarrier(state.access, new_access, state.layout, new_layout,
                                                 state.family, new_family, *image,
                                                 {aspect_mask, level, 1, layer, 1});
            state.access = new_access;
            state.layout = new_layout;
            state.family = new_family;
        }
    }

    // TODO(Rodrigo): Implement a way to use the latest stage across subresources.
    constexpr auto stage_stub = vk::PipelineStageFlagBits::eAllCommands;

    const auto& dld = device.GetDispatchLoader();
    cmdbuf.pipelineBarrier(stage_stub, stage_stub, {}, 0, nullptr, 0, nullptr, static_cast<u32>(i),
                           barriers.data(), dld);
}

void VKImage::CreatePresentView() {
    // Image type has to be 2D to be presented.
    const vk::ImageViewCreateInfo image_view_ci({}, *image, vk::ImageViewType::e2D, format, {},
                                                {aspect_mask, 0, 1, 0, 1});
    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    present_view = dev.createImageViewUnique(image_view_ci, nullptr, dld);
}

VKImage::SubrangeState& VKImage::GetSubrangeState(u32 layer, u32 level) {
    return subrange_states[layer * num_levels + level];
}

} // namespace Vulkan