// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <vector>

#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"

namespace Vulkan {

VKRenderPassCache::VKRenderPassCache(const VKDevice& device) : device{device} {}

VKRenderPassCache::~VKRenderPassCache() = default;

vk::RenderPass VKRenderPassCache::GetRenderPass(const RenderPassParams& params) {
    const auto [pair, is_cache_miss] = cache.try_emplace(params);
    auto& entry = pair->second;
    if (is_cache_miss) {
        entry = CreateRenderPass(params);
    }
    return *entry;
}

UniqueRenderPass VKRenderPassCache::CreateRenderPass(const RenderPassParams& params) const {
    std::vector<vk::AttachmentDescription> descriptors;
    std::vector<vk::AttachmentReference> color_references;

    for (std::size_t rt = 0; rt < params.color_map.Size(); ++rt) {
        const auto attachment = params.color_map[rt];
        const auto [color_format, color_attachable] = MaxwellToVK::SurfaceFormat(
            device, FormatType::Optimal, attachment.pixel_format, attachment.component_type);
        ASSERT_MSG(color_attachable, "Trying to attach a non-attacheable format with format {}",
                   static_cast<u32>(attachment.pixel_format));

        const auto color_layout = attachment.is_texception
                                      ? vk::ImageLayout::eSharedPresentKHR
                                      : vk::ImageLayout::eColorAttachmentOptimal;
        descriptors.push_back(vk::AttachmentDescription(
            {}, color_format, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eLoad,
            vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare, color_layout, color_layout));
        color_references.emplace_back(static_cast<u32>(rt), color_layout);
    }

    const bool has_zeta = params.has_zeta;
    if (has_zeta) {
        const auto [zeta_format, zeta_attachable] = MaxwellToVK::SurfaceFormat(
            device, FormatType::Optimal, params.zeta_pixel_format, params.zeta_component_type);
        ASSERT(zeta_attachable);

        descriptors.push_back(vk::AttachmentDescription(
            {}, zeta_format, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eLoad,
            vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eLoad,
            vk::AttachmentStoreOp::eStore, vk::ImageLayout::eDepthStencilAttachmentOptimal,
            vk::ImageLayout::eDepthStencilAttachmentOptimal));
    }

    const vk::AttachmentReference zeta_attachment_ref(
        static_cast<u32>(params.color_map.Size()), vk::ImageLayout::eDepthStencilAttachmentOptimal);

    const vk::SubpassDescription subpass_description(
        {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, static_cast<u32>(color_references.size()),
        color_references.data(), nullptr, has_zeta ? &zeta_attachment_ref : nullptr, 0, nullptr);

    vk::AccessFlags access;
    vk::PipelineStageFlags stage;
    if (!color_references.empty()) {
        access |=
            vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
        stage |= vk::PipelineStageFlagBits::eColorAttachmentOutput;
    }

    if (has_zeta) {
        access |= vk::AccessFlagBits::eDepthStencilAttachmentRead |
                  vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        stage |= vk::PipelineStageFlagBits::eLateFragmentTests;
    }

    const vk::SubpassDependency subpass_dependency(VK_SUBPASS_EXTERNAL, 0, stage, stage, {}, access,
                                                   {});

    const vk::RenderPassCreateInfo create_info({}, static_cast<u32>(descriptors.size()),
                                               descriptors.data(), 1, &subpass_description, 1,
                                               &subpass_dependency);

    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    return dev.createRenderPassUnique(create_info, nullptr, dld);
}

} // namespace Vulkan