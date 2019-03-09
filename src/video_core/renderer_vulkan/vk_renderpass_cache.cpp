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

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

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
    constexpr vk::AttachmentLoadOp load_op = vk::AttachmentLoadOp::eLoad;
    std::vector<vk::AttachmentDescription> descriptors;

    for (const auto& map : params.color_map) {
        const auto [color_format, color_attachable] = MaxwellToVK::SurfaceFormat(
            device, FormatType::Optimal, map.pixel_format, map.component_type);
        ASSERT_MSG(color_attachable, "Trying to attach a non-attacheable format with format {}",
                   static_cast<u32>(map.pixel_format));

        descriptors.push_back(vk::AttachmentDescription(
            {}, color_format, vk::SampleCountFlagBits::e1, load_op, vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal));
    }

    const bool has_zeta = params.has_zeta;
    if (has_zeta) {
        const auto [zeta_format, zeta_attachable] = MaxwellToVK::SurfaceFormat(
            device, FormatType::Optimal, params.zeta_pixel_format, params.zeta_component_type);
        ASSERT(zeta_attachable);

        descriptors.push_back(vk::AttachmentDescription(
            {}, zeta_format, vk::SampleCountFlagBits::e1, load_op, vk::AttachmentStoreOp::eStore,
            load_op, vk::AttachmentStoreOp::eStore, vk::ImageLayout::eDepthStencilAttachmentOptimal,
            vk::ImageLayout::eDepthStencilAttachmentOptimal));
    }

    const auto color_map_count = static_cast<u32>(params.color_map.Size());
    std::vector<vk::AttachmentReference> color_references;
    for (u32 i = 0; i < color_map_count; ++i) {
        color_references.emplace_back(i, vk::ImageLayout::eColorAttachmentOptimal);
    }
    const vk::AttachmentReference zeta_attachment_ref(
        color_map_count, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    const vk::SubpassDescription subpass_description(
        {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, color_map_count, color_references.data(),
        nullptr, has_zeta ? &zeta_attachment_ref : nullptr, 0, nullptr);

    vk::AccessFlags access;
    vk::PipelineStageFlags stage;
    if (color_map_count > 0) {
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