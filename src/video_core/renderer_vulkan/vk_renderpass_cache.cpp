// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

VKRenderPassCache::VKRenderPassCache(const VKDevice& device) : device{device} {}

VKRenderPassCache::~VKRenderPassCache() = default;

vk::RenderPass VKRenderPassCache::GetDrawRenderPass(const RenderPassParams& params) {
    const auto [pair, is_cache_miss] = cache.try_emplace(params);
    auto& entry = pair->second;

    if (is_cache_miss) {
        entry = std::make_unique<CacheEntry>();
    }
    if (!entry->draw) {
        entry->draw = CreateRenderPass(params, true);
    }
    return *entry->draw;
}

vk::RenderPass VKRenderPassCache::GetClearRenderPass(const RenderPassParams& params) {
    const auto [pair, is_cache_miss] = cache.try_emplace(params);
    auto& entry = pair->second;

    if (is_cache_miss) {
        entry = std::make_unique<CacheEntry>();
    }
    if (!entry->clear) {
        entry->clear = CreateRenderPass(params, false);
    }
    return *entry->clear;
}

UniqueRenderPass VKRenderPassCache::CreateRenderPass(const RenderPassParams& params, bool is_draw) {
    UNIMPLEMENTED_IF(params.color_map.Size() != 1);

    const vk::AttachmentLoadOp load_op =
        is_draw ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear;
    const bool has_zeta = params.has_zeta;

    StaticVector<vk::AttachmentDescription, Maxwell::NumRenderTargets + 1> descriptors;
    const auto& first_map = params.color_map[0];

    const auto [color_format, color_attachable] = MaxwellToVK::SurfaceFormat(
        device, FormatType::Optimal, first_map.pixel_format, first_map.component_type);
    ASSERT(color_attachable);

    descriptors.Push(vk::AttachmentDescription(
        {}, color_format, vk::SampleCountFlagBits::e1, load_op, vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal));

    if (has_zeta) {
        const auto [zeta_format, zeta_attachable] = MaxwellToVK::SurfaceFormat(
            device, FormatType::Optimal, params.zeta_pixel_format, params.zeta_component_type);
        ASSERT(zeta_attachable);

        descriptors.Push(vk::AttachmentDescription(
            {}, zeta_format, vk::SampleCountFlagBits::e1, load_op, vk::AttachmentStoreOp::eStore,
            load_op, vk::AttachmentStoreOp::eStore, vk::ImageLayout::eDepthStencilAttachmentOptimal,
            vk::ImageLayout::eDepthStencilAttachmentOptimal));
    }

    // TODO(Rodrigo): Support multiple attachments
    const vk::AttachmentReference color_attachment_ref(0, vk::ImageLayout::eColorAttachmentOptimal);
    const vk::AttachmentReference zeta_attachment_ref(
        1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    const vk::SubpassDescription subpass_description(
        {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr,
        has_zeta ? &zeta_attachment_ref : nullptr, 0, nullptr);

    vk::AccessFlags access{};
    vk::PipelineStageFlags stage{};
    if (is_draw)
        access |= vk::AccessFlagBits::eColorAttachmentRead;
    access |= vk::AccessFlagBits::eColorAttachmentWrite;
    stage |= vk::PipelineStageFlagBits::eColorAttachmentOutput;

    if (has_zeta) {
        if (is_draw)
            access |= vk::AccessFlagBits::eDepthStencilAttachmentRead;
        access |= vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        stage |= vk::PipelineStageFlagBits::eLateFragmentTests;
    }

    const vk::SubpassDependency subpass_dependency(VK_SUBPASS_EXTERNAL, 0, stage, stage, {}, access,
                                                   {});

    const vk::RenderPassCreateInfo create_info({}, static_cast<u32>(descriptors.Size()),
                                               descriptors.Data(), 1, &subpass_description, 1,
                                               &subpass_dependency);

    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    return dev.createRenderPassUnique(create_info, nullptr, dld);
}

} // namespace Vulkan