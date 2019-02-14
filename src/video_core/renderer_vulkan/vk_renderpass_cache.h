// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <map>
#include <memory>
#include <tuple>
#include "common/static_vector.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/surface.h"

namespace Vulkan {

class VKDevice;

struct RenderPassParams {
    struct ColorAttachment {
        u32 index = 0;
        VideoCore::Surface::PixelFormat pixel_format = VideoCore::Surface::PixelFormat::Invalid;
        VideoCore::Surface::ComponentType component_type =
            VideoCore::Surface::ComponentType::Invalid;

        auto Tie() const {
            return std::tie(index, pixel_format, component_type);
        }

        bool operator<(const ColorAttachment& rhs) const {
            return Tie() < rhs.Tie();
        }
    };

    StaticVector<ColorAttachment, Tegra::Engines::Maxwell3D::Regs::NumRenderTargets> color_map = {};
    // TODO(Rodrigo): Unify has_zeta into zeta_pixel_format and zeta_component_type.
    VideoCore::Surface::PixelFormat zeta_pixel_format = VideoCore::Surface::PixelFormat::Invalid;
    VideoCore::Surface::ComponentType zeta_component_type =
        VideoCore::Surface::ComponentType::Invalid;
    bool has_zeta = false;

    auto Tie() const {
        return std::tie(color_map, zeta_pixel_format, zeta_component_type, has_zeta);
    }

    bool operator<(const RenderPassParams& rhs) const {
        return Tie() < rhs.Tie();
    }
};

class VKRenderPassCache final {
public:
    explicit VKRenderPassCache(const VKDevice& device);
    ~VKRenderPassCache();

    vk::RenderPass GetDrawRenderPass(const RenderPassParams& params);

    vk::RenderPass GetClearRenderPass(const RenderPassParams& params);

private:
    struct CacheEntry {
        UniqueRenderPass draw;
        UniqueRenderPass clear;
    };

    UniqueRenderPass CreateRenderPass(const RenderPassParams& params, bool is_draw);

    const VKDevice& device;
    std::map<RenderPassParams, std::unique_ptr<CacheEntry>> cache;
};

} // namespace Vulkan