// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <tuple>
#include <unordered_map>
#include <boost/functional/hash.hpp>
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

        std::size_t Hash() const {
            std::size_t hash = 0;
            boost::hash_combine(hash, index);
            boost::hash_combine(hash, pixel_format);
            boost::hash_combine(hash, component_type);
            return hash;
        }

        bool operator==(const ColorAttachment& rhs) const {
            return std::tie(index, pixel_format, component_type) ==
                   std::tie(rhs.index, rhs.pixel_format, rhs.component_type);
        }
    };

    StaticVector<ColorAttachment, Tegra::Engines::Maxwell3D::Regs::NumRenderTargets> color_map = {};
    // TODO(Rodrigo): Unify has_zeta into zeta_pixel_format and zeta_component_type.
    VideoCore::Surface::PixelFormat zeta_pixel_format = VideoCore::Surface::PixelFormat::Invalid;
    VideoCore::Surface::ComponentType zeta_component_type =
        VideoCore::Surface::ComponentType::Invalid;
    bool has_zeta = false;

    std::size_t Hash() const {
        std::size_t hash = 0;
        for (const auto& rt : color_map)
            boost::hash_combine(hash, rt.Hash());
        boost::hash_combine(hash, zeta_pixel_format);
        boost::hash_combine(hash, zeta_component_type);
        boost::hash_combine(hash, has_zeta);
        return hash;
    }

    bool operator==(const RenderPassParams& rhs) const {
        return std::tie(color_map, zeta_pixel_format, zeta_component_type, has_zeta) ==
               std::tie(rhs.color_map, rhs.zeta_pixel_format, rhs.zeta_component_type,
                        rhs.has_zeta);
    }
};

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::RenderPassParams> {
    std::size_t operator()(const Vulkan::RenderPassParams& k) const {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

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
    std::unordered_map<RenderPassParams, std::unique_ptr<CacheEntry>> cache;
};

} // namespace Vulkan