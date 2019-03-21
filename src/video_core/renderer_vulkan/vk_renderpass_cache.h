// Copyright 2019 yuzu Emulator Project
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
        bool is_texception = false;

        std::size_t Hash() const;
        bool operator==(const ColorAttachment& rhs) const;
    };

    StaticVector<ColorAttachment, Tegra::Engines::Maxwell3D::Regs::NumRenderTargets>
        color_attachments = {};
    // TODO(Rodrigo): Unify has_zeta into zeta_pixel_format and zeta_component_type.
    VideoCore::Surface::PixelFormat zeta_pixel_format = VideoCore::Surface::PixelFormat::Invalid;
    VideoCore::Surface::ComponentType zeta_component_type =
        VideoCore::Surface::ComponentType::Invalid;
    bool has_zeta = false;
    bool zeta_texception = false;

    std::size_t Hash() const {
        std::size_t hash = 0;
        for (const auto& rt : color_attachments)
            boost::hash_combine(hash, rt.Hash());
        boost::hash_combine(hash, zeta_pixel_format);
        boost::hash_combine(hash, zeta_component_type);
        boost::hash_combine(hash, has_zeta);
        boost::hash_combine(hash, zeta_texception);
        return hash;
    }

    bool operator==(const RenderPassParams& rhs) const {
        return std::tie(color_attachments, zeta_pixel_format, zeta_component_type, has_zeta,
                        zeta_texception) == std::tie(rhs.color_attachments, rhs.zeta_pixel_format,
                                                     rhs.zeta_component_type, rhs.has_zeta,
                                                     rhs.zeta_texception);
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

    vk::RenderPass GetRenderPass(const RenderPassParams& params);

private:
    UniqueRenderPass CreateRenderPass(const RenderPassParams& params) const;

    const VKDevice& device;
    std::unordered_map<RenderPassParams, UniqueRenderPass> cache;
};

} // namespace Vulkan
