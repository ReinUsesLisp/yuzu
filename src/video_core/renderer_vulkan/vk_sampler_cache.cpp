// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <optional>

#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_sampler_cache.h"
#include "video_core/textures/texture.h"

namespace Vulkan {

static std::optional<vk::BorderColor> TryConvertBorderColor(std::array<float, 4> color) {
    // TODO(Rodrigo): Manage integer border colors
    if (color == std::array<float, 4>{0, 0, 0, 0}) {
        return vk::BorderColor::eFloatTransparentBlack;
    } else if (color == std::array<float, 4>{0, 0, 0, 1}) {
        return vk::BorderColor::eFloatOpaqueBlack;
    } else if (color == std::array<float, 4>{1, 1, 1, 1}) {
        return vk::BorderColor::eFloatOpaqueWhite;
    } else {
        return {};
    }
}

VKSamplerCache::VKSamplerCache(const VKDevice& device) : device{device} {}

VKSamplerCache::~VKSamplerCache() = default;

vk::Sampler VKSamplerCache::GetSampler(const Tegra::Texture::TSCEntry& tsc) {
    const auto [entry, is_cache_miss] = cache.try_emplace(SamplerCacheKey{tsc});
    auto& sampler = entry->second;
    if (is_cache_miss) {
        sampler = CreateSampler(tsc);
    }
    return *sampler;
}

UniqueSampler VKSamplerCache::CreateSampler(const Tegra::Texture::TSCEntry& tsc) {
    // Sign extend the 13-bit value.
    constexpr u32 bias_mask = 1U << (13 - 1);
    const float bias_lod =
        static_cast<s32>((tsc.mip_lod_bias.Value() ^ bias_mask) - bias_mask) / 256.0f;

    const auto max_anisotropy = static_cast<float>(1 << tsc.max_anisotropy.Value());
    const bool has_anisotropy = max_anisotropy > 1.0f;

    const auto lod_min = static_cast<float>(tsc.min_lod_clamp.Value()) / 256.0f;
    const auto lod_max = static_cast<float>(tsc.max_lod_clamp.Value()) / 256.0f;

    const std::array<float, 4> border_color = {tsc.border_color_r, tsc.border_color_g,
                                               tsc.border_color_b, tsc.border_color_a};
    const auto vk_border_color = TryConvertBorderColor(border_color);
    UNIMPLEMENTED_IF_MSG(!vk_border_color, "Unimplemented border color {} {} {} {}",
                         border_color[0], border_color[1], border_color[2], border_color[3]);

    constexpr bool unnormalized_coords = false;

    const vk::SamplerCreateInfo sampler_ci(
        {}, MaxwellToVK::Sampler::Filter(tsc.mag_filter),
        MaxwellToVK::Sampler::Filter(tsc.min_filter),
        MaxwellToVK::Sampler::MipmapMode(tsc.mip_filter),
        MaxwellToVK::Sampler::WrapMode(tsc.wrap_u), MaxwellToVK::Sampler::WrapMode(tsc.wrap_v),
        MaxwellToVK::Sampler::WrapMode(tsc.wrap_p), bias_lod, has_anisotropy, max_anisotropy,
        tsc.depth_compare_enabled,
        MaxwellToVK::Sampler::DepthCompareFunction(tsc.depth_compare_func), lod_min, lod_max,
        vk_border_color.value_or(vk::BorderColor::eFloatTransparentBlack), unnormalized_coords);

    const auto& dld = device.GetDispatchLoader();
    const auto dev = device.GetLogical();
    return dev.createSamplerUnique(sampler_ci, nullptr, dld);
}

} // namespace Vulkan
