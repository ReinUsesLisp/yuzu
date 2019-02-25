// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <unordered_map>
#include <boost/functional/hash.hpp>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/textures/texture.h"

namespace Vulkan {

class VKDevice;

struct SamplerCacheKey : public Tegra::Texture::TSCEntry {
    std::size_t Hash() const {
        std::size_t hash = 0;
        boost::hash_combine(hash, raw0);
        boost::hash_combine(hash, raw1);
        boost::hash_combine(hash, raw2);
        boost::hash_combine(hash, raw3);
        boost::hash_combine(hash, border_color_r);
        boost::hash_combine(hash, border_color_g);
        boost::hash_combine(hash, border_color_b);
        boost::hash_combine(hash, border_color_a);
        return hash;
    }

    bool operator==(const SamplerCacheKey& rhs) const {
        return std::tie(raw0, raw1, raw2, raw3, border_color_r, border_color_g, border_color_b,
                        border_color_a) == std::tie(rhs.raw0, rhs.raw1, rhs.raw2, rhs.raw3,
                                                    rhs.border_color_r, rhs.border_color_g,
                                                    rhs.border_color_b, rhs.border_color_a);
    }
};

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::SamplerCacheKey> {
    std::size_t operator()(const Vulkan::SamplerCacheKey& k) const {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class VKSamplerCache {
public:
    VKSamplerCache(const VKDevice& device);
    ~VKSamplerCache();

    vk::Sampler GetSampler(const Tegra::Texture::TSCEntry& tsc);

private:
    UniqueSampler CreateSampler(const Tegra::Texture::TSCEntry& tsc);

    const VKDevice& device;
    std::unordered_map<SamplerCacheKey, UniqueSampler> cache;
};

} // namespace Vulkan