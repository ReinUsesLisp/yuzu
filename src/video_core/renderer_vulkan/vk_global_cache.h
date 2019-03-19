// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <unordered_map>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/rasterizer_cache.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"

namespace Core {
class System;
}

namespace Vulkan {

namespace VKShader {
class GlobalBufferEntry;
}

class VKDevice;
class VKMemoryManager;

class CachedGlobalRegion;

using GlobalRegion = std::shared_ptr<CachedGlobalRegion>;

class CachedGlobalRegion final : public RasterizerCacheObject {
public:
    explicit CachedGlobalRegion(const VKDevice& device, VKMemoryManager& memory_manager,
                                VAddr cpu_addr, u8* host_ptr, u64 size_);
    ~CachedGlobalRegion();

    VAddr GetCpuAddr() const {
        return cpu_addr;
    }

    std::size_t GetSizeInBytes() const {
        return static_cast<std::size_t>(size);
    }

    vk::Buffer GetBufferHandle() const {
        return *buffer;
    }

    void Upload(u64 size_);

    void Flush() override;

private:
    const VKDevice& device;

    VAddr cpu_addr{};
    u8* host_ptr{};
    u64 size{};

    UniqueBuffer buffer;
    VKMemoryCommit commit;
    u8* memory{};
};

class VKGlobalCache : public RasterizerCache<GlobalRegion> {
public:
    explicit VKGlobalCache(Core::System& system, VideoCore::RasterizerInterface& rasterizer,
                           const VKDevice& device, VKMemoryManager& memory_manager);
    ~VKGlobalCache();

    /// Gets the current specified shader stage program.
    GlobalRegion GetGlobalRegion(const VKShader::GlobalBufferEntry& descriptor,
                                 Tegra::Engines::Maxwell3D::Regs::ShaderStage stage);

private:
    GlobalRegion TryGetReservedGlobalRegion(VAddr addr, u64 size) const;
    GlobalRegion GetUncachedGlobalRegion(VAddr addr, u8* host_ptr, u64 size);
    void ReserveGlobalRegion(GlobalRegion region);

    Core::System& system;

    const VKDevice& device;
    VKMemoryManager& memory_manager;

    std::unordered_map<VAddr, GlobalRegion> reserve;
};

} // namespace Vulkan
