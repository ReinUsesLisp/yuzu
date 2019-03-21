// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <list>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/rasterizer_cache.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"

namespace Core {
class System;
}

namespace Vulkan {

namespace VKShader {
class GlobalBufferEntry;
}

class VKDevice;
class VKExecutionContext;
class VKMemoryManager;

class CachedGlobalRegion;

using GlobalRegion = std::shared_ptr<CachedGlobalRegion>;

class CachedGlobalRegion final : public RasterizerCacheObject {
public:
    explicit CachedGlobalRegion(const VKDevice& device, VKMemoryManager& memory_manager,
                                VAddr cpu_addr, u8* host_ptr, u64 size_);
    ~CachedGlobalRegion();

    VAddr GetCpuAddr() const override {
        return cpu_addr;
    }

    std::size_t GetSizeInBytes() const override {
        return static_cast<std::size_t>(size);
    }

    vk::Buffer GetBufferHandle() const {
        return *staging_buffer;
    }

    void CommitRead(VKFence& fence);

    [[nodiscard]] VKExecutionContext Upload(VKExecutionContext exctx);

    void Flush() override;

private:
    void ReserveWatchBucket();

    const VKDevice& device;

    VAddr cpu_addr{};
    u8* host_ptr{};
    u64 size{};

    UniqueBuffer staging_buffer;
    VKMemoryCommit staging_commit;
    u8* memory{};

    VKFenceWatch write_watch;
    std::vector<VKFenceWatch*> read_watches;
    std::vector<std::unique_ptr<VKFenceWatch>> watches_allocation;
};

class VKGlobalCache : public RasterizerCache<GlobalRegion> {
public:
    explicit VKGlobalCache(Core::System& system, VideoCore::RasterizerInterface& rasterizer,
                           const VKDevice& device, VKMemoryManager& memory_manager);
    ~VKGlobalCache();

    /// Gets the current specified shader stage program.
    [[nodiscard]] std::tuple<GlobalRegion, VKExecutionContext> GetGlobalRegion(
        VKExecutionContext exctx, const VKShader::GlobalBufferEntry& descriptor,
        Tegra::Engines::Maxwell3D::Regs::ShaderStage stage);

private:
    [[nodiscard]] std::tuple<GlobalRegion, VKExecutionContext> GetUncachedGlobalRegion(
        VKExecutionContext exctx, VAddr addr, u8* host_ptr, u64 size);

    GlobalRegion TryGetReservedGlobalRegion(u64 size) const;

    void ReserveGlobalRegion(GlobalRegion region);

    Core::System& system;

    const VKDevice& device;
    VKMemoryManager& memory_manager;

    std::unordered_map<u64, std::list<GlobalRegion>> reserve;
};

} // namespace Vulkan
