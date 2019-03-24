// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <memory>

#include "common/assert.h"
#include "common/common_types.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_global_cache.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_decompiler.h"

namespace Vulkan {

CachedGlobalRegion::CachedGlobalRegion(const VKDevice& device, VKMemoryManager& memory_manager,
                                       VAddr cpu_addr, u8* host_ptr, u64 size_)
    : device{device}, cpu_addr{cpu_addr}, host_ptr{host_ptr}, RasterizerCacheObject{host_ptr} {
    const auto max_size{device.GetMaxStorageBufferRange()};
    size = size_;
    if (size > max_size) {
        size = max_size;
        LOG_ERROR(Render_Vulkan,
                  "Global buffer size 0x{:x} is bigger than the maximum supported size by the "
                  "current physical device 0x{:x}",
                  size, max_size);
    }

    const auto& dld{device.GetDispatchLoader()};
    const auto dev{device.GetLogical()};
    const vk::BufferCreateInfo staging_buffer_ci({}, static_cast<vk::DeviceSize>(size),
                                                 vk::BufferUsageFlagBits::eStorageBuffer |
                                                     vk::BufferUsageFlagBits::eTransferSrc,
                                                 vk::SharingMode::eExclusive, 0, nullptr);
    staging_buffer = dev.createBufferUnique(staging_buffer_ci, nullptr, dld);
    staging_commit = memory_manager.Commit(*staging_buffer, true);
    memory = staging_commit->GetData();
}

CachedGlobalRegion::~CachedGlobalRegion() = default;

void CachedGlobalRegion::CommitRead(VKFence& fence) {
    const std::size_t watch_iterator{read_watches.size()};
    if (watches_allocation.size() < watch_iterator) {
        ReserveWatchBucket();
    }
    const auto watch{watches_allocation[watch_iterator].get()};
    read_watches.push_back(watch);
    watch->Watch(fence);
}

VKExecutionContext CachedGlobalRegion::Upload(VKExecutionContext exctx) {
    // TODO(Rodrigo): It might be possible that we could potentially wait an fence that's not in a
    // queue. A lock here will spot this theoretical bug.
    write_watch.Wait();

    for (const auto watch : read_watches) {
        watch->Wait();
    }
    read_watches.clear();

    std::memcpy(memory, host_ptr, static_cast<std::size_t>(size));

    write_watch.Watch(exctx.GetFence());
    return exctx;
}

void CachedGlobalRegion::Flush() {
    UNIMPLEMENTED();
}

void CachedGlobalRegion::ReserveWatchBucket() {
    constexpr std::size_t bucket_size{4};
    const auto old_end = watches_allocation.end();
    watches_allocation.resize(watches_allocation.size() + bucket_size);
    std::generate(old_end, watches_allocation.end(),
                  []() { return std::make_unique<VKFenceWatch>(); });
}

VKGlobalCache::VKGlobalCache(Core::System& system, VideoCore::RasterizerInterface& rasterizer,
                             const VKDevice& device, VKMemoryManager& memory_manager)
    : RasterizerCache{rasterizer}, system{system}, device{device}, memory_manager{memory_manager} {}

VKGlobalCache::~VKGlobalCache() = default;

std::tuple<GlobalRegion, VKExecutionContext> VKGlobalCache::GetGlobalRegion(
    VKExecutionContext exctx, const VKShader::GlobalBufferEntry& descriptor,
    Tegra::Engines::Maxwell3D::Regs::ShaderStage stage) {
    auto& gpu{system.GPU()};
    auto& emu_memory_manager{gpu.MemoryManager()};

    const auto& cbufs{gpu.Maxwell3D().state.shader_stages[static_cast<std::size_t>(stage)]};
    const auto addr{cbufs.const_buffers[descriptor.GetCbufIndex()].address +
                    descriptor.GetCbufOffset()};
    const auto actual_addr{emu_memory_manager.Read<u64>(addr)};
    const auto size{emu_memory_manager.Read<u64>(addr + 8)};

    const auto host_ptr{emu_memory_manager.GetPointer(actual_addr)};
    GlobalRegion region = TryGet(host_ptr);
    if (!region) {
        std::tie(region, exctx) = GetUncachedGlobalRegion(exctx, actual_addr, host_ptr, size);
        Register(region);
    }
    return {region, exctx};
}

std::tuple<GlobalRegion, VKExecutionContext> VKGlobalCache::GetUncachedGlobalRegion(
    VKExecutionContext exctx, VAddr addr, u8* host_ptr, u64 size) {
    GlobalRegion region{TryGetReservedGlobalRegion(size)};
    if (!region) {
        region = std::make_shared<CachedGlobalRegion>(device, memory_manager, addr, host_ptr, size);
        ReserveGlobalRegion(region);
    }
    exctx = region->Upload(exctx);
    return {region, exctx};
}

GlobalRegion VKGlobalCache::TryGetReservedGlobalRegion(u64 size) const {
    if (const auto search{reserve.find(size)}; search != reserve.end()) {
        for (const auto& region : search->second) {
            if (!region->IsRegistered()) {
                return region;
            }
        }
    }
    return {};
}

void VKGlobalCache::ReserveGlobalRegion(GlobalRegion region) {
    const auto [it, is_allocated] = reserve.try_emplace(static_cast<u64>(region->GetSizeInBytes()));
    it->second.push_back(std::move(region));
}

} // namespace Vulkan
