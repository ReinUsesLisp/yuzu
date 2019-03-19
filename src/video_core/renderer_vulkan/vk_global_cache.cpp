// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "common/common_types.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_global_cache.h"
#include "video_core/renderer_vulkan/vk_shader_decompiler.h"

namespace Vulkan {

CachedGlobalRegion::CachedGlobalRegion(const VKDevice& device, VKMemoryManager& memory_manager,
                                       VAddr cpu_addr, u8* host_ptr, u64 size_)
    : device{device}, cpu_addr{cpu_addr}, host_ptr{host_ptr}, RasterizerCacheObject{host_ptr} {
    const auto max_size = device.GetMaxStorageBufferRange();
    size = size_;
    if (size > max_size) {
        size = max_size;
        LOG_ERROR(Render_Vulkan,
                  "Global buffer size 0x{:x} is bigger than the maximum supported size by the "
                  "current physical device 0x{:x}",
                  size, max_size);
    }

    const vk::BufferCreateInfo buffer_ci({}, static_cast<vk::DeviceSize>(size),
                                         vk::BufferUsageFlagBits::eStorageBuffer,
                                         vk::SharingMode::eExclusive, 0, nullptr);
    const auto& dld = device.GetDispatchLoader();
    const auto dev = device.GetLogical();
    buffer = dev.createBufferUnique(buffer_ci, nullptr, dld);
    commit = memory_manager.Commit(*buffer, true);
    memory = commit->GetData();
}

CachedGlobalRegion::~CachedGlobalRegion() = default;

void CachedGlobalRegion::Upload(u64 size_) {
    UNIMPLEMENTED_IF(size != size_);
    std::memcpy(memory, host_ptr, static_cast<std::size_t>(size));
}

void CachedGlobalRegion::Flush() {
    UNIMPLEMENTED();
}

VKGlobalCache::VKGlobalCache(Core::System& system, VideoCore::RasterizerInterface& rasterizer,
                             const VKDevice& device, VKMemoryManager& memory_manager)
    : RasterizerCache{rasterizer}, system{system}, device{device}, memory_manager{memory_manager} {}

VKGlobalCache::~VKGlobalCache() = default;

GlobalRegion VKGlobalCache::TryGetReservedGlobalRegion(VAddr addr, u64 size) const {
    if (const auto search{reserve.find(addr)}; search != reserve.end())
        return search->second;
    return {};
}

GlobalRegion VKGlobalCache::GetUncachedGlobalRegion(VAddr addr, u8* host_ptr, u64 size) {
    GlobalRegion region{TryGetReservedGlobalRegion(addr, size)};
    if (!region) {
        region = std::make_shared<CachedGlobalRegion>(device, memory_manager, addr, host_ptr, size);
        ReserveGlobalRegion(region);
    }
    region->Upload(size);
    return region;
}

void VKGlobalCache::ReserveGlobalRegion(GlobalRegion region) {
    reserve.insert_or_assign(region->GetCacheAddr(), std::move(region));
}

GlobalRegion VKGlobalCache::GetGlobalRegion(const VKShader::GlobalBufferEntry& descriptor,
                                            Tegra::Engines::Maxwell3D::Regs::ShaderStage stage) {
    auto& gpu{system.GPU()};
    auto& emu_memory_manager{gpu.MemoryManager()};

    const auto& cbufs = gpu.Maxwell3D().state.shader_stages[static_cast<std::size_t>(stage)];
    const auto addr =
        cbufs.const_buffers[descriptor.GetCbufIndex()].address + descriptor.GetCbufOffset();
    const auto actual_addr{emu_memory_manager.Read64(addr)};
    const auto size{emu_memory_manager.Read64(addr + 8)};

    const auto host_ptr{emu_memory_manager.GetPointer(actual_addr)};
    GlobalRegion region = TryGet(host_ptr);
    if (!region) {
        region = GetUncachedGlobalRegion(actual_addr, host_ptr, size);
        Register(region);
    }
    return region;
}

} // namespace Vulkan
