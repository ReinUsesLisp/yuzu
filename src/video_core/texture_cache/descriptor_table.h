// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <bit>
#include <memory>
#include <optional>
#include <span>

#include "common/common_types.h"
#include "common/logging/log.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

template <typename Descriptor>
class DescriptorTable {
public:
    explicit DescriptorTable(VideoCore::RasterizerInterface& rasterizer_,
                             Tegra::MemoryManager& gpu_memory_)
        : rasterizer{rasterizer_}, gpu_memory{gpu_memory_} {}

    void WriteMemory(VAddr addr, size_t size) {
        const VAddr overlap_begin = addr;
        const VAddr overlap_end = addr + size;
        is_modified |= cpu_addr_begin < overlap_end && overlap_begin < cpu_addr_end;
    }

    bool Synchronize(GPUVAddr gpu_addr, size_t limit) {
        [[likely]] if (!is_modified && current_gpu_addr == gpu_addr && current_limit == limit) {
            return false;
        }
        Refresh(gpu_addr, limit);
        return true;
    }

    [[nodiscard]] std::span<const Descriptor> Descriptors() const noexcept {
        return std::span(descriptors.get(), num_descriptors);
    }

private:
    void Refresh(GPUVAddr gpu_addr, size_t limit) {
        current_gpu_addr = gpu_addr;
        current_limit = limit;

        if (SizeBytes() == 0) {
            Unregister();
        }
        cpu_addr_begin = 0;
        cpu_addr_end = 0;
        is_modified = false;

        if (gpu_addr == 0) {
            return;
        }
        const std::optional<VAddr> cpu_addr = gpu_memory.GpuToCpuAddress(gpu_addr);
        if (!cpu_addr) {
            LOG_ERROR(HW_GPU, "Invalid gpu_addr=0x{:x}", gpu_addr);
            return;
        }
        num_descriptors = limit + 1;
        const size_t size_bytes = num_descriptors * sizeof(Descriptor);
        cpu_addr_begin = *cpu_addr;
        cpu_addr_end = *cpu_addr + size_bytes;
        if (num_descriptors > descriptors_capacity) {
            descriptors_capacity = std::bit_ceil(num_descriptors);
            descriptors = std::make_unique<Descriptor[]>(descriptors_capacity);
        }
        gpu_memory.ReadBlockUnsafe(gpu_addr, descriptors.get(), size_bytes);
        Register();
    }

    void Register() {
        rasterizer.UpdatePagesCachedCount(cpu_addr_begin, SizeBytes(), 1);
    }

    void Unregister() {
        rasterizer.UpdatePagesCachedCount(cpu_addr_begin, SizeBytes(), -1);
    }

    [[nodiscard]] size_t SizeBytes() const noexcept {
        return cpu_addr_end - cpu_addr_begin;
    }

    VideoCore::RasterizerInterface& rasterizer;
    Tegra::MemoryManager& gpu_memory;
    VAddr cpu_addr_begin{};
    VAddr cpu_addr_end{};
    GPUVAddr current_gpu_addr{};
    size_t current_limit{};
    std::unique_ptr<Descriptor[]> descriptors;
    size_t num_descriptors{};
    size_t descriptors_capacity{};
    bool is_modified{};
};

} // namespace VideoCommon
