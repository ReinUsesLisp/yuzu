// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once


#include "common/common_types.h"
#include "common/logging/log.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

template <typename Descriptor>
class DescriptorTable {
public:
    explicit DescriptorTable(Tegra::MemoryManager& gpu_memory_) : gpu_memory{gpu_memory_} {}

    void SetState(GPUVAddr gpu_addr, size_t limit) {
        Refresh(gpu_addr, limit);
    }

    [[nodiscard]] Descriptor Read(size_t index) {
        ASSERT(index <= current_limit);
        const GPUVAddr gpu_addr = current_gpu_addr + index * sizeof(Descriptor);
        Descriptor descriptor;
        gpu_memory.ReadBlockUnsafe(gpu_addr, &descriptor, sizeof(descriptor));
        return descriptor;
    }

private:
    void Refresh(GPUVAddr gpu_addr, size_t limit) {
        current_gpu_addr = gpu_addr;
        current_limit = limit;
    }

    Tegra::MemoryManager& gpu_memory;
    GPUVAddr current_gpu_addr{};
    size_t current_limit{};
};

} // namespace VideoCommon
