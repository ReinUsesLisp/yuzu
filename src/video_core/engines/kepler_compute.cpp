// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/logging/log.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/memory_manager.h"

namespace Tegra::Engines {

KeplerCompute::KeplerCompute(MemoryManager& memory_manager) : memory_manager{memory_manager} {}

KeplerCompute::~KeplerCompute() = default;

void KeplerCompute::CallMethod(const GPU::MethodCall& method_call) {
    ASSERT_MSG(method_call.method < Regs::NUM_REGS,
               "Invalid KeplerCompute register, increase the size of the Regs structure");

    regs.reg_array[method_call.method] = method_call.argument;

    switch (method_call.method) {
    case KEPLER_COMPUTE_REG_INDEX(launch): {
        FILE* file = fopen("D:\\compute.bin", "wb");

        const auto gpu_desc_addr = regs.launch_address.LaunchAddress();
        const auto cpu_desc_addr = memory_manager.GpuToCpuAddress(gpu_desc_addr);
        ASSERT(cpu_desc_addr);
        const auto desc = reinterpret_cast<const LaunchDesc*>(Memory::GetPointer(*cpu_desc_addr));
        UNIMPLEMENTED_IF(desc->prog_start != 0);

        const auto code_gpu_addr = regs.code_address.CodeAddress();
        const auto code_cpu_addr = memory_manager.GpuToCpuAddress(code_gpu_addr);
        ASSERT(code_cpu_addr);
        const auto ptr = Memory::GetPointer(*code_cpu_addr);
        fwrite(Memory::GetPointer(*code_cpu_addr), 1, 0x1000, file);
        fclose(file);

        LOG_CRITICAL(HW_GPU, "Compute shaders are not implemented");
        UNREACHABLE();
        break;
    }
    default:
        break;
    }
}

} // namespace Tegra::Engines
