// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/memory_manager.h"

#pragma optimize("", off)

namespace Tegra::Engines {

KeplerCompute::KeplerCompute(MemoryManager& memory_manager) : memory_manager{memory_manager} {}

KeplerCompute::~KeplerCompute() = default;

void KeplerCompute::CallMethod(const GPU::MethodCall& method_call) {
    ASSERT_MSG(method_call.method < Regs::NUM_REGS,
               "Invalid KeplerCompute register, increase the size of the Regs structure");

    regs.reg_array[method_call.method] = method_call.argument;

    LOG_CRITICAL(HW_GPU, "{:x} {:x}", method_call.method * 4, method_call.argument);

    switch (method_call.method) {
    case KEPLER_COMPUTE_REG_INDEX(dest_address.high_dest_address):
    case KEPLER_COMPUTE_REG_INDEX(dest_address.low_dest_address):
        upload_cursor = 0;
        break;
    case KEPLER_COMPUTE_REG_INDEX(upload_data):
        memory_manager.Write<u32>(regs.dest_address.Address() + upload_cursor,
                                  method_call.argument);
        upload_cursor += sizeof(u32);
        break;
    case KEPLER_COMPUTE_REG_INDEX(launch): {
        // Abort execution since compute shaders can be used to alter game memory (e.g. CUDA
        // kernels)
        const auto cpu_addr = memory_manager.GpuToCpuAddress(regs.code_address.Address());
        const auto pointer = memory_manager.GetPointer(regs.code_address.Address());
        UNREACHABLE_MSG("Compute shaders are not implemented");
        break;
    }
    default:
        break;
    }
}

} // namespace Tegra::Engines
