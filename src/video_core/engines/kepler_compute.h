// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "video_core/gpu.h"

namespace Tegra {
class MemoryManager;
}

namespace Tegra::Engines {

#define KEPLER_COMPUTE_REG_INDEX(field_name)                                                       \
    (offsetof(Tegra::Engines::KeplerCompute::Regs, field_name) / sizeof(u32))

class KeplerCompute final {
public:
    explicit KeplerCompute(MemoryManager& memory_manager);
    ~KeplerCompute();

    static constexpr std::size_t NumConstBuffers = 8;

    struct Regs {
        static constexpr std::size_t NUM_REGS = 0xCF8;

        union {
            struct {
                INSERT_PADDING_WORDS(0x62);

                struct {
                    u32 high_dest_address;
                    u32 low_dest_address;

                    GPUVAddr Address() const {
                        return (static_cast<GPUVAddr>(high_dest_address) << 32) |
                               static_cast<GPUVAddr>(low_dest_address);
                    }
                } dest_address;

                INSERT_PADDING_WORDS(9);

                u32 upload_data;

                INSERT_PADDING_WORDS(0x41);

                u32 launch;

                INSERT_PADDING_WORDS(0x4D2);

                struct {
                    u32 high_code_address;
                    u32 low_code_address;

                    GPUVAddr Address() const {
                        return (static_cast<GPUVAddr>(high_code_address) << 32) |
                               static_cast<GPUVAddr>(low_code_address);
                    }
                } code_address;

                INSERT_PADDING_WORDS(0x774);
            };
            std::array<u32, NUM_REGS> reg_array;
        };
    } regs{};
    static_assert(sizeof(Regs) == Regs::NUM_REGS * sizeof(u32),
                  "KeplerCompute Regs has wrong size");

    /// Write the value to the register identified by method.
    void CallMethod(const GPU::MethodCall& method_call);

private:
    MemoryManager& memory_manager;

    std::size_t upload_cursor{};
};

#define ASSERT_REG_POSITION(field_name, position)                                                  \
    static_assert(offsetof(KeplerCompute::Regs, field_name) == position * 4,                       \
                  "Field " #field_name " has invalid position");

ASSERT_REG_POSITION(dest_address, 0x62);
ASSERT_REG_POSITION(upload_data, 0x6D);
ASSERT_REG_POSITION(launch, 0xAF);
ASSERT_REG_POSITION(code_address, 0x582);

#undef ASSERT_REG_POSITION

} // namespace Tegra::Engines
