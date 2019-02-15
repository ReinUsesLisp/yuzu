// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <bitset>
#include "common/assert.h"
#include "common/bit_field.h"
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "video_core/gpu.h"
#include "video_core/memory_manager.h"

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
                INSERT_PADDING_WORDS(0xAD);

                struct {
                    u32 launch_address;

                    GPUVAddr LaunchAddress() {
                        return static_cast<u64>(launch_address) << 8;
                    }
                } launch_address;

                INSERT_PADDING_WORDS(1);

                u32 launch;

                INSERT_PADDING_WORDS(0x4D2);

                struct {
                    u32 code_address_high;
                    u32 code_address_low;

                    GPUVAddr CodeAddress() const {
                        return static_cast<GPUVAddr>(
                            (static_cast<GPUVAddr>(code_address_high) << 32) | code_address_low);
                    }
                } code_address;

                INSERT_PADDING_WORDS(0x774);
            };
            std::array<u32, NUM_REGS> reg_array;
        };
    } regs{};
    static_assert(sizeof(Regs) == Regs::NUM_REGS * sizeof(u32),
                  "KeplerCompute Regs has wrong size");

    struct LaunchDesc {
        INSERT_PADDING_WORDS(6);

        u32 prog_start;

        INSERT_PADDING_WORDS(5);

        BitField<0, 31, u32> griddim_x;
        union {
            BitField<0, 16, u32> griddim_y;
            BitField<16, 16, u32> griddim_z;
        };

        INSERT_PADDING_WORDS(3);

        struct {
            BitField<0, 16, u32> shared_alloc;

            u16 SharedAlloc() const {
                ASSERT_MSG((shared_alloc.Value() & 0xff) == 0, "Unaligned shared alloc");
                return static_cast<u16>(shared_alloc);
            }
        } shared_alloc;

        BitField<16, 16, u32> blockdim_x;
        union {
            BitField<0, 16, u32> blockdim_y;
            BitField<16, 16, u32> blockdim_z;
        };

        union {
            std::bitset<8> bitset;
            u32 raw;
        } cb_valid;

        INSERT_PADDING_WORDS(8);

        struct ConstBuffer {
            u32 address_low;
            union {
                BitField<0, 8, u32> address_high;
                BitField<15, 17, u32> size;
            };

            GPUVAddr GetAddress() const {
                return static_cast<GPUVAddr>(address_high) << 32 | address_low;
            }

            u64 GetSize() const {
                return static_cast<u64>(size);
            }
        };
        std::array<ConstBuffer, NumConstBuffers> cb_addresses;

        INSERT_PADDING_WORDS(19);
    };
    static_assert(sizeof(LaunchDesc) == 0x100, "MaxwellCompute launch descriptor has wrong size");

    MemoryManager& memory_manager;

    /// Write the value to the register identified by method.
    void CallMethod(const GPU::MethodCall& method_call);
};

#define ASSERT_REG_POSITION(field_name, position)                                                  \
    static_assert(offsetof(KeplerCompute::Regs, field_name) == position * 4,                       \
                  "Field " #field_name " has invalid position")

ASSERT_REG_POSITION(launch, 0xAF);

#undef ASSERT_REG_POSITION

#define ASSERT_DESC_POSITION(field_name, position)                                                 \
    static_assert(offsetof(KeplerCompute::LaunchDesc, field_name) == position * 4,                 \
                  "Field " #field_name " has invalid position")

ASSERT_DESC_POSITION(prog_start, 6);
ASSERT_DESC_POSITION(griddim_x, 12);
ASSERT_DESC_POSITION(griddim_y, 13);
ASSERT_DESC_POSITION(griddim_z, 13);
ASSERT_DESC_POSITION(shared_alloc, 17);
ASSERT_DESC_POSITION(blockdim_x, 18);
ASSERT_DESC_POSITION(blockdim_y, 19);
ASSERT_DESC_POSITION(blockdim_z, 19);
ASSERT_DESC_POSITION(cb_valid, 20);
ASSERT_DESC_POSITION(cb_addresses, 29);

#undef ASSERT_DESC_POSITION

} // namespace Tegra::Engines
