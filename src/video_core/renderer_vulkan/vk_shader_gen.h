// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/shader_bytecode.h"
#include "video_core/renderer_vulkan/vk_shader_decompiler.h"

namespace Vulkan {
class VKDevice;
}

namespace Vulkan::VKShader {

constexpr std::size_t MAX_PROGRAM_CODE_LENGTH{0x1000};
using ProgramCode = std::vector<u64>;

struct ShaderSetup {
    explicit ShaderSetup(ProgramCode program_code) {
        program.code = std::move(program_code);
    }

    struct {
        ProgramCode code;
        ProgramCode code_b; // Used for dual vertex shaders
    } program;

    /// Used in scenarios where we have a dual vertex shaders
    void SetProgramB(ProgramCode&& program_b) {
        program.code_b = std::move(program_b);
        has_program_b = true;
    }

    bool IsDualProgram() const {
        return has_program_b;
    }

private:
    bool has_program_b{};
};

struct ProgramResult {
    std::vector<u8> code;
    ShaderEntries entries;
};

ProgramResult GenerateVertexShader(const VKDevice& device, const ShaderSetup& setup);

ProgramResult GenerateFragmentShader(const VKDevice& device,const ShaderSetup& setup);

} // namespace Vulkan::VKShader