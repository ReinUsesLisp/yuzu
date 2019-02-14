// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdio>

#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/vk_shader_decompiler.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"
#include "video_core/shader/shader_ir.h"

namespace Vulkan::VKShader {

static constexpr u32 PROGRAM_OFFSET{10};

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using namespace VideoCommon::Shader;

ProgramResult GenerateVertexShader(const ShaderSetup& setup) {
    // TODO(Rodrigo): Add vertex stage B.
    const ShaderIR ir(setup.program.code, PROGRAM_OFFSET);

    const auto [module, entries] = Decompile(ir, Maxwell::ShaderStage::Vertex);
    const auto vertex_entry = entries.entry_function;
    const auto main = module->Emit(
        module->OpFunction(module->OpTypeVoid(), {}, module->OpTypeFunction(module->OpTypeVoid())));
    module->Emit(module->OpLabel());
    module->Emit(module->OpFunctionCall(module->OpTypeVoid(), vertex_entry));
    module->Emit(module->OpReturn());
    module->Emit(module->OpFunctionEnd());
    module->AddEntryPoint(spv::ExecutionModel::Vertex, main, "main", entries.interfaces);

    const auto code = module->Assemble();
    FILE* out = fopen("D:\\vertex.spv", "wb");
    fwrite(code.data(), 1, code.size(), out);
    fclose(out);
    return {code, entries};
}

ProgramResult GenerateFragmentShader(const ShaderSetup& setup) {
    const ShaderIR ir(setup.program.code, PROGRAM_OFFSET);

    const auto [module, entries] = Decompile(ir, Maxwell::ShaderStage::Fragment);
    const auto fragment_entry = entries.entry_function;
    const auto main = module->Emit(
        module->OpFunction(module->OpTypeVoid(), {}, module->OpTypeFunction(module->OpTypeVoid())));
    module->Emit(module->OpLabel());
    module->Emit(module->OpFunctionCall(module->OpTypeVoid(), fragment_entry));
    module->Emit(module->OpReturn());
    module->Emit(module->OpFunctionEnd());
    module->AddEntryPoint(spv::ExecutionModel::Fragment, main, "main", entries.interfaces);
    module->AddExecutionMode(main, spv::ExecutionMode::OriginUpperLeft);

    const auto code = module->Assemble();
    FILE* out = fopen("D:\\fragment.spv", "wb");
    fwrite(code.data(), 1, code.size(), out);
    fclose(out);

    return {code, entries};
}

} // namespace Vulkan::VKShader