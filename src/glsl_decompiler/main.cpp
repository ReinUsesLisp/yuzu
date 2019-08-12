#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include "common/common_types.h"
#include "video_core/renderer_opengl/gl_device.h"
#include "video_core/renderer_opengl/gl_shader_decompiler.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_shader_decompiler.h"
#include "video_core/shader/shader_ir.h"

using OpenGL::ProgramType;
using VideoCommon::Shader::MAX_PROGRAM_LENGTH;
using VideoCommon::Shader::ProgramCode;
using VideoCommon::Shader::ShaderIR;

using ShaderStage = Tegra::Engines::Maxwell3D::Regs::ShaderStage;

static ProgramType InputProgram(std::string_view source) {
    if (source == "vs") {
        return ProgramType::VertexB;
    } else if (source == "fs") {
        return ProgramType::Fragment;
    } else if (source == "gs") {
        return ProgramType::Geometry;
    } else if (source == "tcs") {
        return ProgramType::TessellationControl;
    } else if (source == "tes") {
        return ProgramType::TessellationEval;
    } else if (source == "cs") {
        return ProgramType::Compute;
    } else {
        std::cerr << "Unknown shader stage \"" << source << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}

static ShaderStage InputStage(std::string_view source) {
    if (source == "vs") {
        return ShaderStage::Vertex;
    } else if (source == "fs") {
        return ShaderStage::Fragment;
    } else if (source == "gs") {
        return ShaderStage::Geometry;
    } else if (source == "tcs") {
        return ShaderStage::TesselationControl;
    } else if (source == "tes") {
        return ShaderStage::TesselationEval;
    } else {
        std::cerr << "Unknown shader stage \"" << source << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}

static ProgramCode ReadBytecode(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "File not found: " << path << std::endl;
        exit(EXIT_FAILURE);
    }
    const std::streamsize size{file.tellg()};
    file.seekg(0, std::ios::beg);

    ProgramCode bytecode(MAX_PROGRAM_LENGTH);
    file.read(reinterpret_cast<char*>(bytecode.data()), size);
    file.close();

    return bytecode;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " [glsl|spirv] [vs|fs|gs|tcs|tes] <bytecode> <offset:0>" << std::endl;
        exit(EXIT_FAILURE);
    }

    const std::string_view suffix{argv[2]};
    const ProgramType stage{};
    const ProgramCode bytecode{ReadBytecode(argv[3])};
    const u32 main_offset{argc >= 5 ? static_cast<u32>(atoi(argv[4])) : 0};

    const ShaderIR ir(bytecode, main_offset, bytecode.size() * sizeof(u64));

    const std::string_view target = argv[1];
    if (target == "glsl") {
        std::cout << OpenGL::GLShader::Decompile(OpenGL::Device{nullptr}, ir, InputProgram(suffix),
                                                 std::string{suffix})
                         .first
                  << std::endl;
    } else if (target == "spirv") {
        const auto result =
            Vulkan::VKShader::Decompile(Vulkan::VKDevice{nullptr}, ir, InputStage(suffix));
        result.first->AddCapability(spv::Capability::Linkage);
        const auto code = result.first->Assemble();
        std::ofstream file("output.spv", std::ios::binary | std::ios::ate);
        file.write(reinterpret_cast<const char*>(code.data()), code.size());
        file.close();

        system("spirv-dis output.spv");
        system("spirv-val output.spv");
    }

    return 0;
}