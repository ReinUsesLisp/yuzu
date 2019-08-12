#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include "common/common_types.h"
#include "video_core/renderer_opengl/gl_device.h"
#include "video_core/renderer_opengl/gl_shader_decompiler.h"
#include "video_core/shader/shader_ir.h"

using namespace OpenGL::GLShader;
using namespace VideoCommon::Shader;

using OpenGL::ProgramType;

static ProgramType InputStage(const std::string& source) {
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
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [vs|fs|gs|tcs|tes] <bytecode> <offset:0>"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    const std::string suffix{argv[1]};
    const ProgramType stage{InputStage(suffix)};
    const ProgramCode bytecode{ReadBytecode(argv[2])};
    const u32 main_offset{argc >= 4 ? static_cast<u32>(atoi(argv[3])) : 0};

    const ShaderIR ir(bytecode, main_offset, bytecode.size() * sizeof(u64));
    std::cout << Decompile(OpenGL::Device{nullptr}, ir, stage, suffix).first << std::endl;

    return 0;
}