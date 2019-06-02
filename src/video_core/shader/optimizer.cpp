// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/shader/shader_ir.h"

namespace VideoCommon::Shader {

void ShaderIR::Optimize() {
    FlowStackRemover();
}

} // namespace VideoCommon::Shader
