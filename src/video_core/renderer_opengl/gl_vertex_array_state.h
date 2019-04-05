// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <glad/glad.h>
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_opengl/gl_resource_manager.h"

namespace OpenGL {

class OGLVertexArrayState final : public OGLVertexArray {
public:
    OGLVertexArrayState();

    void SetElementBuffer(GLuint element_buffer_);

    void SetVertexBuffer(std::size_t index, GLuint buffer, GLintptr offset, GLsizei stride);

    void SetDivisor(std::size_t index, GLuint divisor);

    void UpdateVertexBuffers();

private:
    static constexpr std::size_t NumAttributes =
        Tegra::Engines::Maxwell3D::Regs::NumVertexAttributes;

    GLuint element_buffer{};

    std::size_t lowest_dirty{std::numeric_limits<std::size_t>::max()};
    std::size_t highest_dirty{std::numeric_limits<std::size_t>::min()};
    std::array<GLuint, NumAttributes> buffers{};
    std::array<GLintptr, NumAttributes> offsets{};
    std::array<GLsizei, NumAttributes> strides{};

    std::array<GLuint, NumAttributes> divisors{};
};

} // namespace OpenGL
