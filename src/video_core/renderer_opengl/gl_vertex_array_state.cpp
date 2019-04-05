// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <tuple>
#include <glad/glad.h>
#include "video_core/renderer_opengl/gl_vertex_array_state.h"

namespace OpenGL {

OGLVertexArrayState::OGLVertexArrayState() = default;

void OGLVertexArrayState::SetElementBuffer(GLuint element_buffer_) {
    if (element_buffer == element_buffer_) {
        return;
    }
    element_buffer = element_buffer_;
    glVertexArrayElementBuffer(handle, element_buffer_);
}

void OGLVertexArrayState::SetVertexBuffer(std::size_t index, GLuint buffer, GLintptr offset,
                                          GLsizei stride) {
    // Note: In the case a state-tracker is added here, keep in mind that "buffer" might be
    // invalidated in the future, resulting in potential geometry explosions.

    buffers[index] = buffer;
    offsets[index] = offset;
    strides[index] = stride;

    lowest_dirty = std::min(lowest_dirty, index);
    highest_dirty = std::max(highest_dirty, index);
}

void OGLVertexArrayState::SetDivisor(std::size_t index, GLuint divisor) {
    if (divisors[index] == divisor) {
        return;
    }
    divisors[index] = divisor;
    glVertexArrayBindingDivisor(handle, static_cast<GLuint>(index), divisor);
}

void OGLVertexArrayState::UpdateVertexBuffers() {
    if (lowest_dirty <= highest_dirty) {
        const std::size_t first{lowest_dirty};
        const std::size_t count{highest_dirty - lowest_dirty + 1};
        glVertexArrayVertexBuffers(handle, static_cast<GLuint>(first), static_cast<GLsizei>(count),
                                   buffers.data() + first, offsets.data() + first,
                                   strides.data() + first);
    }
    lowest_dirty = std::numeric_limits<std::size_t>::max();
    highest_dirty = std::numeric_limits<std::size_t>::min();
}

} // namespace OpenGL