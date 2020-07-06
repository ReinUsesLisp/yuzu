// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"
#include "video_core/host_buffer_type.h"
#include "video_core/renderer_opengl/gl_resource_manager.h"

namespace OpenGL {

struct Buffer {
    OGLBuffer buffer;
    u8* mapped_address{};

    GLuint Handle() const noexcept {
        return buffer.handle;
    }

    u8* Map([[maybe_unused]] u64 size, u64 offset = 0) const {
        return mapped_address + offset;
    }
};

class StagingBufferPool {
public:
    using Buffer = ::OpenGL::Buffer;

    ~StagingBufferPool();

    Buffer& GetUnusedBuffer(size_t size, VideoCommon::HostBufferType type);

    void TickFrame();

private:
    struct StagingBuffer {
        Buffer buffer;
        u64 last_usage = 0;
    };

    static constexpr size_t NUM_LEVELS = sizeof(size_t) * CHAR_BIT;
    using StagingBuffersCache = std::array<std::vector<StagingBuffer>, NUM_LEVELS>;

    Buffer* TryGetReservedBuffer(size_t size, VideoCommon::HostBufferType type);

    Buffer& CreateStagingBuffer(size_t size, VideoCommon::HostBufferType type);

    void DeleteUnusedEntries();

    StagingBuffersCache& Cache(VideoCommon::HostBufferType type);

    std::array<StagingBuffersCache, VideoCommon::NUM_HOST_BUFFER_TYPES> caches;

    u64 epoch = 0;
    size_t deletion_index = 0;
};

} // namespace OpenGL
