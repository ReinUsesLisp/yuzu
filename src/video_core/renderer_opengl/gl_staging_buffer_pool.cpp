// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>

#include "common/bit_util.h"
#include "common/common_types.h"
#include "video_core/renderer_opengl/gl_staging_buffer_pool.h"

namespace OpenGL {

using VideoCommon::HostBufferType;

static constexpr size_t TICKS_TO_REUSE = 5;
static constexpr size_t TICKS_TO_DESTROY = 240;

StagingBufferPool::~StagingBufferPool() = default;

Buffer& StagingBufferPool::GetUnusedBuffer(size_t size, HostBufferType type) {
    if (Buffer* const buffer = TryGetReservedBuffer(size, type)) {
        return *buffer;
    }
    return CreateStagingBuffer(size, type);
}

void StagingBufferPool::TickFrame() {
    DeleteUnusedEntries();
    ++epoch;
}

Buffer* StagingBufferPool::TryGetReservedBuffer(size_t size, HostBufferType type) {
    for (auto& entry : Cache(type)[Common::Log2Ceil64(size)]) {
        if (epoch - entry.last_usage < TICKS_TO_REUSE) {
            continue;
        }
        entry.last_usage = epoch;
        return &entry.buffer;
    }
    return nullptr;
}

Buffer& StagingBufferPool::CreateStagingBuffer(size_t size, HostBufferType type) {
    const u32 log2 = Common::Log2Ceil64(size);
    auto& entry = Cache(type)[log2].emplace_back();
    entry.last_usage = epoch;

    GLbitfield flags = 0;
    GLbitfield map_flags = 0;
    switch (type) {
    case HostBufferType::Upload:
        flags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_CLIENT_STORAGE_BIT;
        map_flags = GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_PERSISTENT_BIT;
        break;
    case HostBufferType::Download:
        flags = GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_CLIENT_STORAGE_BIT;
        map_flags = GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT;
        break;
    case HostBufferType::DeviceLocal:
        break;
    }

    const GLsizeiptr gl_size = 1ULL << log2;

    Buffer& buffer = entry.buffer;
    buffer.buffer.Create();
    glNamedBufferStorage(buffer.Handle(), gl_size, nullptr, flags);
    if (map_flags != 0) {
        buffer.mapped_address =
            static_cast<u8*>(glMapNamedBufferRange(buffer.Handle(), 0, gl_size, map_flags));
    }
    return buffer;
}

void StagingBufferPool::DeleteUnusedEntries() {
    for (auto& cache : caches) {
        auto& entries = cache[deletion_index];
        for (auto it = entries.begin(); it != entries.end();) {
            StagingBuffer& entry = *it;
            if (epoch - entry.last_usage < TICKS_TO_DESTROY) {
                ++it;
            } else {
                it = entries.erase(it);
            }
        }
    }

    deletion_index = (deletion_index + 1) % NUM_LEVELS;
}

StagingBufferPool::StagingBuffersCache& StagingBufferPool::Cache(HostBufferType type) {
    return caches[static_cast<size_t>(type)];
}

} // namespace OpenGL
