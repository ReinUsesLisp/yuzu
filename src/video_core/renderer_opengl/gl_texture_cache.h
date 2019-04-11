// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>

#include "common/common_types.h"
#include "video_core/texture_cache.h"

namespace OpenGL {

using VideoCommon::SurfaceParams;
using VideoCommon::ViewKey;
using VideoCore::Surface::ComponentType;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::SurfaceTarget;
using VideoCore::Surface::SurfaceType;

struct EmptyStruct {};

class CachedView;
class CachedSurface;

using TextureCacheBase = VideoCommon::TextureCache<CachedSurface, CachedView, EmptyStruct>;

class CachedView final {
public:
    explicit CachedView(CachedSurface& surface, ViewKey key);
    ~CachedView();

    GLuint GetTexture(bool is_array);

private:
    OGLTexture CreateTexture(bool is_array) const;

    CachedSurface& surface;
    const ViewKey key;
    const SurfaceParams params;

    OGLTexture normal_texture;
    OGLTexture arrayed_texture;
};

class CachedSurface final : public VideoCommon::SurfaceBase<CachedView, EmptyStruct> {
    friend CachedView;

public:
    explicit CachedSurface(const SurfaceParams& params);
    ~CachedSurface();

    void LoadBuffer();

    EmptyStruct FlushBuffer(EmptyStruct);

    EmptyStruct UploadTexture(EmptyStruct);

protected:
    std::unique_ptr<CachedView> CreateView(const ViewKey& view_key);

private:
    void UploadTextureMipmap(u32 level);

    GLenum internal_format{};
    GLenum format{};
    GLenum type{};
    bool is_compressed{};

    OGLTexture texture;

    std::vector<u8> staging_buffer;
    u8* host_ptr{};
};

class TextureCacheOpenGL final : TextureCacheBase {
public:
    explicit TextureCacheOpenGL(Core::System& system, VideoCore::RasterizerInterface& rasterizer);
    ~TextureCacheOpenGL();

protected:
    std::tuple<CachedView*, EmptyStruct> TryFastGetSurfaceView(
        EmptyStruct, VAddr cpu_addr, u8* host_ptr, const SurfaceParams& params,
        bool preserve_contents, const std::vector<CachedSurface*>& overlaps);

    std::unique_ptr<CachedSurface> CreateSurface(const SurfaceParams& params);
};

} // namespace OpenGL
