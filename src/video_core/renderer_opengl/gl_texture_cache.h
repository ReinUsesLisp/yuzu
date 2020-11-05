// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <span>

#include <glad/glad.h>

#include "video_core/renderer_opengl/gl_resource_manager.h"
#include "video_core/renderer_opengl/util_shaders.h"
#include "video_core/texture_cache/texture_cache.h"

namespace OpenGL {

class Device;
class ProgramManager;
class StateTracker;

class Framebuffer;
class Image;
class ImageView;

using VideoCommon::ImageId;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewType;
using VideoCommon::NUM_RT;
using VideoCommon::RenderTargets;

class ImageBufferMap {
public:
    explicit ImageBufferMap(GLuint handle, u8* map, size_t size, OGLSync* sync);
    ~ImageBufferMap();

    GLuint Handle() const noexcept {
        return handle;
    }

    std::span<u8> Span() const noexcept {
        return span;
    }

private:
    std::span<u8> span;
    OGLSync* sync;
    GLuint handle;
};

struct FormatProperties {
    GLenum compatibility_class;
    bool compatibility_by_size;
    bool is_compressed;
};

class TextureCacheRuntime {
    friend Image;
    friend ImageView;

public:
    explicit TextureCacheRuntime(const Device& device, ProgramManager& program_manager,
                                 StateTracker& state_tracker);
    ~TextureCacheRuntime();

    ImageBufferMap MapUploadBuffer(size_t size);

    ImageBufferMap MapDownloadBuffer(size_t size);

    void CopyImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void ConvertImage(Framebuffer* dst, ImageView& dst_view, ImageView& src_view) {
        UNIMPLEMENTED();
    }

    void BlitFramebuffer(Framebuffer* dst, Framebuffer* src,
                         const Tegra::Engines::Fermi2D::Config& copy);

    void AccelerateImageUpload(Image& image, const ImageBufferMap& map, size_t buffer_offset,
                               std::span<const VideoCommon::SwizzleParameters> swizzles);

    void InsertUploadMemoryBarrier();

    FormatProperties FormatInfo(VideoCommon::ImageType type, GLenum internal_format) const;

private:
    struct StagingBuffers {
        explicit StagingBuffers(GLenum storage_flags_, GLenum map_flags_);
        ~StagingBuffers();

        ImageBufferMap RequestMap(size_t requested_size, bool insert_fence);

        size_t RequestBuffer(size_t requested_size);

        std::optional<size_t> FindBuffer(size_t requested_size);

        std::vector<OGLSync> syncs;
        std::vector<OGLBuffer> buffers;
        std::vector<u8*> maps;
        std::vector<size_t> sizes;
        GLenum storage_flags;
        GLenum map_flags;
    };

    const Device& device;
    StateTracker& state_tracker;
    UtilShaders util_shaders;

    std::array<std::unordered_map<GLenum, FormatProperties>, 3> format_properties;

    StagingBuffers upload_buffers{GL_MAP_WRITE_BIT, GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT};
    StagingBuffers download_buffers{GL_MAP_READ_BIT, GL_MAP_READ_BIT};

    OGLTexture null_image_1d;
    OGLTexture null_image_2d;
    OGLTexture null_image_3d;
    OGLTexture null_image_rect;
};

class Image : public VideoCommon::ImageBase {
    friend ImageView;

public:
    explicit Image(TextureCacheRuntime&, const VideoCommon::ImageInfo& info, GPUVAddr gpu_addr,
                   VAddr cpu_addr);

    void UploadMemory(const ImageBufferMap& map, size_t buffer_offset,
                      std::span<const VideoCommon::BufferImageCopy> copies);

    void UploadMemory(const ImageBufferMap& map, size_t buffer_offset,
                      std::span<const VideoCommon::BufferCopy> copies);

    void DownloadMemory(ImageBufferMap& map, size_t buffer_offset,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    GLuint Handle() const noexcept {
        return texture.handle;
    }

private:
    void CopyBufferToImage(const VideoCommon::BufferImageCopy& copy, size_t buffer_offset);

    void CopyImageToBuffer(const VideoCommon::BufferImageCopy& copy, size_t buffer_offset);

    OGLTexture texture;
    OGLTextureView store_view;
    OGLBuffer buffer;
    GLenum gl_internal_format = GL_NONE;
    GLenum gl_store_format = GL_NONE;
    GLenum gl_format = GL_NONE;
    GLenum gl_type = GL_NONE;
};

class ImageView : public VideoCommon::ImageViewBase {
    friend Image;

public:
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageViewInfo&, ImageId, Image&);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::NullImageParams&);

    [[nodiscard]] GLuint Handle(ImageViewType type) const noexcept {
        return views[static_cast<size_t>(type)].handle;
    }

    [[nodiscard]] GLuint DefaultHandle() const noexcept {
        return default_handle;
    }

    [[nodiscard]] GLenum Format() const noexcept {
        return internal_format;
    }

    [[nodiscard]] bool Is3D() const noexcept {
        return is_slice_view;
    }

private:
    void SetupView(Image& image, ImageViewType type, GLuint handle,
                   const VideoCommon::ImageViewInfo& info, VideoCommon::SubresourceRange range);

    std::array<OGLTextureView, VideoCommon::NUM_IMAGE_VIEW_TYPES> views;
    GLuint default_handle = 0;
    GLenum internal_format = GL_NONE;
    bool is_slice_view = false;
};

class ImageAlloc : public VideoCommon::ImageAllocBase {};

class Sampler {
public:
    explicit Sampler(TextureCacheRuntime&, const Tegra::Texture::TSCEntry&);

    GLuint Handle() const noexcept {
        return sampler.handle;
    }

private:
    OGLSampler sampler;
};

class Framebuffer {
public:
    explicit Framebuffer(TextureCacheRuntime&, const VideoCommon::SlotVector<Image>&,
                         std::span<ImageView*, NUM_RT> color_buffers, ImageView* depth_buffer,
                         std::array<u8, NUM_RT> draw_buffers, VideoCommon::Extent2D size);

    GLuint Handle() const noexcept {
        return framebuffer.handle;
    }

private:
    OGLFramebuffer framebuffer;
};

struct TextureCacheParams {
    static constexpr bool ENABLE_VALIDATION = true;
    static constexpr bool FRAMEBUFFER_BLITS = true;

    using Runtime = OpenGL::TextureCacheRuntime;
    using Image = OpenGL::Image;
    using ImageAlloc = OpenGL::ImageAlloc;
    using ImageView = OpenGL::ImageView;
    using Sampler = OpenGL::Sampler;
    using Framebuffer = OpenGL::Framebuffer;
};

using TextureCache = VideoCommon::TextureCache<TextureCacheParams>;

} // namespace OpenGL
