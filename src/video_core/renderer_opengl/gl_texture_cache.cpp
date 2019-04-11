// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "common/common_types.h"
#include "common/scope_exit.h"
#include "video_core/morton.h"
#include "video_core/renderer_opengl/gl_resource_manager.h"
#include "video_core/renderer_opengl/gl_texture_cache.h"
#include "video_core/textures/convert.h"
#include "video_core/textures/texture.h"

namespace OpenGL {

using Tegra::Texture::ConvertFromGuestToHost;
using VideoCore::MortonSwizzleMode;

namespace {

struct FormatTuple {
    GLint internal_format;
    GLenum format;
    GLenum type;
    ComponentType component_type;
    bool compressed;
};

constexpr std::array<FormatTuple, VideoCore::Surface::MaxPixelFormat> tex_format_tuples = {{
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, ComponentType::UNorm, false}, // ABGR8U
    {GL_RGBA8, GL_RGBA, GL_BYTE, ComponentType::SNorm, false},                     // ABGR8S
    {GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, ComponentType::UInt, false},   // ABGR8UI
    {GL_RGB8, GL_RGB, GL_UNSIGNED_SHORT_5_6_5_REV, ComponentType::UNorm, false},   // B5G6R5U
    {GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, ComponentType::UNorm,
     false}, // A2B10G10R10U
    {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_1_5_5_5_REV, ComponentType::UNorm, false}, // A1B5G5R5U
    {GL_R8, GL_RED, GL_UNSIGNED_BYTE, ComponentType::UNorm, false},                    // R8U
    {GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE, ComponentType::UInt, false},           // R8UI
    {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, ComponentType::Float, false},                 // RGBA16F
    {GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT, ComponentType::UNorm, false},              // RGBA16U
    {GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, ComponentType::UInt, false},     // RGBA16UI
    {GL_R11F_G11F_B10F, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV, ComponentType::Float,
     false},                                                                     // R11FG11FB10F
    {GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT, ComponentType::UInt, false}, // RGBA32UI
    {GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, ComponentType::UNorm,
     true}, // DXT1
    {GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, ComponentType::UNorm,
     true}, // DXT23
    {GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, ComponentType::UNorm,
     true},                                                                                 // DXT45
    {GL_COMPRESSED_RED_RGTC1, GL_RED, GL_UNSIGNED_INT_8_8_8_8, ComponentType::UNorm, true}, // DXN1
    {GL_COMPRESSED_RG_RGTC2, GL_RG, GL_UNSIGNED_INT_8_8_8_8, ComponentType::UNorm,
     true},                                                                     // DXN2UNORM
    {GL_COMPRESSED_SIGNED_RG_RGTC2, GL_RG, GL_INT, ComponentType::SNorm, true}, // DXN2SNORM
    {GL_COMPRESSED_RGBA_BPTC_UNORM, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, ComponentType::UNorm,
     true}, // BC7U
    {GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, GL_RGB, GL_UNSIGNED_INT_8_8_8_8, ComponentType::Float,
     true}, // BC6H_UF16
    {GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT, GL_RGB, GL_UNSIGNED_INT_8_8_8_8, ComponentType::Float,
     true},                                                                    // BC6H_SF16
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false},        // ASTC_2D_4X4
    {GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false},        // BGRA8
    {GL_RGBA32F, GL_RGBA, GL_FLOAT, ComponentType::Float, false},              // RGBA32F
    {GL_RG32F, GL_RG, GL_FLOAT, ComponentType::Float, false},                  // RG32F
    {GL_R32F, GL_RED, GL_FLOAT, ComponentType::Float, false},                  // R32F
    {GL_R16F, GL_RED, GL_HALF_FLOAT, ComponentType::Float, false},             // R16F
    {GL_R16, GL_RED, GL_UNSIGNED_SHORT, ComponentType::UNorm, false},          // R16U
    {GL_R16_SNORM, GL_RED, GL_SHORT, ComponentType::SNorm, false},             // R16S
    {GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT, ComponentType::UInt, false}, // R16UI
    {GL_R16I, GL_RED_INTEGER, GL_SHORT, ComponentType::SInt, false},           // R16I
    {GL_RG16, GL_RG, GL_UNSIGNED_SHORT, ComponentType::UNorm, false},          // RG16
    {GL_RG16F, GL_RG, GL_HALF_FLOAT, ComponentType::Float, false},             // RG16F
    {GL_RG16UI, GL_RG_INTEGER, GL_UNSIGNED_SHORT, ComponentType::UInt, false}, // RG16UI
    {GL_RG16I, GL_RG_INTEGER, GL_SHORT, ComponentType::SInt, false},           // RG16I
    {GL_RG16_SNORM, GL_RG, GL_SHORT, ComponentType::SNorm, false},             // RG16S
    {GL_RGB32F, GL_RGB, GL_FLOAT, ComponentType::Float, false},                // RGB32F
    {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, ComponentType::UNorm,
     false},                                                                   // RGBA8_SRGB
    {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, ComponentType::UNorm, false},            // RG8U
    {GL_RG8, GL_RG, GL_BYTE, ComponentType::SNorm, false},                     // RG8S
    {GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT, ComponentType::UInt, false},   // RG32UI
    {GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, ComponentType::UInt, false},   // R32UI
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false},        // ASTC_2D_8X8
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false},        // ASTC_2D_8X5
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false},        // ASTC_2D_5X4
    {GL_SRGB8_ALPHA8, GL_BGRA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false}, // BGRA8
    // Compressed sRGB formats
    {GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, ComponentType::UNorm,
     true}, // DXT1_SRGB
    {GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, ComponentType::UNorm,
     true}, // DXT23_SRGB
    {GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, ComponentType::UNorm,
     true}, // DXT45_SRGB
    {GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, ComponentType::UNorm,
     true},                                                                    // BC7U_SRGB
    {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false}, // ASTC_2D_4X4_SRGB
    {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false}, // ASTC_2D_8X8_SRGB
    {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false}, // ASTC_2D_8X5_SRGB
    {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false}, // ASTC_2D_5X4_SRGB
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false},        // ASTC_2D_5X5
    {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false}, // ASTC_2D_5X5_SRGB
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false},        // ASTC_2D_10X8
    {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, ComponentType::UNorm, false}, // ASTC_2D_10X8_SRGB

    // Depth formats
    {GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, ComponentType::Float, false}, // Z32F
    {GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, ComponentType::UNorm,
     false}, // Z16

    // DepthStencil formats
    {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, ComponentType::UNorm,
     false}, // Z24S8
    {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, ComponentType::UNorm,
     false}, // S8Z24
    {GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL, GL_FLOAT_32_UNSIGNED_INT_24_8_REV,
     ComponentType::Float, false}, // Z32FS8
}};

const FormatTuple& GetFormatTuple(PixelFormat pixel_format, ComponentType component_type) {
    ASSERT(static_cast<std::size_t>(pixel_format) < tex_format_tuples.size());
    const auto& format{tex_format_tuples[static_cast<std::size_t>(pixel_format)]};
    ASSERT(component_type == format.component_type);
    return format;
}

GLenum GetTextureTarget(const SurfaceParams& params) {
    switch (params.GetTarget()) {
    case SurfaceTarget::Texture1D:
        return GL_TEXTURE_1D;
    case SurfaceTarget::Texture2D:
        return GL_TEXTURE_2D;
    case SurfaceTarget::Texture3D:
        return GL_TEXTURE_3D;
    case SurfaceTarget::Texture1DArray:
        return GL_TEXTURE_1D_ARRAY;
    case SurfaceTarget::Texture2DArray:
        return GL_TEXTURE_2D_ARRAY;
    case SurfaceTarget::TextureCubemap:
        return GL_TEXTURE_CUBE_MAP;
    case SurfaceTarget::TextureCubeArray:
        return GL_TEXTURE_CUBE_MAP_ARRAY;
    }
    UNREACHABLE();
    return {};
}

GLenum GetTextureViewTarget(const SurfaceParams& params, const ViewKey& key, bool is_array) {
    // TODO(Rodrigo): Support Cube <-> 2D views

    if (is_array) {
        switch (params.GetTarget()) {
        case SurfaceTarget::Texture1D:
        case SurfaceTarget::Texture1DArray:
            return GL_TEXTURE_1D_ARRAY;
        case SurfaceTarget::Texture2D:
        case SurfaceTarget::Texture2DArray:
            return GL_TEXTURE_2D_ARRAY;
        case SurfaceTarget::TextureCubemap:
        case SurfaceTarget::TextureCubeArray:
            return GL_TEXTURE_CUBE_MAP_ARRAY;
        case SurfaceTarget::Texture3D:
            UNREACHABLE_MSG("Tegra has no arrayed 3D textures");
            return GL_NONE;
        default:
            UNREACHABLE();
            return GL_NONE;
        }
    } else {
        switch (params.GetTarget()) {
        case SurfaceTarget::Texture1D:
        case SurfaceTarget::Texture1DArray:
            return GL_TEXTURE_1D_ARRAY;
        case SurfaceTarget::Texture2D:
        case SurfaceTarget::Texture2DArray:
            return GL_TEXTURE_2D_ARRAY;
        case SurfaceTarget::TextureCubemap:
        case SurfaceTarget::TextureCubeArray:
            return GL_TEXTURE_CUBE_MAP_ARRAY;
        case SurfaceTarget::Texture3D:
            return GL_TEXTURE_3D;
        default:
            UNREACHABLE();
            return GL_NONE;
        }
    }
}

void ApplyTextureDefaults(const SurfaceParams& params, GLuint texture) {
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(texture, GL_TEXTURE_MAX_LEVEL, params.GetNumLevels() - 1);
    if (params.GetNumLevels() == 1) {
        glTextureParameterf(texture, GL_TEXTURE_LOD_BIAS, 1000.0f);
    }
}

OGLTexture CreateTexture(const SurfaceParams& params, GLenum internal_format) {
    OGLTexture texture;
    texture.Create(GetTextureTarget(params));

    switch (params.GetTarget()) {
    case SurfaceTarget::Texture1D:
        glTextureStorage1D(texture.handle, params.GetNumLevels(), internal_format,
                           params.GetWidth());
        break;
    case SurfaceTarget::Texture2D:
    case SurfaceTarget::TextureCubemap:
        glTextureStorage2D(texture.handle, params.GetNumLevels(), internal_format,
                           params.GetWidth(), params.GetHeight());
        break;
    case SurfaceTarget::Texture3D:
    case SurfaceTarget::Texture2DArray:
    case SurfaceTarget::TextureCubeArray:
        glTextureStorage3D(texture.handle, params.GetNumLevels(), internal_format,
                           params.GetWidth(), params.GetHeight(), params.GetDepth());
        break;
    default:
        UNREACHABLE();
    }

    ApplyTextureDefaults(params, texture.handle);

    return texture;
}

void SwizzleFunc(MortonSwizzleMode mode, u8* memory, const SurfaceParams& params, u8* buffer,
                 u32 level) {
    const u32 width{params.GetMipWidth(level)};
    const u32 height{params.GetMipHeight(level)};
    const u32 block_height{params.GetMipBlockHeight(level)};
    const u32 block_depth{params.GetMipBlockDepth(level)};

    std::size_t guest_offset{params.GetGuestMipmapLevelOffset(level)};
    if (params.IsLayered()) {
        std::size_t host_offset{0};
        const std::size_t guest_stride = params.GetGuestLayerSize();
        const std::size_t host_stride = params.GetHostLayerSize(level);
        for (u32 layer = 0; layer < params.GetNumLayers(); layer++) {
            MortonSwizzle(mode, params.GetPixelFormat(), width, block_height, height, block_depth,
                          1, params.GetTileWidthSpacing(), buffer + host_offset,
                          memory + guest_offset);
            guest_offset += guest_stride;
            host_offset += host_stride;
        }
    } else {
        MortonSwizzle(mode, params.GetPixelFormat(), width, block_height, height, block_depth,
                      params.GetMipDepth(level), params.GetTileWidthSpacing(), buffer,
                      memory + guest_offset);
    }
}

} // Anonymous namespace

CachedSurface::CachedSurface(const SurfaceParams& params) : SurfaceBase{params} {
    const auto& tuple{GetFormatTuple(params.GetPixelFormat(), params.GetComponentType())};
    internal_format = tuple.internal_format;
    format = tuple.format;
    type = tuple.type;
    is_compressed = tuple.compressed;
    texture = CreateTexture(params, internal_format);
}

CachedSurface::~CachedSurface() = default;

void CachedSurface::LoadBuffer() {
    if (params.IsTiled()) {
        ASSERT_MSG(params.GetBlockWidth() == 1, "Block width is defined as {} on texture target {}",
                   params.GetBlockWidth(), static_cast<u32>(params.GetTarget()));
        for (u32 level = 0; level < params.GetNumLevels(); ++level) {
            u8* const buffer{staging_buffer.data() + params.GetHostMipmapLevelOffset(level)};
            SwizzleFunc(MortonSwizzleMode::MortonToLinear, GetHostPtr(), params, buffer, level);
        }
    } else {
        ASSERT_MSG(params.GetNumLevels() == 1, "Linear mipmap loading is not implemented");
        const u32 bpp{GetFormatBpp(params.GetPixelFormat()) / CHAR_BIT};
        const u32 copy_size{params.GetWidth() * bpp};
        if (params.GetPitch() == copy_size) {
            std::memcpy(staging_buffer.data(), GetHostPtr(), params.GetHostSizeInBytes());
        } else {
            const u8* start{GetHostPtr()};
            u8* write_to{staging_buffer.data()};
            for (u32 h = params.GetHeight(); h > 0; --h) {
                std::memcpy(write_to, start, copy_size);
                start += params.GetPitch();
                write_to += copy_size;
            }
        }
    }

    for (u32 level = 0; level < params.GetNumLevels(); ++level) {
        ConvertFromGuestToHost(staging_buffer.data() + params.GetHostMipmapLevelOffset(level),
                               params.GetPixelFormat(), params.GetMipWidth(level),
                               params.GetMipHeight(level), params.GetMipDepth(level), true, true);
    }
}

EmptyStruct CachedSurface::FlushBuffer(EmptyStruct) {
    if (!IsModified()) {
        return {};
    }

    for (u32 level = 0; level < params.GetNumLevels(); ++level) {
        glGetTextureImage(texture.handle, level, format, type,
                          static_cast<GLsizei>(params.GetHostMipmapSize(level)),
                          staging_buffer.data() + params.GetHostMipmapLevelOffset(level));
    }

    if (params.IsTiled()) {
        ASSERT_MSG(params.GetBlockWidth() == 1, "Block width is defined as {}",
                   params.GetBlockWidth());
        for (u32 level = 0; level < params.GetNumLevels(); ++level) {
            u8* const buffer = staging_buffer.data() + params.GetHostMipmapLevelOffset(level);
            SwizzleFunc(MortonSwizzleMode::LinearToMorton, GetHostPtr(), params, buffer, level);
        }
    } else {
        UNIMPLEMENTED();
        /*
        ASSERT(params.GetTarget() == SurfaceTarget::Texture2D);
        ASSERT(params.GetNumLevels() == 1);

        const u32 bpp{params.GetFormatBpp() / 8};
        const u32 copy_size{params.GetWidth() * bpp};
        if (params.GetPitch() == copy_size) {
            std::memcpy(host_ptr, staging_buffer.data(), GetSizeInBytes());
        } else {
            u8* start{host_ptr};
            const u8* read_to{staging_buffer.data()};
            for (u32 h = params.GetHeight(); h > 0; --h) {
                std::memcpy(start, read_to, copy_size);
                start += params.GetPitch();
                read_to += copy_size;
            }
        }
        */
    }
    return {};
}

EmptyStruct CachedSurface::UploadTexture(EmptyStruct) {
    for (u32 level = 0; level < params.GetNumLevels(); ++level) {
        UploadTextureMipmap(level);
    }
    return {};
}

void CachedSurface::UploadTextureMipmap(u32 level) {
    u8* buffer{staging_buffer.data() + params.GetHostMipmapLevelOffset(level)};

    // TODO(Rodrigo): Optimize alignment
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, static_cast<GLint>(params.GetMipWidth(level)));
    SCOPE_EXIT({ glPixelStorei(GL_UNPACK_ROW_LENGTH, 0); });

    if (is_compressed) {
        const auto image_size{static_cast<GLsizei>(params.GetHostMipmapSize(level))};
        switch (params.GetTarget()) {
        case SurfaceTarget::Texture2D:
            glCompressedTextureSubImage2D(texture.handle, level, 0, 0,
                                          static_cast<GLsizei>(params.GetMipWidth(level)),
                                          static_cast<GLsizei>(params.GetMipHeight(level)),
                                          internal_format, image_size, buffer);
            break;
        case SurfaceTarget::Texture3D:
            glCompressedTextureSubImage3D(texture.handle, level, 0, 0, 0,
                                          static_cast<GLsizei>(params.GetMipWidth(level)),
                                          static_cast<GLsizei>(params.GetMipHeight(level)),
                                          static_cast<GLsizei>(params.GetMipDepth(level)),
                                          internal_format, image_size, buffer);
            break;
        case SurfaceTarget::Texture2DArray:
        case SurfaceTarget::TextureCubeArray:
            glCompressedTextureSubImage3D(
                texture.handle, level, 0, 0, 0, static_cast<GLsizei>(params.GetMipWidth(level)),
                static_cast<GLsizei>(params.GetMipHeight(level)),
                static_cast<GLsizei>(params.GetDepth()), internal_format, image_size, buffer);
            break;
        case SurfaceTarget::TextureCubemap: {
            const std::size_t layer_size{params.GetHostLayerSize(level)};
            for (std::size_t face = 0; face < params.GetDepth(); ++face) {
                glCompressedTextureSubImage3D(texture.handle, level, 0, 0, static_cast<GLint>(face),
                                              static_cast<GLsizei>(params.GetMipWidth(level)),
                                              static_cast<GLsizei>(params.GetMipHeight(level)), 1,
                                              internal_format, static_cast<GLsizei>(layer_size),
                                              buffer);
                buffer += layer_size;
            }
            break;
        }
        default:
            UNREACHABLE();
        }
    } else {
        switch (params.GetTarget()) {
        case SurfaceTarget::Texture1D:
            glTextureSubImage1D(texture.handle, level, 0, params.GetMipWidth(level), format, type,
                                buffer);
            break;
        case SurfaceTarget::Texture1DArray:
        case SurfaceTarget::Texture2D:
            glTextureSubImage2D(texture.handle, level, 0, 0, params.GetMipWidth(level),
                                params.GetMipHeight(level), format, type, buffer);
            break;
        case SurfaceTarget::Texture2DArray:
        case SurfaceTarget::TextureCubeArray:
            glTextureSubImage3D(texture.handle, level, 0, 0, 0, params.GetMipWidth(level),
                                params.GetMipHeight(level), params.GetDepth(), format, type,
                                buffer);
            break;
        case SurfaceTarget::TextureCubemap:
            for (std::size_t face = 0; face < params.GetDepth(); ++face) {
                glTextureSubImage3D(texture.handle, level, 0, 0, static_cast<GLint>(face),
                                    params.GetMipWidth(level), params.GetMipHeight(level), 1,
                                    format, type, buffer);
                buffer += params.GetHostLayerSize(level);
            }
            break;
        default:
            UNREACHABLE();
        }
    }
}

std::unique_ptr<CachedView> CachedSurface::CreateView(const ViewKey& view_key) {
    return std::make_unique<CachedView>(*this, view_key);
}

CachedView::CachedView(CachedSurface& surface, ViewKey key)
    : surface{surface}, key{key}, params{surface.GetSurfaceParams()} {}

CachedView::~CachedView() = default;

GLuint CachedView::GetTexture(bool is_array) {
    OGLTexture& texture = is_array ? arrayed_texture : normal_texture;
    if (texture.handle == 0) {
        texture = CreateTexture(is_array);
    }
    return texture.handle;
}

OGLTexture CachedView::CreateTexture(bool is_array) const {
    const FormatTuple& tuple{GetFormatTuple(params.GetPixelFormat(), params.GetComponentType())};

    OGLTexture texture;
    glGenTextures(1, &texture.handle);
    glTextureView(texture.handle, GetTextureViewTarget(params, key, is_array),
                  surface.texture.handle, tuple.internal_format, key.base_level, key.num_levels,
                  key.base_layer, key.num_layers);
    ApplyTextureDefaults(params, texture.handle);
    return texture;
}

TextureCacheOpenGL::TextureCacheOpenGL(Core::System& system,
                                       VideoCore::RasterizerInterface& rasterizer)
    : TextureCacheBase{system, rasterizer} {}

TextureCacheOpenGL::~TextureCacheOpenGL() = default;

std::tuple<CachedView*, EmptyStruct> TextureCacheOpenGL::TryFastGetSurfaceView(
    EmptyStruct, VAddr cpu_addr, u8* host_ptr, const SurfaceParams& params, bool preserve_contents,
    const std::vector<CachedSurface*>& overlaps) {
    return {{}, {}};
}

std::unique_ptr<CachedSurface> TextureCacheOpenGL::CreateSurface(const SurfaceParams& params) {
    return std::make_unique<CachedSurface>(params);
}

} // namespace OpenGL
