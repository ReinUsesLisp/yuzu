// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <bit>
#include <string>

#include <glad/glad.h>

#include "video_core/renderer_opengl/gl_device.h"
#include "video_core/renderer_opengl/gl_shader_manager.h"
#include "video_core/renderer_opengl/gl_state_tracker.h"
#include "video_core/renderer_opengl/gl_texture_cache.h"
#include "video_core/renderer_opengl/maxwell_to_gl.h"
#include "video_core/renderer_opengl/util_shaders.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/format_lookup_table.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/textures/decoders.h"

namespace OpenGL {

namespace {

using Tegra::Texture::SwizzleSource;
using Tegra::Texture::TextureMipmapFilter;
using Tegra::Texture::TextureType;
using Tegra::Texture::TICEntry;
using Tegra::Texture::TSCEntry;
using VideoCommon::CalculateLevelStrideAlignment;
using VideoCommon::ImageCopy;
using VideoCommon::ImageType;
using VideoCommon::NUM_RT;
using VideoCommon::SwizzleParameters;
using VideoCore::Surface::BytesPerBlock;
using VideoCore::Surface::IsPixelFormatASTC;
using VideoCore::Surface::IsPixelFormatSRGB;
using VideoCore::Surface::MaxPixelFormat;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::SurfaceType;

struct CopyOrigin {
    GLint level;
    GLint x;
    GLint y;
    GLint z;
};

struct CopyRegion {
    GLsizei width;
    GLsizei height;
    GLsizei depth;
};

struct FormatTuple {
    GLenum internal_format;
    GLenum format = GL_NONE;
    GLenum type = GL_NONE;
    GLenum store_format = internal_format;
};

constexpr std::array<FormatTuple, MaxPixelFormat> FORMAT_TABLE = {{
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV},                  // A8B8G8R8_UNORM
    {GL_RGBA8_SNORM, GL_RGBA, GL_BYTE},                                // A8B8G8R8_SNORM
    {GL_RGBA8I, GL_RGBA_INTEGER, GL_BYTE},                             // A8B8G8R8_SINT
    {GL_RGBA8UI, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE},                   // A8B8G8R8_UINT
    {GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},                      // R5G6B5_UNORM
    {GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5_REV},                  // B5G6R5_UNORM
    {GL_RGB5_A1, GL_BGRA, GL_UNSIGNED_SHORT_1_5_5_5_REV},              // A1R5G5B5_UNORM
    {GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV},            // A2B10G10R10_UNORM
    {GL_RGB10_A2UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV},  // A2B10G10R10_UINT
    {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_1_5_5_5_REV},              // A1B5G5R5_UNORM
    {GL_R8, GL_RED, GL_UNSIGNED_BYTE},                                 // R8_UNORM
    {GL_R8_SNORM, GL_RED, GL_BYTE},                                    // R8_SNORM
    {GL_R8I, GL_RED_INTEGER, GL_BYTE},                                 // R8_SINT
    {GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE},                       // R8_UINT
    {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT},                              // R16G16B16A16_FLOAT
    {GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT},                           // R16G16B16A16_UNORM
    {GL_RGBA16_SNORM, GL_RGBA, GL_SHORT},                              // R16G16B16A16_SNORM
    {GL_RGBA16I, GL_RGBA_INTEGER, GL_SHORT},                           // R16G16B16A16_SINT
    {GL_RGBA16UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT},                 // R16G16B16A16_UINT
    {GL_R11F_G11F_B10F, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV},      // B10G11R11_FLOAT
    {GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT},                   // R32G32B32A32_UINT
    {GL_COMPRESSED_RGBA_S3TC_DXT1_EXT},                                // BC1_RGBA_UNORM
    {GL_COMPRESSED_RGBA_S3TC_DXT3_EXT},                                // BC2_UNORM
    {GL_COMPRESSED_RGBA_S3TC_DXT5_EXT},                                // BC3_UNORM
    {GL_COMPRESSED_RED_RGTC1},                                         // BC4_UNORM
    {GL_COMPRESSED_SIGNED_RED_RGTC1},                                  // BC4_SNORM
    {GL_COMPRESSED_RG_RGTC2},                                          // BC5_UNORM
    {GL_COMPRESSED_SIGNED_RG_RGTC2},                                   // BC5_SNORM
    {GL_COMPRESSED_RGBA_BPTC_UNORM},                                   // BC7_UNORM
    {GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT},                           // BC6H_UFLOAT
    {GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT},                             // BC6H_SFLOAT
    {GL_COMPRESSED_RGBA_ASTC_4x4_KHR},                                 // ASTC_2D_4X4_UNORM
    {GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE},                             // B8G8R8A8_UNORM
    {GL_RGBA32F, GL_RGBA, GL_FLOAT},                                   // R32G32B32A32_FLOAT
    {GL_RGBA32I, GL_RGBA_INTEGER, GL_INT},                             // R32G32B32A32_SINT
    {GL_RG32F, GL_RG, GL_FLOAT},                                       // R32G32_FLOAT
    {GL_RG32I, GL_RG_INTEGER, GL_INT},                                 // R32G32_SINT
    {GL_R32F, GL_RED, GL_FLOAT},                                       // R32_FLOAT
    {GL_R16F, GL_RED, GL_HALF_FLOAT},                                  // R16_FLOAT
    {GL_R16, GL_RED, GL_UNSIGNED_SHORT},                               // R16_UNORM
    {GL_R16_SNORM, GL_RED, GL_SHORT},                                  // R16_SNORM
    {GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT},                     // R16_UINT
    {GL_R16I, GL_RED_INTEGER, GL_SHORT},                               // R16_SINT
    {GL_RG16, GL_RG, GL_UNSIGNED_SHORT},                               // R16G16_UNORM
    {GL_RG16F, GL_RG, GL_HALF_FLOAT},                                  // R16G16_FLOAT
    {GL_RG16UI, GL_RG_INTEGER, GL_UNSIGNED_SHORT},                     // R16G16_UINT
    {GL_RG16I, GL_RG_INTEGER, GL_SHORT},                               // R16G16_SINT
    {GL_RG16_SNORM, GL_RG, GL_SHORT},                                  // R16G16_SNORM
    {GL_RGB32F, GL_RGB, GL_FLOAT},                                     // R32G32B32_FLOAT
    {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, GL_RGBA8}, // A8B8G8R8_SRGB
    {GL_RG8, GL_RG, GL_UNSIGNED_BYTE},                                 // R8G8_UNORM
    {GL_RG8_SNORM, GL_RG, GL_BYTE},                                    // R8G8_SNORM
    {GL_RG8I, GL_RG_INTEGER, GL_BYTE},                                 // R8G8_SINT
    {GL_RG8UI, GL_RG_INTEGER, GL_UNSIGNED_BYTE},                       // R8G8_UINT
    {GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT},                       // R32G32_UINT
    {GL_RGB16F, GL_RGBA, GL_HALF_FLOAT},                               // R16G16B16X16_FLOAT
    {GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT},                       // R32_UINT
    {GL_R32I, GL_RED_INTEGER, GL_INT},                                 // R32_SINT
    {GL_COMPRESSED_RGBA_ASTC_8x8_KHR},                                 // ASTC_2D_8X8_UNORM
    {GL_COMPRESSED_RGBA_ASTC_8x5_KHR},                                 // ASTC_2D_8X5_UNORM
    {GL_COMPRESSED_RGBA_ASTC_5x4_KHR},                                 // ASTC_2D_5X4_UNORM
    {GL_SRGB8_ALPHA8, GL_BGRA, GL_UNSIGNED_BYTE, GL_RGBA8},            // B8G8R8A8_UNORM
    {GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT},                          // BC1_RGBA_SRGB
    {GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT},                          // BC2_SRGB
    {GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT},                          // BC3_SRGB
    {GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM},                             // BC7_SRGB
    {GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4_REV},                // A4B4G4R4_UNORM
    {GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR},                         // ASTC_2D_4X4_SRGB
    {GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR},                         // ASTC_2D_8X8_SRGB
    {GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR},                         // ASTC_2D_8X5_SRGB
    {GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR},                         // ASTC_2D_5X4_SRGB
    {GL_COMPRESSED_RGBA_ASTC_5x5_KHR},                                 // ASTC_2D_5X5_UNORM
    {GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR},                         // ASTC_2D_5X5_SRGB
    {GL_COMPRESSED_RGBA_ASTC_10x8_KHR},                                // ASTC_2D_10X8_UNORM
    {GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR},                        // ASTC_2D_10X8_SRGB
    {GL_COMPRESSED_RGBA_ASTC_6x6_KHR},                                 // ASTC_2D_6X6_UNORM
    {GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR},                         // ASTC_2D_6X6_SRGB
    {GL_COMPRESSED_RGBA_ASTC_10x10_KHR},                               // ASTC_2D_10X10_UNORM
    {GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR},                       // ASTC_2D_10X10_SRGB
    {GL_COMPRESSED_RGBA_ASTC_12x12_KHR},                               // ASTC_2D_12X12_UNORM
    {GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR},                       // ASTC_2D_12X12_SRGB
    {GL_COMPRESSED_RGBA_ASTC_8x6_KHR},                                 // ASTC_2D_8X6_UNORM
    {GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR},                         // ASTC_2D_8X6_SRGB
    {GL_COMPRESSED_RGBA_ASTC_6x5_KHR},                                 // ASTC_2D_6X5_UNORM
    {GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR},                         // ASTC_2D_6X5_SRGB
    {GL_RGB9_E5, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV},                 // E5B9G9R9_FLOAT
    {GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT},             // D32_FLOAT
    {GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT},     // D16_UNORM
    {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8},     // D24_UNORM_S8_UINT
    {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8},     // S8_UINT_D24_UNORM
    {GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL,
     GL_FLOAT_32_UNSIGNED_INT_24_8_REV}, // D32_FLOAT_S8_UINT
}};

constexpr std::array ACCELERATED_FORMATS{
    GL_RGBA32F,   GL_RGBA16F,   GL_RG32F,    GL_RG16F,        GL_R11F_G11F_B10F, GL_R32F,
    GL_R16F,      GL_RGBA32UI,  GL_RGBA16UI, GL_RGB10_A2UI,   GL_RGBA8UI,        GL_RG32UI,
    GL_RG16UI,    GL_RG8UI,     GL_R32UI,    GL_R16UI,        GL_R8UI,           GL_RGBA32I,
    GL_RGBA16I,   GL_RGBA8I,    GL_RG32I,    GL_RG16I,        GL_RG8I,           GL_R32I,
    GL_R16I,      GL_R8I,       GL_RGBA16,   GL_RGB10_A2,     GL_RGBA8,          GL_RG16,
    GL_RG8,       GL_R16,       GL_R8,       GL_RGBA16_SNORM, GL_RGBA8_SNORM,    GL_RG16_SNORM,
    GL_RG8_SNORM, GL_R16_SNORM, GL_R8_SNORM,
};

const FormatTuple& GetFormatTuple(PixelFormat pixel_format) {
    ASSERT(static_cast<size_t>(pixel_format) < FORMAT_TABLE.size());
    return FORMAT_TABLE[static_cast<size_t>(pixel_format)];
}

GLenum ImageTarget(const VideoCommon::ImageInfo& info) {
    switch (info.type) {
    case ImageType::e1D:
        return GL_TEXTURE_1D_ARRAY;
    case ImageType::e2D:
        if (info.num_samples > 1) {
            return GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
        }
        return GL_TEXTURE_2D_ARRAY;
    case ImageType::e3D:
        return GL_TEXTURE_3D;
    case ImageType::Linear:
        return GL_TEXTURE_2D_ARRAY;
    case ImageType::Rect:
        ASSERT(info.num_samples == 1);
        return GL_TEXTURE_RECTANGLE;
    case ImageType::Buffer:
        return GL_TEXTURE_BUFFER;
    }
    UNREACHABLE_MSG("Invalid image type={}", static_cast<int>(info.type));
    return GL_NONE;
}

GLenum ImageTarget(ImageViewType type, int num_samples = 1) {
    const bool is_multisampled = num_samples > 1;
    switch (type) {
    case ImageViewType::e1D:
        return GL_TEXTURE_1D;
    case ImageViewType::e2D:
        return is_multisampled ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;
    case ImageViewType::Cube:
        return GL_TEXTURE_CUBE_MAP;
    case ImageViewType::e3D:
        return GL_TEXTURE_3D;
    case ImageViewType::e1DArray:
        return GL_TEXTURE_1D_ARRAY;
    case ImageViewType::e2DArray:
        return is_multisampled ? GL_TEXTURE_2D_MULTISAMPLE_ARRAY : GL_TEXTURE_2D_ARRAY;
    case ImageViewType::CubeArray:
        return GL_TEXTURE_CUBE_MAP_ARRAY;
    case ImageViewType::Rect:
        return GL_TEXTURE_RECTANGLE;
    case ImageViewType::Buffer:
        return GL_TEXTURE_BUFFER;
    }
    UNREACHABLE_MSG("Invalid image view type={}", static_cast<int>(type));
    return GL_NONE;
}

GLenum TextureMode(PixelFormat format, bool is_first) {
    switch (format) {
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::D32_FLOAT_S8_UINT:
        return is_first ? GL_DEPTH_COMPONENT : GL_STENCIL_INDEX;
    case PixelFormat::S8_UINT_D24_UNORM:
        return is_first ? GL_STENCIL_INDEX : GL_DEPTH_COMPONENT;
    default:
        UNREACHABLE();
        return GL_DEPTH_COMPONENT;
    }
}

constexpr SwizzleSource SwapRedGreen(SwizzleSource value) {
    switch (value) {
    case SwizzleSource::R:
        return SwizzleSource::G;
    case SwizzleSource::G:
        return SwizzleSource::R;
    default:
        return value;
    }
}

GLint Swizzle(SwizzleSource source) {
    switch (source) {
    case SwizzleSource::Zero:
        return GL_ZERO;
    case SwizzleSource::R:
        return GL_RED;
    case SwizzleSource::G:
        return GL_GREEN;
    case SwizzleSource::B:
        return GL_BLUE;
    case SwizzleSource::A:
        return GL_ALPHA;
    case SwizzleSource::OneInt:
    case SwizzleSource::OneFloat:
        return GL_ONE;
    }

    UNREACHABLE_MSG("Invalid swizzle source={}", static_cast<int>(source));
    return GL_NONE;
}

GLenum AttachmentType(PixelFormat format) {
    switch (const SurfaceType type = VideoCore::Surface::GetFormatType(format); type) {
    case SurfaceType::Depth:
        return GL_DEPTH_ATTACHMENT;
    case SurfaceType::DepthStencil:
        return GL_DEPTH_STENCIL_ATTACHMENT;
    default:
        UNIMPLEMENTED_MSG("Unimplemented type={}", static_cast<int>(type));
        return GL_NONE;
    }
}

constexpr std::string_view DepthStencilDebugName(GLenum attachment) {
    switch (attachment) {
    case GL_DEPTH_ATTACHMENT:
        return "D";
    case GL_DEPTH_STENCIL_ATTACHMENT:
        return "DS";
    case GL_STENCIL_ATTACHMENT:
        return "S";
    }
    return "X";
}

std::string NameView(const VideoCommon::ImageViewBase& image_view) {
    const auto size = image_view.size;
    const u32 num_mipmaps = image_view.range.extent.mipmaps;
    const u32 num_layers = image_view.range.extent.layers;

    const std::string mipmap = num_mipmaps > 1 ? fmt::format(":{}", num_mipmaps) : "";
    switch (image_view.type) {
    case ImageViewType::e1D:
        return fmt::format("1D {}{}", size.width, mipmap);
    case ImageViewType::e2D:
        return fmt::format("2D {}x{}{}", size.width, size.height, mipmap);
    case ImageViewType::Cube:
        return fmt::format("Cube {}x{}{}", size.width, size.height, mipmap);
    case ImageViewType::e3D:
        return fmt::format("3D {}x{}x{}{}", size.width, size.height, size.depth, mipmap);
    case ImageViewType::e1DArray:
        return fmt::format("1DArray {}{}|{}", size.width, mipmap, num_layers);
    case ImageViewType::e2DArray:
        return fmt::format("2DArray {}x{}{}|{}", size.width, size.height, mipmap, num_layers);
    case ImageViewType::CubeArray:
        return fmt::format("CubeArray {}x{}{}|{}", size.width, size.height, mipmap, num_layers);
    case ImageViewType::Rect:
        return fmt::format("Rect {}x{}{}", size.width, size.height, mipmap);
    case ImageViewType::Buffer:
        return fmt::format("Buffer {}", size.width);
    }
    return "Invalid";
}

void ApplySwizzle(GLuint handle, PixelFormat format, std::array<SwizzleSource, 4> swizzle) {
    switch (format) {
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::D32_FLOAT_S8_UINT:
    case PixelFormat::S8_UINT_D24_UNORM:
        UNIMPLEMENTED_IF(swizzle[0] != SwizzleSource::R && swizzle[0] != SwizzleSource::G);
        glTextureParameteri(handle, GL_DEPTH_STENCIL_TEXTURE_MODE,
                            TextureMode(format, swizzle[0] == SwizzleSource::R));
        break;
    default:
        break;
    }
    if (format == PixelFormat::S8_UINT_D24_UNORM) {
        // Make sure we sample the first component
        std::ranges::transform(swizzle, swizzle.begin(), SwapRedGreen);
    }
    std::array<GLint, 4> gl_swizzle;
    std::ranges::transform(swizzle, gl_swizzle.begin(), Swizzle);
    glTextureParameteriv(handle, GL_TEXTURE_SWIZZLE_RGBA, gl_swizzle.data());
}

[[nodiscard]] bool CanBeAccelerated(const TextureCacheRuntime& runtime,
                                    const VideoCommon::ImageInfo& info) {
    switch (info.type) {
    case ImageType::e2D:
    case ImageType::e3D:
    case ImageType::Linear:
        break;
    default:
        return false;
    }
    const GLenum internal_format = GetFormatTuple(info.format).internal_format;
    const auto& format_info = runtime.FormatInfo(info.type, internal_format);
    if (format_info.is_compressed) {
        return false;
    }
    if (std::ranges::find(ACCELERATED_FORMATS, internal_format) == ACCELERATED_FORMATS.end()) {
        return false;
    }
    if (format_info.compatibility_by_size) {
        return true;
    }
    const GLenum store_format = StoreFormat(BytesPerBlock(info.format));
    const GLenum store_class = runtime.FormatInfo(info.type, store_format).compatibility_class;
    return format_info.compatibility_class == store_class;
}

[[nodiscard]] CopyOrigin MakeCopyOrigin(VideoCommon::Offset3D offset,
                                        VideoCommon::SubresourceLayers subresource, GLenum target) {
    switch (target) {
    case GL_TEXTURE_2D_ARRAY:
        return CopyOrigin{
            .level = static_cast<GLint>(subresource.base_mipmap),
            .x = static_cast<GLint>(offset.x),
            .y = static_cast<GLint>(offset.y),
            .z = static_cast<GLint>(subresource.base_layer),
        };
    case GL_TEXTURE_3D:
        return CopyOrigin{
            .level = static_cast<GLint>(subresource.base_mipmap),
            .x = static_cast<GLint>(offset.x),
            .y = static_cast<GLint>(offset.y),
            .z = static_cast<GLint>(offset.z),
        };
    default:
        UNIMPLEMENTED_MSG("Unimplemented copy target={}", target);
        return CopyOrigin{};
    }
}

[[nodiscard]] CopyRegion MakeCopyRegion(VideoCommon::Extent3D extent,
                                        VideoCommon::SubresourceLayers dst_subresource,
                                        GLenum target) {
    switch (target) {
    case GL_TEXTURE_2D_ARRAY:
        return CopyRegion{
            .width = static_cast<GLsizei>(extent.width),
            .height = static_cast<GLsizei>(extent.height),
            .depth = static_cast<GLsizei>(dst_subresource.num_layers),
        };
    case GL_TEXTURE_3D:
        return CopyRegion{
            .width = static_cast<GLsizei>(extent.width),
            .height = static_cast<GLsizei>(extent.height),
            .depth = static_cast<GLsizei>(extent.depth),
        };
    default:
        UNIMPLEMENTED_MSG("Unimplemented copy target={}", target);
        return CopyRegion{};
    }
}

void AttachTexture(GLuint fbo, GLenum attachment, const ImageView* image_view) {
    if (!image_view->Is3D()) {
        const GLuint texture = image_view->DefaultHandle();
        glNamedFramebufferTexture(fbo, attachment, texture, 0);
        return;
    }
    const GLuint texture = image_view->Handle(ImageViewType::e3D);
    if (image_view->range.extent.layers > 1) {
        // TODO: OpenGL doesn't support rendering to a fixed number of slices
        glNamedFramebufferTexture(fbo, attachment, texture, 0);
    } else {
        const u32 slice = image_view->range.base.layer;
        glNamedFramebufferTextureLayer(fbo, attachment, texture, 0, slice);
    }
}

} // Anonymous namespace

ImageBufferMap::ImageBufferMap(GLuint handle_, u8* map, size_t size, OGLSync* sync_)
    : span(map, size), sync{sync_}, handle{handle_} {}

ImageBufferMap::~ImageBufferMap() {
    if (sync) {
        sync->Create();
    }
}

TextureCacheRuntime::TextureCacheRuntime(const Device& device_, ProgramManager& program_manager,
                                         StateTracker& state_tracker_)
    : device{device_}, state_tracker{state_tracker_}, util_shaders(program_manager) {
    static constexpr std::array TARGETS{GL_TEXTURE_1D_ARRAY, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_3D};
    for (size_t i = 0; i < TARGETS.size(); ++i) {
        const GLenum target = TARGETS[i];
        for (const FormatTuple& tuple : FORMAT_TABLE) {
            const GLenum format = tuple.internal_format;

            GLint compat_class;
            GLint compat_type;
            GLint is_compressed;
            glGetInternalformativ(target, format, GL_IMAGE_COMPATIBILITY_CLASS, 1, &compat_class);
            glGetInternalformativ(target, format, GL_IMAGE_FORMAT_COMPATIBILITY_TYPE, 1,
                                  &compat_type);
            glGetInternalformativ(target, format, GL_TEXTURE_COMPRESSED, 1, &is_compressed);

            const FormatProperties properties{
                .compatibility_class = static_cast<GLenum>(compat_class),
                .compatibility_by_size = compat_type == GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE,
                .is_compressed = is_compressed == GL_TRUE,
            };
            format_properties[i].emplace(format, properties);
        }
    }
    null_image_1d.Create(GL_TEXTURE_1D);
    null_image_2d.Create(GL_TEXTURE_2D_ARRAY);
    null_image_3d.Create(GL_TEXTURE_3D);
    null_image_rect.Create(GL_TEXTURE_RECTANGLE);
    glTextureStorage1D(null_image_1d.handle, 1, GL_R8, 1);
    glTextureStorage3D(null_image_2d.handle, 1, GL_R8, 1, 1, 6);
    glTextureStorage3D(null_image_3d.handle, 1, GL_R8, 1, 1, 1);
    glTextureStorage2D(null_image_rect.handle, 1, GL_R8, 1, 1);
}

TextureCacheRuntime::~TextureCacheRuntime() = default;

ImageBufferMap TextureCacheRuntime::MapUploadBuffer(size_t size) {
    return upload_buffers.RequestMap(size, true);
}

ImageBufferMap TextureCacheRuntime::MapDownloadBuffer(size_t size) {
    return download_buffers.RequestMap(size, false);
}

void TextureCacheRuntime::CopyImage(Image& dst_image, Image& src_image,
                                    std::span<const ImageCopy> copies) {
    const GLuint dst_name = dst_image.Handle();
    const GLuint src_name = src_image.Handle();
    const GLenum src_target = ImageTarget(dst_image.info);
    const GLenum dst_target = ImageTarget(src_image.info);
    for (const ImageCopy& copy : copies) {
        const auto src_origin = MakeCopyOrigin(copy.src_offset, copy.src_subresource, src_target);
        const auto dst_origin = MakeCopyOrigin(copy.dst_offset, copy.dst_subresource, dst_target);
        const auto region = MakeCopyRegion(copy.extent, copy.dst_subresource, dst_target);
        glCopyImageSubData(src_name, src_target, src_origin.level, src_origin.x, src_origin.y,
                           src_origin.z, dst_name, dst_target, dst_origin.level, dst_origin.x,
                           dst_origin.y, dst_origin.z, region.width, region.height, region.depth);
    }
}

void TextureCacheRuntime::BlitFramebuffer(Framebuffer* dst, Framebuffer* src,
                                          const Tegra::Engines::Fermi2D::Config& copy) {
    state_tracker.NotifyScissor0();
    state_tracker.NotifyRasterizeEnable();
    state_tracker.NotifyFramebufferSRGB();

    glDisable(GL_FRAMEBUFFER_SRGB);
    glDisable(GL_RASTERIZER_DISCARD);
    glDisablei(GL_SCISSOR_TEST, 0);

    const bool is_linear = copy.filter == Tegra::Engines::Fermi2D::Filter::Bilinear;
    glBlitNamedFramebuffer(src->Handle(), dst->Handle(), copy.src_x0, copy.src_y0, copy.src_x1,
                           copy.src_y1, copy.dst_x0, copy.dst_y0, copy.dst_x1, copy.dst_y1,
                           GL_COLOR_BUFFER_BIT, is_linear ? GL_LINEAR : GL_NEAREST);
}

void TextureCacheRuntime::AccelerateImageUpload(Image& image, const ImageBufferMap& map,
                                                size_t buffer_offset,
                                                std::span<const SwizzleParameters> swizzles) {
    switch (image.info.type) {
    case ImageType::e2D:
        return util_shaders.BlockLinearUpload2D(image, map, buffer_offset, swizzles);
    case ImageType::e3D:
        return util_shaders.BlockLinearUpload3D(image, map, buffer_offset, swizzles);
    case ImageType::Linear:
        return util_shaders.PitchUpload(image, map, buffer_offset, swizzles);
    default:
        UNREACHABLE();
        break;
    }
}

void TextureCacheRuntime::InsertUploadMemoryBarrier() {
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

FormatProperties TextureCacheRuntime::FormatInfo(ImageType type, GLenum internal_format) const {
    switch (type) {
    case ImageType::e1D:
        return format_properties[0].at(internal_format);
    case ImageType::e2D:
    case ImageType::Rect:
    case ImageType::Linear:
        return format_properties[1].at(internal_format);
    case ImageType::e3D:
        return format_properties[2].at(internal_format);
    default:
        UNREACHABLE();
    }
}

TextureCacheRuntime::StagingBuffers::StagingBuffers(GLenum storage_flags_, GLenum map_flags_)
    : storage_flags{storage_flags_}, map_flags{map_flags_} {}

TextureCacheRuntime::StagingBuffers::~StagingBuffers() = default;

ImageBufferMap TextureCacheRuntime::StagingBuffers::RequestMap(size_t requested_size,
                                                               bool insert_fence) {
    const size_t index = RequestBuffer(requested_size);
    OGLSync* const sync = insert_fence ? &syncs[index] : nullptr;
    return ImageBufferMap(buffers[index].handle, maps[index], requested_size, sync);
}

size_t TextureCacheRuntime::StagingBuffers::RequestBuffer(size_t requested_size) {
    if (const std::optional<size_t> index = FindBuffer(requested_size); index) {
        return *index;
    }

    OGLBuffer& buffer = buffers.emplace_back();
    buffer.Create();
    glNamedBufferStorage(buffer.handle, requested_size, nullptr,
                         storage_flags | GL_MAP_PERSISTENT_BIT);
    maps.push_back(static_cast<u8*>(glMapNamedBufferRange(buffer.handle, 0, requested_size,
                                                          map_flags | GL_MAP_PERSISTENT_BIT)));

    syncs.emplace_back();
    sizes.push_back(requested_size);

    ASSERT(syncs.size() == buffers.size() && buffers.size() == maps.size() &&
           maps.size() == sizes.size());

    return buffers.size() - 1;
}

std::optional<size_t> TextureCacheRuntime::StagingBuffers::FindBuffer(size_t requested_size) {
    size_t smallest_buffer = std::numeric_limits<size_t>::max();
    std::optional<size_t> found;
    const size_t num_buffers = sizes.size();
    for (size_t index = 0; index < num_buffers; ++index) {
        const size_t buffer_size = sizes[index];
        if (buffer_size < requested_size || buffer_size >= smallest_buffer) {
            continue;
        }
        if (syncs[index].handle != 0) {
            GLint status;
            glGetSynciv(syncs[index].handle, GL_SYNC_STATUS, 1, nullptr, &status);
            if (status != GL_SIGNALED) {
                continue;
            }
            syncs[index].Release();
        }
        smallest_buffer = buffer_size;
        found = index;
    }
    return found;
}

Image::Image(TextureCacheRuntime& runtime, const VideoCommon::ImageInfo& info, GPUVAddr gpu_addr,
             VAddr cpu_addr)
    : VideoCommon::ImageBase(info, gpu_addr, cpu_addr) {
    if (CanBeAccelerated(runtime, info)) {
        flags |= VideoCommon::ImageFlagBits::AcceleratedUpload;
    }
    if (runtime.device.HasASTC() || !IsPixelFormatASTC(info.format)) {
        const auto& tuple = GetFormatTuple(info.format);
        gl_internal_format = tuple.internal_format;
        gl_store_format = tuple.store_format;
        gl_format = tuple.format;
        gl_type = tuple.type;
    } else {
        flags |= VideoCommon::ImageFlagBits::Converted;
        gl_internal_format = IsPixelFormatSRGB(info.format) ? GL_SRGB8_ALPHA8 : GL_RGBA8;
        gl_store_format = GL_RGBA8;
        gl_format = GL_RGBA;
        gl_type = GL_UNSIGNED_INT_8_8_8_8_REV;
    }
    const GLenum target = ImageTarget(info);
    const GLsizei width = info.size.width;
    const GLsizei height = info.size.height;
    const GLsizei depth = info.size.depth;
    const GLsizei num_mipmaps = info.resources.mipmaps;
    const GLsizei num_layers = info.resources.layers;
    const GLsizei num_samples = info.num_samples;

    GLuint handle = 0;
    if (target != GL_TEXTURE_BUFFER) {
        texture.Create(target);
        handle = texture.handle;
    }
    switch (target) {
    case GL_TEXTURE_1D_ARRAY:
        glTextureStorage2D(handle, num_mipmaps, gl_store_format, width, num_layers);
        break;
    case GL_TEXTURE_2D_ARRAY:
        glTextureStorage3D(handle, num_mipmaps, gl_store_format, width, height, num_layers);
        break;
    case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        // TODO: Where should 'fixedsamplelocations' come from?
        glTextureStorage3DMultisample(handle, num_samples, gl_store_format, width, height,
                                      num_layers, GL_FALSE);
        break;
    case GL_TEXTURE_RECTANGLE:
        glTextureStorage2D(handle, num_mipmaps, gl_store_format, width, height);
        break;
    case GL_TEXTURE_3D:
        glTextureStorage3D(handle, num_mipmaps, gl_store_format, width, height, depth);
        break;
    case GL_TEXTURE_BUFFER:
        buffer.Create();
        glNamedBufferStorage(buffer.handle, guest_size_bytes, nullptr, 0);
        break;
    default:
        UNREACHABLE_MSG("Invalid target=0x{:x}", target);
        break;
    }
    if (target != GL_TEXTURE_BUFFER) {
        const std::string name = fmt::format("Image 0x{:x}", gpu_addr);
        glObjectLabel(GL_TEXTURE, handle, static_cast<GLsizei>(name.size()), name.data());
    } else {
        const std::string name = fmt::format("Buffer 0x{:x}", gpu_addr);
        glObjectLabel(GL_BUFFER, buffer.handle, static_cast<GLsizei>(name.size()), name.data());
    }
}

void Image::UploadMemory(const ImageBufferMap& map, size_t buffer_offset,
                         std::span<const VideoCommon::BufferImageCopy> copies) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, map.Handle());
    glFlushMappedBufferRange(GL_PIXEL_UNPACK_BUFFER, buffer_offset, unswizzled_size_bytes);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // FIXME

    u32 current_row_length = std::numeric_limits<u32>::max();
    u32 current_image_height = std::numeric_limits<u32>::max();

    for (const VideoCommon::BufferImageCopy& copy : copies) {
        if (current_row_length != copy.buffer_row_length) {
            current_row_length = copy.buffer_row_length;
            glPixelStorei(GL_UNPACK_ROW_LENGTH, current_row_length);
        }
        if (current_image_height != copy.buffer_image_height) {
            current_image_height = copy.buffer_image_height;
            glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, current_image_height);
        }
        CopyBufferToImage(copy, buffer_offset);
    }
}

void Image::UploadMemory(const ImageBufferMap& map, size_t buffer_offset,
                         std::span<const VideoCommon::BufferCopy> copies) {
    for (const VideoCommon::BufferCopy& copy : copies) {
        glCopyNamedBufferSubData(map.Handle(), buffer.handle, copy.src_offset + buffer_offset,
                                 copy.dst_offset, copy.size);
    }
}

void Image::DownloadMemory(ImageBufferMap& map, size_t buffer_offset,
                           std::span<const VideoCommon::BufferImageCopy> copies) {
    glMemoryBarrier(GL_PIXEL_BUFFER_BARRIER_BIT); // TODO: Move this to its own API

    glBindBuffer(GL_PIXEL_PACK_BUFFER, map.Handle());
    glPixelStorei(GL_PACK_ALIGNMENT, 1); // FIXME

    u32 current_row_length = std::numeric_limits<u32>::max();
    u32 current_image_height = std::numeric_limits<u32>::max();

    for (const VideoCommon::BufferImageCopy& copy : copies) {
        if (current_row_length != copy.buffer_row_length) {
            current_row_length = copy.buffer_row_length;
            glPixelStorei(GL_PACK_ROW_LENGTH, current_row_length);
        }
        if (current_image_height != copy.buffer_image_height) {
            current_image_height = copy.buffer_image_height;
            glPixelStorei(GL_PACK_IMAGE_HEIGHT, current_image_height);
        }
        CopyImageToBuffer(copy, buffer_offset);
    }
    glFinish();
}

void Image::CopyBufferToImage(const VideoCommon::BufferImageCopy& copy, size_t buffer_offset) {
    // Compressed formats don't have a pixel format or type
    const bool is_compressed = gl_format == GL_NONE;
    const void* const offset = reinterpret_cast<const void*>(copy.buffer_offset + buffer_offset);

    switch (info.type) {
    case ImageType::e1D:
        if (is_compressed) {
            glCompressedTextureSubImage2D(texture.handle, copy.image_subresource.base_mipmap,
                                          copy.image_offset.x, copy.image_subresource.base_layer,
                                          copy.image_extent.width,
                                          copy.image_subresource.num_layers, gl_internal_format,
                                          static_cast<GLsizei>(copy.buffer_size), offset);
        } else {
            glTextureSubImage2D(texture.handle, copy.image_subresource.base_mipmap,
                                copy.image_offset.x, copy.image_subresource.base_layer,
                                copy.image_extent.width, copy.image_subresource.num_layers,
                                gl_format, gl_type, offset);
        }
        break;
    case ImageType::e2D:
    case ImageType::Linear:
        if (is_compressed) {
            glCompressedTextureSubImage3D(
                texture.handle, copy.image_subresource.base_mipmap, copy.image_offset.x,
                copy.image_offset.y, copy.image_subresource.base_layer, copy.image_extent.width,
                copy.image_extent.height, copy.image_subresource.num_layers, gl_internal_format,
                static_cast<GLsizei>(copy.buffer_size), offset);
        } else {
            glTextureSubImage3D(texture.handle, copy.image_subresource.base_mipmap,
                                copy.image_offset.x, copy.image_offset.y,
                                copy.image_subresource.base_layer, copy.image_extent.width,
                                copy.image_extent.height, copy.image_subresource.num_layers,
                                gl_format, gl_type, offset);
        }
        break;
    case ImageType::e3D:
        if (is_compressed) {
            glCompressedTextureSubImage3D(
                texture.handle, copy.image_subresource.base_mipmap, copy.image_offset.x,
                copy.image_offset.y, copy.image_offset.z, copy.image_extent.width,
                copy.image_extent.height, copy.image_extent.depth, gl_internal_format,
                static_cast<GLsizei>(copy.buffer_size), offset);
        } else {
            glTextureSubImage3D(texture.handle, copy.image_subresource.base_mipmap,
                                copy.image_offset.x, copy.image_offset.y, copy.image_offset.z,
                                copy.image_extent.width, copy.image_extent.height,
                                copy.image_extent.depth, gl_format, gl_type, offset);
        }
        break;
    default:
        UNREACHABLE();
    }
}

void Image::CopyImageToBuffer(const VideoCommon::BufferImageCopy& copy, size_t buffer_offset) {
    const GLint x_offset = copy.image_offset.x;
    const GLsizei width = copy.image_extent.width;

    const GLint level = copy.image_subresource.base_mipmap;
    const GLsizei buffer_size = static_cast<GLsizei>(copy.buffer_size);
    void* const offset = reinterpret_cast<void*>(copy.buffer_offset + buffer_offset);

    GLint y_offset = 0;
    GLint z_offset = 0;
    GLsizei height = 1;
    GLsizei depth = 1;

    switch (info.type) {
    case ImageType::e1D:
        y_offset = copy.image_subresource.base_layer;
        height = copy.image_subresource.num_layers;
        break;
    case ImageType::e2D:
    case ImageType::Linear:
        y_offset = copy.image_offset.y;
        z_offset = copy.image_subresource.base_layer;
        height = copy.image_extent.height;
        depth = copy.image_subresource.num_layers;
        break;
    case ImageType::e3D:
        y_offset = copy.image_offset.y;
        z_offset = copy.image_offset.z;
        height = copy.image_extent.height;
        depth = copy.image_extent.depth;
        break;
    default:
        UNREACHABLE();
    }
    // Compressed formats don't have a pixel format or type
    const bool is_compressed = gl_format == GL_NONE;
    if (is_compressed) {
        glGetCompressedTextureSubImage(texture.handle, level, x_offset, y_offset, z_offset, width,
                                       height, depth, buffer_size, offset);
    } else {
        glGetTextureSubImage(texture.handle, level, x_offset, y_offset, z_offset, width, height,
                             depth, gl_format, gl_type, buffer_size, offset);
    }
}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageViewInfo& info,
                     ImageId image_id, Image& image)
    : VideoCommon::ImageViewBase{info, image.info, image_id} {
    if (runtime.device.HasASTC() || !IsPixelFormatASTC(info.format)) {
        internal_format = GetFormatTuple(format).internal_format;
    } else {
        internal_format = IsPixelFormatSRGB(info.format) ? GL_SRGB8_ALPHA8 : GL_RGBA8;
    }
    VideoCommon::SubresourceRange flatten_range = info.range;
    std::array<GLuint, 2> handles;
    switch (info.type) {
    case ImageViewType::e1DArray:
        flatten_range.extent.layers = 1;
        [[fallthrough]];
    case ImageViewType::e1D:
        glGenTextures(2, handles.data());
        SetupView(image, ImageViewType::e1D, handles[0], info, flatten_range);
        SetupView(image, ImageViewType::e1DArray, handles[1], info, info.range);
        break;
    case ImageViewType::e2DArray:
        flatten_range.extent.layers = 1;
        [[fallthrough]];
    case ImageViewType::e2D:
        if (image.info.type == ImageType::e3D) {
            // 2D and 2D array views on a 3D textures are used exclusively for render targets
            ASSERT(info.range.extent.mipmaps == 1);
            glGenTextures(1, handles.data());
            SetupView(image, ImageViewType::e3D, handles[0], info,
                      {
                          .base = {.mipmap = info.range.base.mipmap, .layer = 0},
                          .extent = {.mipmaps = 1, .layers = 1},
                      });
            is_slice_view = true;
            break;
        }
        glGenTextures(2, handles.data());
        SetupView(image, ImageViewType::e2D, handles[0], info, flatten_range);
        SetupView(image, ImageViewType::e2DArray, handles[1], info, info.range);
        break;
    case ImageViewType::e3D:
        glGenTextures(1, handles.data());
        SetupView(image, ImageViewType::e3D, handles[0], info, info.range);
        break;
    case ImageViewType::CubeArray:
        flatten_range.extent.layers = 6;
        [[fallthrough]];
    case ImageViewType::Cube:
        glGenTextures(2, handles.data());
        SetupView(image, ImageViewType::Cube, handles[0], info, flatten_range);
        SetupView(image, ImageViewType::CubeArray, handles[1], info, info.range);
        break;
    case ImageViewType::Rect:
        glGenTextures(1, handles.data());
        SetupView(image, ImageViewType::Rect, handles[0], info, info.range);
        break;
    case ImageViewType::Buffer:
        glCreateTextures(GL_TEXTURE_BUFFER, 1, handles.data());
        SetupView(image, ImageViewType::Buffer, handles[0], info, info.range);
        break;
    }
    default_handle = Handle(info.type);
}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::NullImageParams& params)
    : VideoCommon::ImageViewBase{params} {
    static constexpr std::array NULL_SWIZZLE{GL_ZERO, GL_ZERO, GL_ZERO, GL_ZERO};
    std::array<GLuint, VideoCommon::NUM_IMAGE_VIEW_TYPES> handles;
    glGenTextures(static_cast<GLsizei>(handles.size()), handles.data());

    const auto set_view = [&](ImageViewType type, GLuint base, GLint num_layers) {
        const GLuint view = handles[static_cast<size_t>(type)];
        views[static_cast<size_t>(type)].handle = view;
        glTextureView(view, ImageTarget(type), base, GL_R8, 0, 1, 0, num_layers);
        glTextureParameteriv(view, GL_TEXTURE_SWIZZLE_RGBA, NULL_SWIZZLE.data());
    };
    set_view(ImageViewType::e1D, runtime.null_image_1d.handle, 1);
    set_view(ImageViewType::e2D, runtime.null_image_2d.handle, 1);
    set_view(ImageViewType::Cube, runtime.null_image_2d.handle, 6);
    set_view(ImageViewType::e3D, runtime.null_image_3d.handle, 1);
    set_view(ImageViewType::e1DArray, runtime.null_image_1d.handle, 1);
    set_view(ImageViewType::e2DArray, runtime.null_image_2d.handle, 1);
    set_view(ImageViewType::CubeArray, runtime.null_image_2d.handle, 6);
    set_view(ImageViewType::Rect, runtime.null_image_rect.handle, 1);
}

void ImageView::SetupView(Image& image, ImageViewType type, GLuint handle,
                          const VideoCommon::ImageViewInfo& info,
                          VideoCommon::SubresourceRange range) {
    std::string name;
    if (info.type == ImageViewType::Buffer) {
        // TODO: Take offset from buffer cache
        glTextureBufferRange(handle, internal_format, image.buffer.handle, 0,
                             image.guest_size_bytes);
        name = fmt::format("BufferView {} ({})", NameView(*this), handle);
    } else {
        const GLuint parent = image.texture.handle;
        const GLenum target = ImageTarget(type, image.info.num_samples);
        glTextureView(handle, target, parent, internal_format, range.base.mipmap,
                      range.extent.mipmaps, range.base.layer, range.extent.layers);
        ApplySwizzle(handle, format, info.Swizzle());
        name = fmt::format("ImageView {} ({})", NameView(*this), handle);
    }
    glObjectLabel(GL_TEXTURE, handle, static_cast<GLsizei>(name.size()), name.data());

    views[static_cast<size_t>(type)].handle = handle;
}

Sampler::Sampler(TextureCacheRuntime&, const TSCEntry& config) {
    const GLenum compare_mode = config.depth_compare_enabled ? GL_COMPARE_REF_TO_TEXTURE : GL_NONE;
    const GLenum compare_func = MaxwellToGL::DepthCompareFunc(config.depth_compare_func);
    const GLenum mag = MaxwellToGL::TextureFilterMode(config.mag_filter, TextureMipmapFilter::None);
    const GLenum min = MaxwellToGL::TextureFilterMode(config.min_filter, config.mipmap_filter);
    const GLenum reduction_filter = MaxwellToGL::ReductionFilter(config.reduction_filter);

    UNIMPLEMENTED_IF(config.cubemap_anisotropy != 1);
    UNIMPLEMENTED_IF(config.cubemap_interface_filtering != 1);
    UNIMPLEMENTED_IF(config.float_coord_normalization != 0);

    sampler.Create();
    const GLuint handle = sampler.handle;
    glSamplerParameteri(handle, GL_TEXTURE_WRAP_S, MaxwellToGL::WrapMode(config.wrap_u));
    glSamplerParameteri(handle, GL_TEXTURE_WRAP_T, MaxwellToGL::WrapMode(config.wrap_v));
    glSamplerParameteri(handle, GL_TEXTURE_WRAP_R, MaxwellToGL::WrapMode(config.wrap_p));
    glSamplerParameteri(handle, GL_TEXTURE_COMPARE_MODE, compare_mode);
    glSamplerParameteri(handle, GL_TEXTURE_COMPARE_FUNC, compare_func);
    glSamplerParameteri(handle, GL_TEXTURE_MAG_FILTER, mag);
    glSamplerParameteri(handle, GL_TEXTURE_MIN_FILTER, min);
    glSamplerParameterf(handle, GL_TEXTURE_LOD_BIAS, config.LodBias());
    glSamplerParameterf(handle, GL_TEXTURE_MIN_LOD, config.MinLod());
    glSamplerParameterf(handle, GL_TEXTURE_MAX_LOD, config.MaxLod());
    glSamplerParameterfv(handle, GL_TEXTURE_BORDER_COLOR, config.BorderColor().data());

    if (GLAD_GL_ARB_texture_filter_anisotropic || GLAD_GL_EXT_texture_filter_anisotropic) {
        glSamplerParameterf(handle, GL_TEXTURE_MAX_ANISOTROPY, config.MaxAnisotropy());
    } else {
        LOG_WARNING(Render_OpenGL, "GL_ARB_texture_filter_anisotropic is required");
    }
    if (GLAD_GL_ARB_texture_filter_minmax || GLAD_GL_EXT_texture_filter_minmax) {
        glSamplerParameteri(handle, GL_TEXTURE_REDUCTION_MODE_ARB, reduction_filter);
    } else if (reduction_filter != GL_WEIGHTED_AVERAGE_ARB) {
        LOG_WARNING(Render_OpenGL, "GL_ARB_texture_filter_minmax is required");
    }

    const std::string name = fmt::format("Sampler 0x{:x}", std::hash<TSCEntry>{}(config));
    glObjectLabel(GL_SAMPLER, handle, static_cast<GLsizei>(name.size()), name.data());
}

Framebuffer::Framebuffer(TextureCacheRuntime&, const VideoCommon::SlotVector<Image>&,
                         std::span<ImageView*, NUM_RT> color_buffers, ImageView* depth_buffer,
                         std::array<u8, NUM_RT> draw_buffers, VideoCommon::Extent2D size) {
    // Bind to READ_FRAMEBUFFER to stop Nvidia's driver from creating an EXT_framebuffer instead of
    // a core framebuffer. EXT framebuffer attachments have to match in size and can be shared
    // across contexts. yuzu doesn't share framebuffers across contexts and we need attachments with
    // mismatching size, this is why core framebuffers are preferred.
    GLuint handle;
    glGenFramebuffers(1, &handle);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, handle);

    GLsizei num_buffers = 0;
    std::array<GLenum, NUM_RT> gl_draw_buffers;
    gl_draw_buffers.fill(GL_NONE);

    for (size_t index = 0; index < color_buffers.size(); ++index) {
        const ImageView* const image_view = color_buffers[index];
        if (!image_view) {
            continue;
        }
        gl_draw_buffers[index] = GL_COLOR_ATTACHMENT0 + draw_buffers[index];
        num_buffers = static_cast<GLsizei>(index + 1);

        const GLenum attachment = static_cast<GLenum>(GL_COLOR_ATTACHMENT0 + index);
        AttachTexture(handle, attachment, image_view);
    }

    std::string_view debug_prefix = "C";

    if (const ImageView* const image_view = depth_buffer; image_view) {
        const GLenum attachment = AttachmentType(image_view->format);
        debug_prefix = DepthStencilDebugName(attachment);
        AttachTexture(handle, attachment, image_view);
    }

    if (num_buffers > 1) {
        glNamedFramebufferDrawBuffers(handle, num_buffers, gl_draw_buffers.data());
    } else if (num_buffers > 0) {
        glNamedFramebufferDrawBuffer(handle, gl_draw_buffers[0]);
    } else {
        glNamedFramebufferDrawBuffer(handle, GL_NONE);
    }

    glNamedFramebufferParameteri(handle, GL_FRAMEBUFFER_DEFAULT_WIDTH, size.width);
    glNamedFramebufferParameteri(handle, GL_FRAMEBUFFER_DEFAULT_HEIGHT, size.height);
    // TODO
    // glNamedFramebufferParameteri(handle, GL_FRAMEBUFFER_DEFAULT_LAYERS, ...);
    // glNamedFramebufferParameteri(handle, GL_FRAMEBUFFER_DEFAULT_SAMPLES, ...);
    // glNamedFramebufferParameteri(handle, GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS, ...);

    const std::string name =
        fmt::format("Framebuffer {}{} ({})", debug_prefix, num_buffers, handle);
    glObjectLabel(GL_FRAMEBUFFER, handle, static_cast<GLsizei>(name.size()), name.data());

    framebuffer.handle = handle;
}

} // namespace OpenGL
