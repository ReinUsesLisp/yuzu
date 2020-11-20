// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <bit>

#include <glad/glad.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "video_core/host_shaders/block_linear_unswizzle_2d_comp.h"
#include "video_core/host_shaders/block_linear_unswizzle_3d_comp.h"
#include "video_core/host_shaders/opengl_copy_bc4_comp.h"
#include "video_core/host_shaders/pitch_unswizzle_comp.h"
#include "video_core/renderer_opengl/gl_resource_manager.h"
#include "video_core/renderer_opengl/gl_shader_manager.h"
#include "video_core/renderer_opengl/gl_texture_cache.h"
#include "video_core/renderer_opengl/util_shaders.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"
#include "video_core/texture_cache/util.h"
#include "video_core/textures/decoders.h"

namespace OpenGL {

using namespace HostShaders;

using Tegra::Texture::GOB_SIZE_SHIFT;
using Tegra::Texture::GOB_SIZE_X;
using Tegra::Texture::GOB_SIZE_X_SHIFT;
using Tegra::Texture::GOB_SIZE_Y_SHIFT;
using VideoCommon::Extent3D;
using VideoCommon::ImageCopy;
using VideoCommon::ImageType;
using VideoCommon::SwizzleParameters;
using VideoCore::Surface::BytesPerBlock;

namespace {

OGLProgram MakeProgram(std::string_view source) {
    OGLShader shader;
    shader.Create(source, GL_COMPUTE_SHADER);

    OGLProgram program;
    program.Create(true, false, shader.handle);
    return program;
}

} // Anonymous namespace

UtilShaders::UtilShaders(ProgramManager& program_manager_)
    : program_manager{program_manager_},
      block_linear_unswizzle_2d_program(MakeProgram(BLOCK_LINEAR_UNSWIZZLE_2D_COMP)),
      block_linear_unswizzle_3d_program(MakeProgram(BLOCK_LINEAR_UNSWIZZLE_3D_COMP)),
      pitch_unswizzle_program(MakeProgram(PITCH_UNSWIZZLE_COMP)),
      copy_bc4_program(MakeProgram(OPENGL_COPY_BC4_COMP)) {
    const auto swizzle_table = Tegra::Texture::MakeSwizzleTable();
    swizzle_table_buffer.Create();
    glNamedBufferStorage(swizzle_table_buffer.handle, sizeof(swizzle_table), &swizzle_table, 0);
}

UtilShaders::~UtilShaders() = default;

void UtilShaders::BlockLinearUpload2D(Image& image, const ImageBufferMap& map, size_t buffer_offset,
                                      std::span<const SwizzleParameters> swizzles) {
    static constexpr Extent3D WORKGROUP_SIZE{32, 32, 1};
    static constexpr GLuint BINDING_SWIZZLE_BUFFER = 0;
    static constexpr GLuint BINDING_INPUT_BUFFER = 1;
    static constexpr GLuint BINDING_OUTPUT_IMAGE = 0;
    static constexpr GLuint LOC_ORIGIN = 0;
    static constexpr GLuint LOC_DESTINATION = 1;
    static constexpr GLuint LOC_BYTES_PER_BLOCK = 2;
    static constexpr GLuint LOC_LAYER_STRIDE = 3;
    static constexpr GLuint LOC_BLOCK_SIZE = 4;
    static constexpr GLuint LOC_X_SHIFT = 5;
    static constexpr GLuint LOC_BLOCK_HEIGHT = 6;
    static constexpr GLuint LOC_BLOCK_HEIGHT_MASK = 7;

    const u32 bytes_per_block = BytesPerBlock(image.info.format);
    const u32 bytes_per_block_log2 = std::countr_zero(bytes_per_block);

    program_manager.BindHostCompute(block_linear_unswizzle_2d_program.handle);
    glFlushMappedNamedBufferRange(map.Handle(), buffer_offset, image.guest_size_bytes);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_SWIZZLE_BUFFER, swizzle_table_buffer.handle);
    glUniform3ui(LOC_ORIGIN, 0, 0, 0);     // TODO
    glUniform3i(LOC_DESTINATION, 0, 0, 0); // TODO
    glUniform1ui(LOC_BYTES_PER_BLOCK, bytes_per_block_log2);
    glUniform1ui(LOC_LAYER_STRIDE, image.info.layer_stride);
    for (const SwizzleParameters& swizzle : swizzles) {
        const Extent3D block = swizzle.block;
        const Extent3D num_tiles = swizzle.num_tiles;
        const size_t offset = swizzle.buffer_offset + buffer_offset;

        const u32 num_dispatches_x = Common::DivCeil(num_tiles.width, WORKGROUP_SIZE.width);
        const u32 num_dispatches_y = Common::DivCeil(num_tiles.height, WORKGROUP_SIZE.height);

        const u32 stride_alignment = CalculateLevelStrideAlignment(image.info, swizzle.level);
        const u32 stride = Common::AlignBits(num_tiles.width, stride_alignment) * bytes_per_block;

        const u32 gobs_in_x = (stride + GOB_SIZE_X - 1) >> GOB_SIZE_X_SHIFT;
        const u32 block_size = gobs_in_x << (GOB_SIZE_SHIFT + block.height + block.depth);

        const u32 block_height_mask = (1U << block.height) - 1;
        const u32 x_shift = GOB_SIZE_SHIFT + block.height + block.depth;

        glUniform1ui(LOC_BLOCK_SIZE, block_size);
        glUniform1ui(LOC_X_SHIFT, x_shift);
        glUniform1ui(LOC_BLOCK_HEIGHT, block.height);
        glUniform1ui(LOC_BLOCK_HEIGHT_MASK, block_height_mask);
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, BINDING_INPUT_BUFFER, map.Handle(), offset,
                          image.guest_size_bytes - swizzle.buffer_offset);
        glBindImageTexture(BINDING_OUTPUT_IMAGE, image.Handle(), swizzle.level, GL_TRUE, 0,
                           GL_WRITE_ONLY, StoreFormat(bytes_per_block));
        glDispatchCompute(num_dispatches_x, num_dispatches_y, image.info.resources.layers);
    }
    program_manager.RestoreGuestCompute();
}

void UtilShaders::BlockLinearUpload3D(Image& image, const ImageBufferMap& map, size_t buffer_offset,
                                      std::span<const SwizzleParameters> swizzles) {
    static constexpr Extent3D WORKGROUP_SIZE{16, 8, 8};

    static constexpr GLuint BINDING_SWIZZLE_BUFFER = 0;
    static constexpr GLuint BINDING_INPUT_BUFFER = 1;
    static constexpr GLuint BINDING_OUTPUT_IMAGE = 0;

    static constexpr GLuint LOC_ORIGIN = 0;
    static constexpr GLuint LOC_DESTINATION = 1;
    static constexpr GLuint LOC_BYTES_PER_BLOCK = 2;
    static constexpr GLuint SLICE_SIZE_LOC = 3;
    static constexpr GLuint LOC_BLOCK_SIZE = 4;
    static constexpr GLuint LOC_X_SHIFT = 5;
    static constexpr GLuint LOC_BLOCK_HEIGHT = 6;
    static constexpr GLuint LOC_BLOCK_HEIGHT_MASK = 7;
    static constexpr GLuint BLOCK_DEPTH_LOC = 8;
    static constexpr GLuint BLOCK_DEPTH_MASK_LOC = 9;

    const u32 bytes_per_block = BytesPerBlock(image.info.format);
    const u32 bytes_per_block_log2 = std::countr_zero(bytes_per_block);

    glFlushMappedNamedBufferRange(map.Handle(), buffer_offset, image.guest_size_bytes);
    program_manager.BindHostCompute(block_linear_unswizzle_3d_program.handle);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_SWIZZLE_BUFFER, swizzle_table_buffer.handle);
    glUniform3ui(LOC_ORIGIN, 0, 0, 0);     // TODO
    glUniform3i(LOC_DESTINATION, 0, 0, 0); // TODO
    glUniform1ui(LOC_BYTES_PER_BLOCK, bytes_per_block_log2);
    for (const SwizzleParameters& swizzle : swizzles) {
        const Extent3D block = swizzle.block;
        const Extent3D num_tiles = swizzle.num_tiles;
        const size_t offset = swizzle.buffer_offset + buffer_offset;

        const u32 num_dispatches_x = Common::DivCeil(num_tiles.width, WORKGROUP_SIZE.width);
        const u32 num_dispatches_y = Common::DivCeil(num_tiles.height, WORKGROUP_SIZE.height);
        const u32 num_dispatches_z = Common::DivCeil(num_tiles.depth, WORKGROUP_SIZE.depth);

        const u32 stride_alignment = CalculateLevelStrideAlignment(image.info, swizzle.level);
        const u32 stride = Common::AlignBits(num_tiles.width, stride_alignment) * bytes_per_block;

        const u32 gobs_in_x = (stride + GOB_SIZE_X - 1) >> GOB_SIZE_X_SHIFT;
        const u32 block_size = gobs_in_x << (GOB_SIZE_SHIFT + block.height + block.depth);
        const u32 slice_size =
            Common::DivCeilLog2(num_tiles.height, block.height + GOB_SIZE_Y_SHIFT) * block_size;

        const u32 block_height_mask = (1U << block.height) - 1;
        const u32 block_depth_mask = (1U << block.depth) - 1;
        const u32 x_shift = GOB_SIZE_SHIFT + block.height + block.depth;

        glUniform1ui(SLICE_SIZE_LOC, slice_size);
        glUniform1ui(LOC_BLOCK_SIZE, block_size);
        glUniform1ui(LOC_X_SHIFT, x_shift);
        glUniform1ui(LOC_BLOCK_HEIGHT, block.height);
        glUniform1ui(LOC_BLOCK_HEIGHT_MASK, block_height_mask);
        glUniform1ui(BLOCK_DEPTH_LOC, block.depth);
        glUniform1ui(BLOCK_DEPTH_MASK_LOC, block_depth_mask);

        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, BINDING_INPUT_BUFFER, map.Handle(), offset,
                          image.guest_size_bytes - swizzle.buffer_offset);
        glBindImageTexture(BINDING_OUTPUT_IMAGE, image.Handle(), swizzle.level, GL_TRUE, 0,
                           GL_WRITE_ONLY, StoreFormat(bytes_per_block));

        glDispatchCompute(num_dispatches_x, num_dispatches_y, num_dispatches_z);
    }
    program_manager.RestoreGuestCompute();
}

void UtilShaders::PitchUpload(Image& image, const ImageBufferMap& map, size_t buffer_offset,
                              std::span<const SwizzleParameters> swizzles) {
    static constexpr Extent3D WORKGROUP_SIZE{32, 32, 1};
    static constexpr GLuint BINDING_INPUT_BUFFER = 0;
    static constexpr GLuint BINDING_OUTPUT_IMAGE = 0;
    static constexpr GLuint LOC_ORIGIN = 0;
    static constexpr GLuint LOC_DESTINATION = 1;
    static constexpr GLuint LOC_BYTES_PER_BLOCK = 2;
    static constexpr GLuint LOC_PITCH = 3;

    const u32 bytes_per_block = BytesPerBlock(image.info.format);
    const GLenum format = StoreFormat(bytes_per_block);
    const u32 pitch = image.info.pitch;

    UNIMPLEMENTED_IF_MSG(!std::has_single_bit(bytes_per_block),
                         "Non-power of two images are not implemented");

    program_manager.BindHostCompute(pitch_unswizzle_program.handle);
    glFlushMappedNamedBufferRange(map.Handle(), buffer_offset, image.guest_size_bytes);
    glUniform2ui(LOC_ORIGIN, 0, 0);     // TODO
    glUniform2i(LOC_DESTINATION, 0, 0); // TODO
    glUniform1ui(LOC_BYTES_PER_BLOCK, bytes_per_block);
    glUniform1ui(LOC_PITCH, pitch);
    glBindImageTexture(BINDING_OUTPUT_IMAGE, image.Handle(), 0, GL_FALSE, 0, GL_WRITE_ONLY, format);
    for (const SwizzleParameters& swizzle : swizzles) {
        const Extent3D num_tiles = swizzle.num_tiles;
        const size_t offset = swizzle.buffer_offset + buffer_offset;

        const u32 aligned_width = Common::AlignUp(num_tiles.width, WORKGROUP_SIZE.width);
        const u32 aligned_height = Common::AlignUp(num_tiles.height, WORKGROUP_SIZE.height);
        const u32 num_dispatches_x = aligned_width / WORKGROUP_SIZE.width;
        const u32 num_dispatches_y = aligned_height / WORKGROUP_SIZE.height;

        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, BINDING_INPUT_BUFFER, map.Handle(), offset,
                          image.guest_size_bytes - swizzle.buffer_offset);
        glDispatchCompute(num_dispatches_x, num_dispatches_y, 1);
    }
    program_manager.RestoreGuestCompute();
}

void UtilShaders::CopyBC4(Image& dst_image, Image& src_image, std::span<const ImageCopy> copies) {
    static constexpr GLuint BINDING_INPUT_IMAGE = 0;
    static constexpr GLuint BINDING_OUTPUT_IMAGE = 1;
    static constexpr GLuint LOC_SRC_OFFSET = 0;
    static constexpr GLuint LOC_DST_OFFSET = 1;

    program_manager.BindHostCompute(copy_bc4_program.handle);

    for (const ImageCopy& copy : copies) {
        ASSERT(copy.src_subresource.base_layer == 0);
        ASSERT(copy.src_subresource.num_layers == 1);
        ASSERT(copy.dst_subresource.base_layer == 0);
        ASSERT(copy.dst_subresource.num_layers == 1);

        glUniform3ui(LOC_SRC_OFFSET, copy.src_offset.x, copy.src_offset.y, copy.src_offset.z);
        glUniform3ui(LOC_DST_OFFSET, copy.dst_offset.x, copy.dst_offset.y, copy.dst_offset.z);
        glBindImageTexture(BINDING_INPUT_IMAGE, src_image.Handle(), copy.src_subresource.base_level,
                           GL_FALSE, 0, GL_READ_ONLY, GL_RG32UI);
        glBindImageTexture(BINDING_OUTPUT_IMAGE, dst_image.Handle(),
                           copy.dst_subresource.base_level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8UI);
        glDispatchCompute(copy.extent.width, copy.extent.height, copy.extent.depth);
    }
    program_manager.RestoreGuestCompute();
}

GLenum StoreFormat(u32 bytes_per_block) {
    switch (bytes_per_block) {
    case 1:
        return GL_R8UI;
    case 2:
        return GL_R16UI;
    case 4:
        return GL_R32UI;
    case 8:
        return GL_RG32UI;
    case 16:
        return GL_RGBA32UI;
    }
    UNREACHABLE();
    return GL_R8UI;
}

} // namespace OpenGL
