// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <map>
#include <memory>
#include <tuple>
#include "common/assert.h"
#include "common/common_types.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/math_util.h"
#include "video_core/rasterizer_cache.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_image.h"
#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"
#include "video_core/surface.h"
#include "video_core/textures/decoders.h"
#include "video_core/textures/texture.h"

namespace Core {
class System;
} // namespace Core

namespace Vulkan {

class RasterizerVulkan;
class VKDevice;
class VKResourceManager;

using VideoCore::Surface::ComponentType;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::SurfaceTarget;
using VideoCore::Surface::SurfaceType;

class CachedSurface;
using Surface = std::shared_ptr<CachedSurface>;
using SurfaceSurfaceRect_Tuple = std::tuple<Surface, Surface, MathUtil::Rectangle<u32>>;

struct SurfaceParams {
    /// Creates SurfaceParams from a texture configuration
    static SurfaceParams CreateForTexture(Core::System& system,
                                          const Tegra::Texture::FullTextureInfo& config,
                                          const VKShader::SamplerEntry& entry);

    /// Creates SurfaceParams for a depth buffer configuration
    static SurfaceParams CreateForDepthBuffer(
        Core::System& system, u32 zeta_width, u32 zeta_height, Tegra::GPUVAddr zeta_address,
        Tegra::DepthFormat format, u32 block_width, u32 block_height, u32 block_depth,
        Tegra::Engines::Maxwell3D::Regs::InvMemoryLayout type);

    /// Creates SurfaceParams from a framebuffer configuration
    static SurfaceParams CreateForFramebuffer(Core::System& system, std::size_t index);

    /// Returns the total size of this surface in bytes, adjusted for compression
    std::size_t SizeInBytesRaw(bool ignore_tiled = false) const {
        const u32 compression_factor{GetCompressionFactor(pixel_format)};
        const u32 bytes_per_pixel{GetBytesPerPixel(pixel_format)};
        const size_t uncompressed_size{
            Tegra::Texture::CalculateSize((ignore_tiled ? false : is_tiled), bytes_per_pixel, width,
                                          height, depth, block_height, block_depth)};

        // Divide by compression_factor^2, as height and width are factored by this
        return uncompressed_size / (compression_factor * compression_factor);
    }

    /// Returns the size of this surface as an Vulkan texture in bytes
    std::size_t SizeInBytesVK() const {
        return SizeInBytesRaw(true);
    }

    /// Checks if surfaces are compatible for caching
    bool IsCompatibleSurface(const SurfaceParams& other) const {
        return std::tie(pixel_format, type, width, height, target, depth, block_height, block_depth,
                        tile_width_spacing) == std::tie(other.pixel_format, other.type, other.width,
                                                        other.height, other.target, other.depth,
                                                        other.block_height, other.block_depth,
                                                        other.tile_width_spacing);
    }

    /// Initializes parameters for caching, should be called after everything has been initialized
    void InitCacheParameters(Core::System& system, Tegra::GPUVAddr gpu_addr);

    vk::ImageCreateInfo CreateInfo(const VKDevice& device) const;

    bool is_tiled;
    u32 block_width;
    u32 block_height;
    u32 block_depth;
    u32 tile_width_spacing;
    PixelFormat pixel_format;
    ComponentType component_type;
    SurfaceType type;
    u32 width;
    u32 height;
    u32 depth;
    u32 unaligned_height;
    SurfaceTarget target;
    // Parameters used for caching
    VAddr addr;
    Tegra::GPUVAddr gpu_addr;
    std::size_t size_in_bytes;
    std::size_t size_in_bytes_vk;
};

class CachedSurface final : public RasterizerCacheObject, public VKImage {
public:
    explicit CachedSurface(Core::System& system, const VKDevice& device,
                           VKResourceManager& resource_manager, VKMemoryManager& memory_manager,
                           const SurfaceParams& params);
    ~CachedSurface();

    // Read/Write data in Switch memory to/from vk_buffer
    void LoadVKBuffer();
    VKExecutionContext FlushVKBuffer(VKExecutionContext exctx);

    // Upload data in vk_buffer to this surface's texture
    VKExecutionContext UploadVKTexture(VKExecutionContext exctx);

    VAddr GetAddr() const override {
        return params.addr;
    }

    std::size_t GetSizeInBytes() const override {
        return cached_size_in_bytes;
    }

    void Flush() override {
        UNIMPLEMENTED();
    }

    const SurfaceParams& GetSurfaceParams() const {
        return params;
    }

private:
    const VKDevice& device;
    VKResourceManager& resource_manager;
    VKMemoryManager& memory_manager;
    const SurfaceParams params;
    const std::size_t buffer_size;

    vk::Image image;
    VKMemoryCommit image_commit;

    UniqueBuffer buffer;
    VKMemoryCommit buffer_commit;
    u8* vk_buffer{};

    UniqueImageView image_view;

    std::size_t cached_size_in_bytes;
};

} // namespace Vulkan

/// Hashable variation of SurfaceParams, used for a key in the surface cache
struct SurfaceReserveKey : Common::HashableStruct<Vulkan::SurfaceParams> {
    static SurfaceReserveKey Create(const Vulkan::SurfaceParams& params) {
        SurfaceReserveKey res;
        res.state = params;
        res.state.gpu_addr = {}; // Ignore GPU vaddr in caching
        // res.state.rt = {};       // Ignore rt config in caching
        return res;
    }
};
namespace std {
template <>
struct hash<SurfaceReserveKey> {
    std::size_t operator()(const SurfaceReserveKey& k) const {
        return k.Hash();
    }
};
} // namespace std

namespace Vulkan {

class VKRasterizerCache final : public RasterizerCache<Surface> {
public:
    explicit VKRasterizerCache(Core::System& system, RasterizerVulkan& rasterizer,
                               const VKDevice& device, VKResourceManager& resource_manager,
                               VKMemoryManager& memory_manager);
    ~VKRasterizerCache();

    /// Get a surface based on the texture configuration
    [[nodiscard]] std::tuple<Surface, VKExecutionContext> GetTextureSurface(
        VKExecutionContext exctx, const Tegra::Texture::FullTextureInfo& config,
        const VKShader::SamplerEntry& entry);

    /// Get the depth surface based on the framebuffer configuration
    [[nodiscard]] std::tuple<Surface, VKExecutionContext> GetDepthBufferSurface(
        VKExecutionContext exctx, bool preserve_contents);

    /// Get the color surface based on the framebuffer configuration and the specified render target
    [[nodiscard]] std::tuple<Surface, VKExecutionContext> GetColorBufferSurface(
        VKExecutionContext exctx, std::size_t index, bool preserve_contents);

    /// Tries to find a framebuffer using on the provided CPU address
    Surface TryFindFramebufferSurface(VAddr addr) const;

private:
    Core::System& system;
    const VKDevice& device;
    VKResourceManager& resource_manager;
    VKMemoryManager& memory_manager;

    [[nodiscard]] VKExecutionContext LoadSurface(VKExecutionContext exctx, const Surface& surface);

    [[nodiscard]] std::tuple<Surface, VKExecutionContext> GetSurface(VKExecutionContext exctx,
                                                                     const SurfaceParams& params,
                                                                     bool preserve_contents = true);

    /// Gets an uncached surface, creating it if need be
    Surface GetUncachedSurface(const SurfaceParams& params);

    /// Recreates a surface with new parameters
    [[nodiscard]] std::tuple<Surface, VKExecutionContext> RecreateSurface(
        VKExecutionContext exctx, const Surface& old_surface, const SurfaceParams& new_params);

    /// Reserves a unique surface that can be reused later
    void ReserveSurface(const Surface& surface);

    /// Tries to get a reserved surface for the specified parameters
    Surface TryGetReservedSurface(const SurfaceParams& params);

    /// The surface reserve is a "backup" cache, this is where we put unique surfaces that have
    /// previously been used. This is to prevent surfaces from being constantly created and
    /// destroyed when used with different surface parameters.
    std::unordered_map<SurfaceReserveKey, Surface> surface_reserve;
};

} // namespace Vulkan