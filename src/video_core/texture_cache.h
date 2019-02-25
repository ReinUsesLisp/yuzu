// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/surface.h"

namespace Core {
class System;
}

namespace Tegra::Texture {
struct FullTextureInfo;
}

namespace VideoCommon {

struct SurfaceParams {
    /// Creates SurfaceParams from a texture configuration
    static SurfaceParams CreateForTexture(Core::System& system,
                                          const Tegra::Texture::FullTextureInfo& config);

    /// Creates SurfaceParams for a depth buffer configuration
    static SurfaceParams CreateForDepthBuffer(
        Core::System& system, u32 zeta_width, u32 zeta_height, Tegra::DepthFormat format,
        u32 block_width, u32 block_height, u32 block_depth,
        Tegra::Engines::Maxwell3D::Regs::InvMemoryLayout type);

    /// Creates SurfaceParams from a framebuffer configuration
    static SurfaceParams CreateForFramebuffer(Core::System& system, std::size_t index);

    u32 GetMipWidth(u32 level) const;

    u32 GetMipHeight(u32 level) const;

    u32 GetMipDepth(u32 level) const;

    u32 GetLayersCount() const;

    bool IsLayered() const;

    u32 GetMipBlockHeight(u32 level) const;

    u32 GetMipBlockDepth(u32 level) const;

    std::size_t GetGuestMipmapLevelOffset(u32 level) const;

    std::size_t GetHostMipmapLevelOffset(u32 level) const;

    std::size_t GetGuestLayerMemorySize() const;

    std::size_t GetHostLayerSize(u32 level) const;

    bool is_tiled;
    u32 block_width;
    u32 block_height;
    u32 block_depth;
    u32 tile_width_spacing;
    VideoCore::Surface::PixelFormat pixel_format;
    VideoCore::Surface::ComponentType component_type;
    VideoCore::Surface::SurfaceType type;
    VideoCore::Surface::SurfaceTarget target;
    u32 width;
    u32 height;
    u32 depth;
    u32 pitch;
    u32 unaligned_height;
    u32 levels_count;

    // Cached data
    std::size_t guest_size_in_bytes;
    std::size_t host_size_in_bytes;

private:
    void CalculateSizes();

    std::size_t GetInnerMipmapMemorySize(u32 level, bool as_host_size, bool layer_only,
                                         bool uncompressed) const;

    std::size_t GetInnerMemorySize(bool as_host_size, bool layer_only, bool uncompressed) const;
};

} // namespace VideoCommon