// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <list>
#include <memory>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include <boost/icl/interval_map.hpp>

#include "common/assert.h"
#include "common/common_types.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/surface.h"

namespace Core {
class System;
}

namespace Tegra::Texture {
struct FullTextureInfo;
}

namespace VideoCore {
class RasterizerInterface;
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

    /// Creates SurfaceParams from a Fermi2D surface configuration
    static SurfaceParams CreateForFermiCopySurface(
        const Tegra::Engines::Fermi2D::Regs::Surface& config);

    std::map<u64, std::pair<u32, u32>> CreateViewOffsetMap() const;

    u32 GetMipWidth(u32 level) const;

    u32 GetMipHeight(u32 level) const;

    u32 GetMipDepth(u32 level) const;

    bool IsLayered() const;

    u32 GetMipBlockHeight(u32 level) const;

    u32 GetMipBlockDepth(u32 level) const;

    std::size_t GetGuestMipmapLevelOffset(u32 level) const;

    std::size_t GetHostMipmapLevelOffset(u32 level) const;

    std::size_t GetGuestLayerMemorySize() const;

    std::size_t GetHostLayerSize(u32 level) const;

    bool IsFamiliar(const SurfaceParams& view_params) const;

    bool IsViewValid(const SurfaceParams& view_params, u32 layer, u32 level) const;

    std::size_t Hash() const;

    bool operator==(const SurfaceParams& rhs) const;

    bool is_tiled;
    u32 block_width;
    u32 block_height;
    u32 block_depth;
    u32 tile_width_spacing;
    u32 width;
    u32 height;
    u32 depth;
    u32 pitch;
    u32 unaligned_height;
    u32 num_levels;
    VideoCore::Surface::PixelFormat pixel_format;
    VideoCore::Surface::ComponentType component_type;
    VideoCore::Surface::SurfaceType type;
    VideoCore::Surface::SurfaceTarget target;

    // Cached data
    std::size_t guest_size_in_bytes;
    std::size_t host_size_in_bytes;
    u32 num_layers;

private:
    void CalculateCachedValues();

    std::size_t GetInnerMipmapMemorySize(u32 level, bool as_host_size, bool layer_only,
                                         bool uncompressed) const;

    std::size_t GetInnerMemorySize(bool as_host_size, bool layer_only, bool uncompressed) const;

    bool IsDimensionValid(const SurfaceParams& view_params, u32 level) const;

    bool IsDepthValid(const SurfaceParams& view_params, u32 level) const;

    bool IsInBounds(const SurfaceParams& view_params, u32 layer, u32 level) const;
};

struct ViewKey {
    std::size_t Hash() const;

    bool operator==(const ViewKey& rhs) const;

    u32 base_layer{};
    u32 num_layers{};
    u32 base_level{};
    u32 num_levels{};
};

} // namespace VideoCommon

namespace std {

template <>
struct hash<VideoCommon::SurfaceParams> {
    std::size_t operator()(const VideoCommon::SurfaceParams& k) const {
        return k.Hash();
    }
};

template <>
struct hash<VideoCommon::ViewKey> {
    std::size_t operator()(const VideoCommon::ViewKey& k) const {
        return k.Hash();
    }
};

} // namespace std

namespace VideoCommon {

template <typename TView, typename TExecutionContext>
class SurfaceBase {
    static_assert(std::is_trivially_copyable<TExecutionContext>::value);

public:
    SurfaceBase(const SurfaceParams& params)
        : params{params}, view_offset_map{params.CreateViewOffsetMap()} {}

    virtual void LoadBuffer() = 0;

    virtual TExecutionContext FlushBuffer(TExecutionContext exctx) = 0;

    virtual TExecutionContext UploadTexture(TExecutionContext exctx) = 0;

    TView* TryGetView(VAddr view_address, const SurfaceParams& view_params) {
        if (view_address < address || !params.IsFamiliar(view_params)) {
            // It can't be a view if it's in a prior address.
            return {};
        }

        const auto relative_offset = static_cast<u64>(view_address - address);
        const auto it = view_offset_map.find(relative_offset);
        if (it == view_offset_map.end()) {
            // Couldn't find an aligned view.
            return {};
        }
        const auto [layer, level] = it->second;

        if (!params.IsViewValid(view_params, layer, level)) {
            return {};
        }

        return GetView(layer, view_params.num_layers, level, view_params.num_levels);
    }

    VAddr GetAddress() const {
        return address;
    }

    std::size_t GetSizeInBytes() const {
        return params.guest_size_in_bytes;
    }

    void MarkAsModified(bool is_modified_) {
        is_modified = is_modified_;
    }

    const SurfaceParams& GetSurfaceParams() const {
        return params;
    }

    TView* GetView(VAddr view_address, const SurfaceParams& view_params) {
        TView* view = TryGetView(view_address, view_params);
        ASSERT(view != nullptr);
        return view;
    }

    void Register(VAddr address_) {
        ASSERT(!is_registered);
        is_registered = true;
        address = address_;
    }

    void Unregister() {
        ASSERT(is_registered);
        is_registered = false;
    }

    bool IsRegistered() const {
        return is_registered;
    }

protected:
    virtual std::unique_ptr<TView> CreateView(const ViewKey& view_key) = 0;

    bool IsModified() const {
        return is_modified;
    }

    const SurfaceParams params;

private:
    TView* GetView(u32 base_layer, u32 num_layers, u32 base_level, u32 num_levels) {
        const ViewKey key{base_layer, num_layers, base_level, num_levels};
        const auto [entry, is_cache_miss] = views.try_emplace(key);
        auto& view = entry->second;
        if (is_cache_miss) {
            view = CreateView(key);
        }
        return view.get();
    }

    const std::map<u64, std::pair<u32, u32>> view_offset_map;

    std::unordered_map<ViewKey, std::unique_ptr<TView>> views;

    VAddr address{};
    bool is_modified{};
    bool is_registered{};
};

template <typename TSurface, typename TView, typename TExecutionContext>
class TextureCache {
    static_assert(std::is_trivially_copyable<TExecutionContext>::value);
    using ResultType = std::tuple<TView*, TExecutionContext>;
    using IntervalMap = boost::icl::interval_map<VAddr, std::set<TSurface*>>;
    using IntervalType = typename IntervalMap::interval_type;

public:
    TextureCache(Core::System& system, VideoCore::RasterizerInterface& rasterizer)
        : system{system}, rasterizer{rasterizer} {}
    ~TextureCache() = default;

    void InvalidateRegion(VAddr address, std::size_t size) {
        for (TSurface* surface : GetSurfacesInRegion(address, size)) {
            if (!surface->IsRegistered()) {
                // Skip duplicates
                continue;
            }
            Unregister(surface);
        }
    }

    ResultType GetTextureSurface(TExecutionContext exctx,
                                 const Tegra::Texture::FullTextureInfo& config) {
        auto& memory_manager{system.GPU().MemoryManager()};
        const auto cpu_addr{memory_manager.GpuToCpuAddress(config.tic.Address())};
        if (!cpu_addr) {
            return {{}, exctx};
        }
        const auto params = SurfaceParams::CreateForTexture(system, config);
        return GetSurfaceView(exctx, *cpu_addr, params, true);
    }

    ResultType GetDepthBufferSurface(TExecutionContext exctx, bool preserve_contents) {
        const auto& regs{system.GPU().Maxwell3D().regs};
        if (!regs.zeta.Address() || !regs.zeta_enable) {
            return {{}, exctx};
        }

        auto& memory_manager{system.GPU().MemoryManager()};
        const auto cpu_addr{memory_manager.GpuToCpuAddress(regs.zeta.Address())};
        if (!cpu_addr) {
            return {{}, exctx};
        }

        const auto depth_params = SurfaceParams::CreateForDepthBuffer(
            system, regs.zeta_width, regs.zeta_height, regs.zeta.format,
            regs.zeta.memory_layout.block_width, regs.zeta.memory_layout.block_height,
            regs.zeta.memory_layout.block_depth, regs.zeta.memory_layout.type);

        return GetSurfaceView(exctx, *cpu_addr, depth_params, preserve_contents);
    }

    ResultType GetColorBufferSurface(TExecutionContext exctx, std::size_t index,
                                     bool preserve_contents) {
        const auto& regs{system.GPU().Maxwell3D().regs};
        ASSERT(index < Tegra::Engines::Maxwell3D::Regs::NumRenderTargets);

        if (index >= regs.rt_control.count ||
            regs.rt[index].Address() == 0 ||
            regs.rt[index].format == Tegra::RenderTargetFormat::NONE) {
            return {{}, exctx};
        }

        auto& memory_manager{system.GPU().MemoryManager()};
        const auto& config{system.GPU().Maxwell3D().regs.rt[index]};
        const auto cpu_addr{memory_manager.GpuToCpuAddress(
            config.Address() + config.base_layer * config.layer_stride * sizeof(u32))};
        if (!cpu_addr) {
            return {{}, exctx};
        }

        return GetSurfaceView(exctx, *cpu_addr, SurfaceParams::CreateForFramebuffer(system, index),
                              preserve_contents);
    }

    ResultType GetFermiSurface(TExecutionContext exctx,
                               const Tegra::Engines::Fermi2D::Regs::Surface& config) {
        const auto cpu_addr{memory_manager.GpuToCpuAddress(config.Address())};
        ASSERT(cpu_addr);
        return GetSurfaceView(exctx, *cpu_addr, SurfaceParams::CreateForFermiCopySurface(config),
                              true);
    }

    TSurface* TryFindFramebufferSurface(VAddr address) const {
        const auto it = registered_surfaces.find(address);
        return it != registered_surfaces.end() ? *it->second.begin() : nullptr;
    }

protected:
    virtual ResultType TryFastGetSurfaceView(TExecutionContext exctx, VAddr address,
                                             const SurfaceParams& params, bool preserve_contents,
                                             const std::vector<TSurface*>& overlaps) = 0;

    virtual std::unique_ptr<TSurface> CreateSurface(const SurfaceParams& params) = 0;

    void Register(TSurface* surface, VAddr address) {
        surface->Register(address);
        registered_surfaces.add({GetSurfaceInterval(surface), {surface}});
        rasterizer.UpdatePagesCachedCount(surface->GetAddress(), surface->GetSizeInBytes(), 1);
    }

    void Unregister(TSurface* surface) {
        surface->Unregister();
        registered_surfaces.subtract({GetSurfaceInterval(surface), {surface}});
        rasterizer.UpdatePagesCachedCount(surface->GetAddress(), surface->GetSizeInBytes(), -1);
    }

    TSurface* GetUncachedSurface(const SurfaceParams& params) {
        if (TSurface* surface = TryGetReservedSurface(params); surface)
            return surface;
        // No reserved surface available, create a new one and reserve it
        auto new_surface = CreateSurface(params);
        TSurface* surface = new_surface.get();
        ReserveSurface(params, std::move(new_surface));
        return surface;
    }

    Core::System& system;

private:
    ResultType GetSurfaceView(TExecutionContext exctx, VAddr address, const SurfaceParams& params,
                              bool preserve_contents) {
        const std::vector<TSurface*> overlaps =
            GetSurfacesInRegion(address, params.guest_size_in_bytes);
        if (overlaps.empty()) {
            return LoadSurfaceView(exctx, address, params, preserve_contents);
        }

        if (overlaps.size() == 1) {
            if (TView* view = overlaps[0]->TryGetView(address, params); view)
                return {view, exctx};
        }

        TView* fast_view;
        std::tie(fast_view, exctx) =
            TryFastGetSurfaceView(exctx, address, params, preserve_contents, overlaps);

        for (TSurface* surface : overlaps) {
            if (!fast_view) {
                // Flush even when we don't care about the contents, to preserve memory not written
                // by the new surface.
                exctx = surface->FlushBuffer(exctx);
            }
            Unregister(surface);
        }

        if (fast_view) {
            return {fast_view, exctx};
        }

        return LoadSurfaceView(exctx, address, params, preserve_contents);
    }

    ResultType LoadSurfaceView(TExecutionContext exctx, VAddr address, const SurfaceParams& params,
                               bool preserve_contents) {
        TSurface* new_surface = GetUncachedSurface(params);
        Register(new_surface, address);
        if (preserve_contents) {
            exctx = LoadSurface(exctx, new_surface);
        }
        return {new_surface->GetView(address, params), exctx};
    }

    TExecutionContext LoadSurface(TExecutionContext exctx, TSurface* surface) {
        surface->LoadBuffer();
        exctx = surface->UploadTexture(exctx);
        surface->MarkAsModified(false);
        return exctx;
    }

    std::vector<TSurface*> GetSurfacesInRegion(VAddr address, std::size_t size) const {
        if (size == 0) {
            return {};
        }
        const IntervalType interval{address, address + size};

        std::vector<TSurface*> surfaces;
        for (auto& pair : boost::make_iterator_range(registered_surfaces.equal_range(interval))) {
            surfaces.push_back(*pair.second.begin());
        }
        return surfaces;
    }

    void ReserveSurface(const SurfaceParams& params, std::unique_ptr<TSurface> surface) {
        surface_reserve[params].push_back(std::move(surface));
    }

    TSurface* TryGetReservedSurface(const SurfaceParams& params) {
        auto search = surface_reserve.find(params);
        if (search == surface_reserve.end()) {
            return {};
        }
        for (auto& surface : search->second) {
            if (!surface->IsRegistered()) {
                return surface.get();
            }
        }
        return {};
    }

    IntervalType GetSurfaceInterval(TSurface* surface) const {
        return IntervalType::right_open(surface->GetAddress(),
                                        surface->GetAddress() + surface->GetSizeInBytes());
    }

    VideoCore::RasterizerInterface& rasterizer;

    IntervalMap registered_surfaces;

    /// The surface reserve is a "backup" cache, this is where we put unique surfaces that have
    /// previously been used. This is to prevent surfaces from being constantly created and
    /// destroyed when used with different surface parameters.
    std::unordered_map<SurfaceParams, std::list<std::unique_ptr<TSurface>>> surface_reserve;
};

} // namespace VideoCommon