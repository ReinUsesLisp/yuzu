// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <bitset>
#include <memory>
#include <utility>
#include <vector>

#include <boost/icl/interval.hpp>

#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Core {
class System;
}

namespace Core::Frontend {
class EmuWindow;
}

namespace Vulkan {

struct VKScreenInfo;
class VKFence;
class VKTextureCache;
class VKResourceManager;
class VKMemoryManager;
class VKDevice;
class VKPipelineCache;
class VKBufferCache;
class VKRenderPassCache;
class VKSamplerCache;
class VKGlobalCache;

class CachedView;

using ImageViewsPack = StaticVector<vk::ImageView, Maxwell::NumRenderTargets + 1>;

struct FramebufferCacheKey {
    vk::RenderPass renderpass;
    ImageViewsPack views;
    u32 width;
    u32 height;

    std::size_t Hash() const {
        std::size_t hash = 0;
        boost::hash_combine(hash, static_cast<VkRenderPass>(renderpass));
        for (const auto& view : views)
            boost::hash_combine(hash, static_cast<VkImageView>(view));
        boost::hash_combine(hash, width);
        boost::hash_combine(hash, height);
        return hash;
    }

    bool operator==(const FramebufferCacheKey& rhs) const {
        return std::tie(renderpass, views, width, height) ==
               std::tie(rhs.renderpass, rhs.views, rhs.width, rhs.height);
    }
};

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::FramebufferCacheKey> {
    std::size_t operator()(const Vulkan::FramebufferCacheKey& k) const {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class PipelineState {
public:
    void Reset();

    void AddVertexBinding(vk::Buffer buffer, vk::DeviceSize offset);

    void SetIndexBinding(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type);

    void AddDescriptor(u32 current_binding, vk::DescriptorType descriptor_type, vk::Buffer buffer,
                       u64 offset, std::size_t size);

    void AddDescriptor(u32 current_binding, vk::DescriptorType descriptor_type, vk::Sampler sampler,
                       vk::ImageView image_view, vk::ImageLayout image_layout);

    void UpdateDescriptorSet(const VKDevice& device, vk::DescriptorSet descriptor_set,
                             vk::DescriptorUpdateTemplate descriptor_update_template) const;

    void BindVertexBuffers(vk::CommandBuffer cmdbuf, const vk::DispatchLoaderDynamic& dld) const;

    void BindIndexBuffer(vk::CommandBuffer cmdbuf, const vk::DispatchLoaderDynamic& dld) const;

private:
    std::vector<std::pair<vk::Buffer, vk::DeviceSize>> vertex_bindings;

    vk::Buffer index_buffer{};
    vk::DeviceSize index_offset{};
    vk::IndexType index_type{};

    std::vector<DescriptorUpdateEntry> update_entries;
};

class RasterizerVulkan : public VideoCore::RasterizerInterface {
public:
    explicit RasterizerVulkan(Core::System& system, Core::Frontend::EmuWindow& render_window,
                              VKScreenInfo& screen_info, const VKDevice& device,
                              VKResourceManager& resource_manager, VKMemoryManager& memory_manager,
                              VKScheduler& scheduler);
    ~RasterizerVulkan() override;

    void DrawArrays() override;
    void Clear() override;
    void FlushAll() override;
    void FlushRegion(CacheAddr addr, u64 size) override;
    void InvalidateRegion(CacheAddr addr, u64 size) override;
    void FlushAndInvalidateRegion(CacheAddr addr, u64 size) override;
    bool AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Regs::Surface& src,
                               const Tegra::Engines::Fermi2D::Regs::Surface& dst,
                               const Common::Rectangle<u32>& src_rect,
                               const Common::Rectangle<u32>& dst_rect) override;
    bool AccelerateDisplay(const Tegra::FramebufferConfig& config, VAddr framebuffer_addr,
                           u32 pixel_stride) override;
    bool AccelerateDrawBatch(bool is_indexed_) override;
    void UpdatePagesCachedCount(VAddr addr, u64 size, int delta) override;

    /// Maximum supported size that a constbuffer can have in bytes.
    static constexpr std::size_t MaxConstbufferSize = 0x10000;
    static_assert(MaxConstbufferSize % (4 * sizeof(float)) == 0,
                  "The maximum size of a constbuffer must be a multiple of the size of GLvec4");

private:
    using Texceptions = std::bitset<Maxwell::NumRenderTargets + 1>;

    static constexpr u64 STREAM_BUFFER_SIZE = 128 * 1024 * 1024;
    static constexpr std::size_t ZETA_TEXCEPTION_INDEX = 8;

    void PrepareDraw();

    FixedPipelineState GetFixedPipelineState() const;

    [[nodiscard]] std::tuple<std::array<CachedView*, Maxwell::NumRenderTargets>, VKExecutionContext>
    GetColorAttachments(VKExecutionContext exctx);

    [[nodiscard]] std::tuple<CachedView*, VKExecutionContext> GetZetaAttachment(
        VKExecutionContext exctx);

    [[nodiscard]] std::tuple<vk::Framebuffer, vk::Extent2D, VKExecutionContext>
    ConfigureFramebuffers(
        VKExecutionContext exctx,
        const std::array<CachedView*, Maxwell::NumRenderTargets>& color_attachments,
        CachedView* zeta_attachment, vk::RenderPass renderpass);

    void SetupGeometry(FixedPipelineState& fixed_state);

    [[nodiscard]] std::tuple<Texceptions, VKExecutionContext> SetupShaderDescriptors(
        VKExecutionContext exctx,
        const std::array<CachedView*, Maxwell::NumRenderTargets>& color_attachments,
        CachedView* zeta_attachment, const std::array<Shader, Maxwell::MaxShaderStage>& shaders);

    [[nodiscard]] VKExecutionContext SetupImageTransitions(
        VKExecutionContext exctx, Texceptions texceptions,
        const std::array<CachedView*, Maxwell::NumRenderTargets>& color_attachments,
        CachedView* zeta_attachment);

    void DispatchDraw(VKExecutionContext exctx, vk::PipelineLayout pipeline_layout,
                      vk::DescriptorSet descriptor_set, vk::Pipeline pipeline,
                      vk::RenderPass renderpass, vk::Framebuffer framebuffer,
                      vk::Extent2D render_area);

    void SetupVertexArrays(FixedPipelineState& params);

    void SetupIndexBuffer();

    void SetupConstBuffers(const Shader& shader, Maxwell::ShaderStage stage);

    [[nodiscard]] VKExecutionContext SetupGlobalBuffers(VKExecutionContext exctx,
                                                        const Shader& shader,
                                                        Maxwell::ShaderStage stage);

    [[nodiscard]] std::tuple<Texceptions, VKExecutionContext> SetupTextures(
        VKExecutionContext exctx,
        const std::array<CachedView*, Maxwell::NumRenderTargets>& color_attachments,
        CachedView* zeta_attachment, Texceptions texceptions, const Shader& shader,
        Maxwell::ShaderStage stage);

    std::size_t CalculateVertexArraysSize() const;

    std::size_t CalculateIndexBufferSize() const;

    RenderPassParams GetRenderPassParams(Texceptions texceptions) const;

    void SyncDepthStencil(FixedPipelineState& fixed_state) const;
    void SyncInputAssembly(FixedPipelineState& fixed_state) const;
    void SyncColorBlending(FixedPipelineState& fixed_state) const;
    void SyncViewportState(FixedPipelineState& fixed_state) const;
    void SyncRasterizerState(FixedPipelineState& fixed_state) const;

    Core::System& system;
    Core::Frontend::EmuWindow& render_window;
    VKScreenInfo& screen_info;
    const VKDevice& device;
    VKResourceManager& resource_manager;
    VKMemoryManager& memory_manager;
    VKScheduler& scheduler;
    const u64 uniform_buffer_alignment;

    std::unique_ptr<VKTextureCache> texture_cache;
    std::unique_ptr<VKPipelineCache> pipeline_cache;
    std::unique_ptr<VKBufferCache> buffer_cache;
    std::unique_ptr<VKGlobalCache> global_cache;

    std::unique_ptr<VKRenderPassCache> renderpass_cache;
    std::unique_ptr<VKSamplerCache> sampler_cache;

    PipelineState state;
    std::vector<std::pair<CachedView*, bool>> sampled_views;
    bool is_indexed{};

    // TODO(Rodrigo): Invalidate on image destruction
    std::unordered_map<FramebufferCacheKey, UniqueFramebuffer> framebuffer_cache;

    using CachedPageMap = boost::icl::interval_map<u64, int>;
    CachedPageMap cached_pages;
};

} // namespace Vulkan