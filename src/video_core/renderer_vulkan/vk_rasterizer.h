// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <utility>

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
struct FramebufferInfo;

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

    void AddDescriptor(vk::DescriptorSet descriptor_set, u32 current_binding,
                       vk::DescriptorType descriptor_type, vk::Buffer buffer, u64 offset,
                       std::size_t size);

    void AddDescriptor(vk::DescriptorSet descriptor_set, u32 current_binding,
                       vk::DescriptorType descriptor_type, vk::Sampler sampler,
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
    void FlushRegion(Tegra::GPUVAddr addr, u64 size) override;
    void InvalidateRegion(Tegra::GPUVAddr addr, u64 size) override;
    void FlushAndInvalidateRegion(Tegra::GPUVAddr addr, u64 size) override;
    bool AccelerateDisplay(const Tegra::FramebufferConfig& config, VAddr framebuffer_addr,
                           u32 pixel_stride) override;
    bool AccelerateDrawBatch(bool is_indexed) override;
    void UpdatePagesCachedCount(Tegra::GPUVAddr addr, u64 size, int delta) override;

    /// Maximum supported size that a constbuffer can have in bytes.
    static constexpr std::size_t MaxConstbufferSize = 0x10000;
    static_assert(MaxConstbufferSize % (4 * sizeof(float)) == 0,
                  "The maximum size of a constbuffer must be a multiple of the size of GLvec4");

private:
    static constexpr u64 STREAM_BUFFER_SIZE = 128 * 1024 * 1024;

    [[nodiscard]] std::tuple<FramebufferInfo, VKExecutionContext> ConfigureFramebuffers(
        VKExecutionContext exctx, vk::RenderPass renderpass, bool using_color_fb = true,
        bool use_zeta_fb = true, bool preserve_contents = true);

    void SetupVertexArrays(PipelineParams& params);

    void SetupIndexBuffer();

    void SetupConstBuffers(const Shader& shader, Maxwell::ShaderStage stage,
                           vk::DescriptorSet descriptor_set);

    void SetupGlobalBuffers(const Shader& shader, Maxwell::ShaderStage stage,
                            vk::DescriptorSet descriptor_set);

    VKExecutionContext SetupTextures(VKExecutionContext exctx, const Shader& shader,
                                     Maxwell::ShaderStage stage, vk::DescriptorSet descriptor_set);

    std::size_t CalculateVertexArraysSize() const;

    std::size_t CalculateIndexBufferSize() const;

    RenderPassParams GetRenderPassParams() const;

    void SyncDepthStencil(PipelineParams& params);
    void SyncInputAssembly(PipelineParams& params);
    void SyncColorBlending(PipelineParams& params);
    void SyncViewportState(PipelineParams& params);
    void SyncRasterizerState(PipelineParams& params);

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
    std::vector<CachedView*> used_views;

    // TODO(Rodrigo): Invalidate on image destruction
    std::unordered_map<FramebufferCacheKey, UniqueFramebuffer> framebuffer_cache;

    enum class AccelDraw { Disabled, Arrays, Indexed };
    AccelDraw accelerate_draw = AccelDraw::Disabled;

    using CachedPageMap = boost::icl::interval_map<u64, int>;
    CachedPageMap cached_pages;
};

} // namespace Vulkan