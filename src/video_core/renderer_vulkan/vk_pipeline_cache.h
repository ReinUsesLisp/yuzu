// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "common/common_types.h"
#include "common/static_vector.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/rasterizer_cache.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_shader_decompiler.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"
#include "video_core/surface.h"

namespace Core {
class System;
}

namespace Vulkan {

class RasterizerVulkan;
class VKDevice;
class VKFence;
class VKScheduler;

class CachedShader;
using Shader = std::shared_ptr<CachedShader>;
using Maxwell = Tegra::Engines::Maxwell3D::Regs;

using PipelineCacheShaders = std::array<VAddr, Maxwell::MaxShaderProgram>;

struct FixedPipelineState {
    using ComponentType = VideoCore::Surface::ComponentType;
    using PixelFormat = VideoCore::Surface::PixelFormat;

    struct VertexBinding {
        u32 index = 0;
        u32 stride = 0;
        u32 divisor = 0;

        std::size_t Hash() const;
        bool operator==(const VertexBinding& rhs) const;
    };

    struct VertexAttribute {
        u32 index = 0;
        u32 buffer = 0;
        Maxwell::VertexAttribute::Type type = Maxwell::VertexAttribute::Type::UnsignedNorm;
        Maxwell::VertexAttribute::Size size = Maxwell::VertexAttribute::Size::Size_8;
        u32 offset = 0;

        std::size_t Hash() const;
        bool operator==(const VertexAttribute& rhs) const;
    };

    struct StencilFace {
        Maxwell::StencilOp action_stencil_fail = Maxwell::StencilOp::Keep;
        Maxwell::StencilOp action_depth_fail = Maxwell::StencilOp::Keep;
        Maxwell::StencilOp action_depth_pass = Maxwell::StencilOp::Keep;
        Maxwell::ComparisonOp test_func = Maxwell::ComparisonOp::Always;
        s32 test_ref = 0;
        u32 test_mask = 0;
        u32 write_mask = 0;

        std::size_t Hash() const;
        bool operator==(const StencilFace& rhs) const;
    };

    struct BlendingAttachment {
        bool enable = false;
        Maxwell::Blend::Equation rgb_equation = Maxwell::Blend::Equation::Add;
        Maxwell::Blend::Factor src_rgb_func = Maxwell::Blend::Factor::One;
        Maxwell::Blend::Factor dst_rgb_func = Maxwell::Blend::Factor::Zero;
        Maxwell::Blend::Equation a_equation = Maxwell::Blend::Equation::Add;
        Maxwell::Blend::Factor src_a_func = Maxwell::Blend::Factor::One;
        Maxwell::Blend::Factor dst_a_func = Maxwell::Blend::Factor::Zero;
        std::array<bool, 4> components{true, true, true, true};

        std::size_t Hash() const;
        bool operator==(const BlendingAttachment& rhs) const;
    };

    struct VertexInput {
        StaticVector<VertexBinding, Maxwell::NumVertexArrays> bindings;
        StaticVector<VertexAttribute, Maxwell::NumVertexAttributes> attributes;

        std::size_t Hash() const;

        bool operator==(const VertexInput& rhs) const;
    } vertex_input;

    struct InputAssembly {
        Maxwell::PrimitiveTopology topology = Maxwell::PrimitiveTopology::Points;
        bool primitive_restart_enable = false;

        std::size_t Hash() const;
        bool operator==(const InputAssembly& rhs) const;
    } input_assembly;

    struct ViewportState {
        float x{};
        float y{};
        float width{};
        float height{};

        std::size_t Hash() const;
        bool operator==(const ViewportState& rhs) const;
    } viewport_state;

    struct Rasterizer {
        bool cull_enable = false;
        Maxwell::Cull::CullFace cull_face = Maxwell::Cull::CullFace::Back;
        Maxwell::Cull::FrontFace front_face = Maxwell::Cull::FrontFace::CounterClockWise;

        std::size_t Hash() const;
        bool operator==(const Rasterizer& rhs) const;
    } rasterizer;

    /*
    struct Multisampling {
        std::size_t Hash() const {
            return 0;
        }

        bool operator==(const Rasterizer& rhs) const {
            return true;
        }
    } multisampling;
    */

    struct DepthStencil {
        bool depth_test_enable = false;
        bool depth_write_enable = true;
        bool depth_bounds_enable = false;
        bool stencil_enable = false;
        Maxwell::ComparisonOp depth_test_function = Maxwell::ComparisonOp::Always;
        StencilFace front_stencil;
        StencilFace back_stencil;
        float depth_bounds_min = 0.0f;
        float depth_bounds_max = 0.0f;

        std::size_t Hash() const;
        bool operator==(const DepthStencil& rhs) const;
    } depth_stencil;

    struct ColorBlending {
        std::array<float, 4> blend_constants{};

        u32 attachments_count{};
        std::array<BlendingAttachment, Maxwell::NumRenderTargets> attachments;

        std::size_t Hash() const;
        bool operator==(const ColorBlending& rhs) const;
    } color_blending;

    std::size_t hash;

    void CalculateHash();

    std::size_t Hash() const;
    bool operator==(const FixedPipelineState& rhs) const;
};

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::FixedPipelineState> {
    std::size_t operator()(const Vulkan::FixedPipelineState& k) const {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

struct Pipeline {
    vk::Pipeline handle;
    vk::PipelineLayout layout;
    vk::DescriptorSet descriptor_set;
    vk::DescriptorUpdateTemplate descriptor_template;
};

class CachedShader final : public RasterizerCacheObject {
public:
    CachedShader(Core::System& system, const VKDevice& device, VAddr addr,
                 Maxwell::ShaderProgram program_type);

    void FillDescriptorLayout(std::vector<vk::DescriptorSetLayoutBinding>& bindings) const;

    VAddr GetAddr() const override {
        return addr;
    }

    std::size_t GetSizeInBytes() const override {
        return entries.shader_length;
    }

    // We do not have to flush this cache as things in it are never modified by us.
    void Flush() override {}

    /// Gets the module handle for the shader.
    vk::ShaderModule GetHandle(vk::PrimitiveTopology primitive_mode) {
        return *shader_module;
    }

    /// Gets the module entries for the shader.
    const VKShader::ShaderEntries& GetEntries() const {
        return entries;
    }

private:
    const VKDevice& device;
    const VAddr addr;
    const Maxwell::ShaderProgram program_type;

    VKShader::ShaderSetup setup;
    VKShader::ShaderEntries entries;

    UniqueShaderModule shader_module;
};

struct PipelineCacheKey {
    PipelineCacheShaders shaders;
    RenderPassParams renderpass;
    FixedPipelineState fixed_state;

    std::size_t Hash() const {
        std::size_t hash = 0;
        for (const auto& shader : shaders)
            boost::hash_combine(hash, shader);
        boost::hash_combine(hash, renderpass.Hash());
        boost::hash_combine(hash, fixed_state.Hash());
        return hash;
    }

    bool operator==(const PipelineCacheKey& rhs) const {
        return std::tie(shaders, renderpass, fixed_state) ==
               std::tie(rhs.shaders, rhs.renderpass, rhs.fixed_state);
    }
};

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::PipelineCacheKey> {
    std::size_t operator()(const Vulkan::PipelineCacheKey& k) const {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class DescriptorPool final : public VKFencedPool {
public:
    explicit DescriptorPool(const VKDevice& device,
                            const std::vector<vk::DescriptorPoolSize>& pool_sizes,
                            const vk::DescriptorSetLayout layout);
    ~DescriptorPool();

    vk::DescriptorSet Commit(VKFence& fence);

protected:
    void Allocate(std::size_t begin, std::size_t end) override;

private:
    const VKDevice& device;
    const std::vector<vk::DescriptorPoolSize> stored_pool_sizes;
    const vk::DescriptorPoolCreateInfo pool_ci;
    const vk::DescriptorSetLayout layout;

    std::vector<UniqueDescriptorPool> pools;
    std::vector<std::vector<UniqueDescriptorSet>> allocations;
};

class DescriptorUpdateEntry {
public:
    DescriptorUpdateEntry(vk::DescriptorImageInfo&& image) : image{image} {}
    DescriptorUpdateEntry(vk::DescriptorBufferInfo&& buffer) : buffer{buffer} {}

private:
    union {
        vk::DescriptorImageInfo image;
        vk::DescriptorBufferInfo buffer;
    };
};

class VKPipelineCache final : public RasterizerCache<Shader> {
public:
    explicit VKPipelineCache(Core::System& system, RasterizerVulkan& rasterizer,
                             const VKDevice& device, VKScheduler& scheduler);

    std::pair<std::array<Shader, Maxwell::MaxShaderStage>, PipelineCacheShaders> GetShaders();

    Pipeline GetPipeline(const FixedPipelineState& fixed_state,
                         const RenderPassParams& renderpass_params,
                         const std::array<Shader, Maxwell::MaxShaderStage>& shaders,
                         const PipelineCacheShaders& shader_addresses, vk::RenderPass renderpass,
                         VKFence& fence);

protected:
    void ObjectInvalidated(const Shader& shader) override;

private:
    struct CacheEntry {
        UniquePipeline pipeline;
        UniquePipelineLayout pipeline_layout;
        UniqueDescriptorSetLayout descriptor_set_layout;
        UniqueDescriptorUpdateTemplate descriptor_template;
        UniqueRenderPass renderpass;
        std::unique_ptr<DescriptorPool> descriptor_pool;
    };

    UniqueDescriptorSetLayout CreateDescriptorSetLayout(
        const std::array<Shader, Maxwell::MaxShaderStage>& shaders) const;

    std::unique_ptr<DescriptorPool> CreateDescriptorPool(
        const std::array<Shader, Maxwell::MaxShaderStage>& shaders,
        vk::DescriptorSetLayout descriptor_set_layout) const;

    UniquePipelineLayout CreatePipelineLayout(vk::DescriptorSetLayout descriptor_set_layout) const;

    UniqueDescriptorUpdateTemplate CreateDescriptorUpdateTemplate(
        const std::array<Shader, Maxwell::MaxShaderStage>& shaders,
        vk::DescriptorSetLayout descriptor_set_layout, vk::PipelineLayout pipeline_layout) const;

    UniquePipeline CreatePipeline(const FixedPipelineState& fixed_state, vk::PipelineLayout layout,
                                  vk::RenderPass renderpass,
                                  const std::array<Shader, Maxwell::MaxShaderStage>& shaders) const;

    Core::System& system;
    const VKDevice& device;
    VKScheduler& scheduler;

    std::unordered_map<PipelineCacheKey, std::unique_ptr<CacheEntry>> cache;
};

} // namespace Vulkan