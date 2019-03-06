// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <tuple>
#include <unordered_map>

#include "common/common_types.h"
#include "common/static_vector.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/rasterizer_cache.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_shader_decompiler.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"
#include "video_core/surface.h"

namespace Core {
class System;
} // namespace Core

namespace Vulkan {

class RasterizerVulkan;
class VKDevice;
class VKFence;
class VKScheduler;

class CachedShader;
using Shader = std::shared_ptr<CachedShader>;
using Maxwell = Tegra::Engines::Maxwell3D::Regs;

using PipelineCacheShaders = std::array<VAddr, Maxwell::MaxShaderProgram>;

struct PipelineParams {
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
        std::array<BlendingAttachment, Maxwell::NumRenderTargets> attachments;
        bool independent_blend = false;

        std::size_t Hash() const;
        bool operator==(const ColorBlending& rhs) const;
    } color_blending;

    std::size_t hash;

    void CalculateHash();

    std::size_t Hash() const;
    bool operator==(const PipelineParams& rhs) const;
};

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::PipelineParams> {
    std::size_t operator()(const Vulkan::PipelineParams& k) const {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

struct Pipeline {
    vk::Pipeline handle;
    vk::PipelineLayout layout;
    std::array<Shader, Maxwell::MaxShaderStage> shaders;
};

class CachedShader final : public RasterizerCacheObject {
public:
    CachedShader(Core::System& system, const VKDevice& device, VAddr addr,
                 Maxwell::ShaderProgram program_type);

    /// Gets a descriptor set from the internal pool.
    vk::DescriptorSet CommitDescriptorSet(VKFence& fence);

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

    /// Gets the descriptor set layout of the shader.
    vk::DescriptorSetLayout GetDescriptorSetLayout() const {
        return *descriptor_set_layout;
    }

    /// Gets the module entries for the shader.
    const VKShader::ShaderEntries& GetEntries() const {
        return entries;
    }

private:
    class DescriptorPool;

    void CreateDescriptorSetLayout();
    void CreateDescriptorPool();

    const VKDevice& device;
    const VAddr addr;
    const Maxwell::ShaderProgram program_type;

    VKShader::ShaderSetup setup;
    VKShader::ShaderEntries entries;

    UniqueShaderModule shader_module;

    UniqueDescriptorSetLayout descriptor_set_layout;
    std::unique_ptr<DescriptorPool> descriptor_pool;
};

struct PipelineCacheKey {
    PipelineCacheShaders shaders;
    RenderPassParams renderpass;
    PipelineParams pipeline;

    std::size_t Hash() const {
        std::size_t hash = 0;
        for (const auto& shader : shaders)
            boost::hash_combine(hash, shader);
        boost::hash_combine(hash, renderpass.Hash());
        boost::hash_combine(hash, pipeline.Hash());
        return hash;
    }

    bool operator==(const PipelineCacheKey& rhs) const {
        return std::tie(shaders, renderpass, pipeline) ==
               std::tie(rhs.shaders, rhs.renderpass, rhs.pipeline);
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

class VKPipelineCache final : public RasterizerCache<Shader> {
public:
    explicit VKPipelineCache(Core::System& system, RasterizerVulkan& rasterizer,
                             const VKDevice& device, VKScheduler& scheduler);

    // Passing a renderpass object is not really needed (since it could be found from rp_params),
    // but this would require searching for the entry twice. Instead of doing that, pass the (draw)
    // renderpass that fulfills those params.
    Pipeline GetPipeline(const PipelineParams& params, const RenderPassParams& renderpass_params,
                         vk::RenderPass renderpass);

protected:
    void ObjectInvalidated(const Shader& shader) override;

private:
    struct CacheEntry {
        UniquePipeline pipeline;
        UniquePipelineLayout layout;
        UniqueRenderPass renderpass;
    };

    Core::System& system;
    const VKDevice& device;
    VKScheduler& scheduler;

    UniquePipelineLayout CreatePipelineLayout(const PipelineParams& params,
                                              const Pipeline& pipeline) const;
    UniquePipeline CreatePipeline(const PipelineParams& params, const Pipeline& pipeline,
                                  vk::PipelineLayout layout, vk::RenderPass renderpass) const;

    std::unordered_map<PipelineCacheKey, std::unique_ptr<CacheEntry>> cache;
    UniqueDescriptorSetLayout empty_set_layout;
};

} // namespace Vulkan