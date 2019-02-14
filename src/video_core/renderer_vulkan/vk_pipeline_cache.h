// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <map>
#include <memory>
#include <tuple>
#include "common/common_types.h"
#include "common/static_vector.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/rasterizer_cache.h"
#include "video_core/renderer_vulkan/declarations.h"
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

class CachedShader;
using Shader = std::shared_ptr<CachedShader>;
using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct RenderPassParams;

struct PipelineParams {
    using ComponentType = VideoCore::Surface::ComponentType;
    using PixelFormat = VideoCore::Surface::PixelFormat;

    struct VertexBinding {
        u32 index = 0;
        u32 stride = 0;
        u32 divisor = 0;

        auto Tie() const {
            return std::tie(index, stride, divisor);
        }

        bool operator<(const VertexBinding& rhs) const {
            return Tie() < rhs.Tie();
        }
    };

    struct VertexAttribute {
        u32 index = 0;
        u32 buffer = 0;
        Maxwell::VertexAttribute::Type type = Maxwell::VertexAttribute::Type::UnsignedNorm;
        Maxwell::VertexAttribute::Size size = Maxwell::VertexAttribute::Size::Size_8;
        u32 offset = 0;

        auto Tie() const {
            return std::tie(index, buffer, type, size, offset);
        }

        bool operator<(const VertexAttribute& rhs) const {
            return Tie() < rhs.Tie();
        }
    };

    struct StencilFace {
        Maxwell::StencilOp action_stencil_fail = Maxwell::StencilOp::Keep;
        Maxwell::StencilOp action_depth_fail = Maxwell::StencilOp::Keep;
        Maxwell::StencilOp action_depth_pass = Maxwell::StencilOp::Keep;
        Maxwell::ComparisonOp test_func = Maxwell::ComparisonOp::Always;
        s32 test_ref = 0;
        u32 test_mask = 0;
        u32 write_mask = 0;

        auto Tie() const {
            return std::tie(action_stencil_fail, action_depth_fail, action_depth_pass, test_func,
                            test_ref, test_mask, write_mask);
        }

        bool operator<(const StencilFace& rhs) const {
            return Tie() < rhs.Tie();
        }
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

        auto Tie() const {
            return std::tie(enable, rgb_equation, src_rgb_func, dst_rgb_func, a_equation,
                            src_a_func, dst_a_func, components);
        }

        bool operator<(const BlendingAttachment& rhs) const {
            return Tie() < rhs.Tie();
        }
    };

    struct {
        StaticVector<VertexBinding, Maxwell::NumVertexArrays> bindings;
        StaticVector<VertexAttribute, Maxwell::NumVertexAttributes> attributes;

        auto Tie() const {
            return std::tie(bindings, attributes);
        }
    } vertex_input;

    struct {
        Maxwell::PrimitiveTopology topology = Maxwell::PrimitiveTopology::Points;
        bool primitive_restart_enable = false;

        auto Tie() const {
            return std::tie(topology, primitive_restart_enable);
        }
    } input_assembly;

    struct {
        float x{};
        float y{};
        float width{};
        float height{};

        auto Tie() const {
            return std::tie(x, y, width, height);
        }
    } viewport_state;

    struct {
        bool cull_enable = false;
        Maxwell::Cull::CullFace cull_face = Maxwell::Cull::CullFace::Back;
        Maxwell::Cull::FrontFace front_face = Maxwell::Cull::FrontFace::CounterClockWise;

        auto Tie() const {
            return std::tie(cull_enable, cull_face, front_face);
        }
    } rasterizer;

    struct {
        auto Tie() const {
            return std::tie();
        }
    } multisampling;

    struct {
        bool depth_test_enable = false;
        bool depth_write_enable = true;
        bool depth_bounds_enable = false;
        bool stencil_enable = false;
        Maxwell::ComparisonOp depth_test_function = Maxwell::ComparisonOp::Always;
        StencilFace front_stencil;
        StencilFace back_stencil;
        float depth_bounds_min = 0.f;
        float depth_bounds_max = 0.f;

        auto Tie() const {
            return std::tie(depth_test_enable, depth_write_enable, depth_bounds_enable,
                            depth_test_function, stencil_enable, front_stencil, back_stencil,
                            depth_bounds_min, depth_bounds_max);
        }
    } depth_stencil;

    struct {
        std::array<float, 4> blend_constants{};
        std::array<BlendingAttachment, Maxwell::NumRenderTargets> attachments;
        bool independent_blend = false;

        auto Tie() const {
            return std::tie(blend_constants, attachments, independent_blend);
        }
    } color_blending;

    bool operator<(const PipelineParams& rhs) const {
        return vertex_input.Tie() < rhs.vertex_input.Tie() ||
               input_assembly.Tie() < rhs.input_assembly.Tie() ||
               viewport_state.Tie() < rhs.viewport_state.Tie() ||
               rasterizer.Tie() < rhs.rasterizer.Tie() ||
               multisampling.Tie() < rhs.multisampling.Tie() ||
               depth_stencil.Tie() < rhs.depth_stencil.Tie() ||
               color_blending.Tie() < rhs.color_blending.Tie();
    }
};

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

class VKPipelineCache final : public RasterizerCache<Shader> {
public:
    explicit VKPipelineCache(Core::System& system, RasterizerVulkan& rasterizer,
                             const VKDevice& device);

    // Passing a renderpass object is not really needed (since it could be found from rp_params),
    // but this would require searching for the entry twice. Instead of doing that, pass the (draw)
    // renderpass that fulfills those params.
    Pipeline GetPipeline(const PipelineParams& params, const RenderPassParams& renderpass_params,
                         vk::RenderPass renderpass);

protected:
    void ObjectInvalidated(const Shader& shader) override;

private:
    using ShaderPipeline = std::array<VAddr, Maxwell::MaxShaderProgram>;
    using CacheKey = std::tuple<ShaderPipeline, RenderPassParams, PipelineParams>;

    struct CacheEntry {
        UniquePipeline pipeline;
        UniquePipelineLayout layout;
        UniqueRenderPass renderpass;
    };

    Core::System& system;
    const VKDevice& device;

    UniquePipelineLayout CreatePipelineLayout(const PipelineParams& params,
                                              const Pipeline& pipeline) const;
    UniquePipeline CreatePipeline(const PipelineParams& params, const Pipeline& pipeline,
                                  vk::RenderPass renderpass) const;

    std::map<CacheKey, std::unique_ptr<CacheEntry>> cache;
    UniqueDescriptorSetLayout empty_set_layout;
};

} // namespace Vulkan