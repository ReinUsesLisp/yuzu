// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <tuple>
#include <unordered_map>
#include <boost/functional/hash.hpp>
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

        std::size_t Hash() const {
            std::size_t hash = 0;
            boost::hash_combine(hash, index);
            boost::hash_combine(hash, stride);
            boost::hash_combine(hash, divisor);
            return hash;
        }

        bool operator==(const VertexBinding& rhs) const {
            return std::tie(index, stride, divisor) == std::tie(rhs.index, rhs.stride, rhs.divisor);
        }
    };

    struct VertexAttribute {
        u32 index = 0;
        u32 buffer = 0;
        Maxwell::VertexAttribute::Type type = Maxwell::VertexAttribute::Type::UnsignedNorm;
        Maxwell::VertexAttribute::Size size = Maxwell::VertexAttribute::Size::Size_8;
        u32 offset = 0;

        std::size_t Hash() const {
            std::size_t hash = 0;
            boost::hash_combine(hash, index);
            boost::hash_combine(hash, buffer);
            boost::hash_combine(hash, type);
            boost::hash_combine(hash, size);
            boost::hash_combine(hash, offset);
            return hash;
        }

        bool operator==(const VertexAttribute& rhs) const {
            return std::tie(index, buffer, type, size, offset) ==
                   std::tie(rhs.index, rhs.buffer, rhs.type, rhs.size, rhs.offset);
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

        std::size_t Hash() const {
            std::size_t hash = 0;
            boost::hash_combine(hash, action_stencil_fail);
            boost::hash_combine(hash, action_depth_fail);
            boost::hash_combine(hash, action_depth_pass);
            boost::hash_combine(hash, test_func);
            boost::hash_combine(hash, test_ref);
            boost::hash_combine(hash, test_mask);
            boost::hash_combine(hash, write_mask);
            return hash;
        }

        bool operator==(const StencilFace& rhs) const {
            return std::tie(action_stencil_fail, action_depth_fail, action_depth_pass, test_func,
                            test_ref, test_mask, write_mask) ==
                   std::tie(rhs.action_stencil_fail, rhs.action_depth_fail, rhs.action_depth_pass,
                            rhs.test_func, rhs.test_ref, rhs.test_mask, rhs.write_mask);
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

        std::size_t Hash() const {
            std::size_t hash = 0;
            boost::hash_combine(hash, enable);
            boost::hash_combine(hash, rgb_equation);
            boost::hash_combine(hash, src_rgb_func);
            boost::hash_combine(hash, dst_rgb_func);
            boost::hash_combine(hash, a_equation);
            boost::hash_combine(hash, src_a_func);
            boost::hash_combine(hash, dst_a_func);
            boost::hash_combine(hash, components);
            return hash;
        }

        bool operator==(const BlendingAttachment& rhs) const {
            return std::tie(enable, rgb_equation, src_rgb_func, dst_rgb_func, a_equation,
                            src_a_func, dst_a_func, components) ==
                   std::tie(rhs.enable, rhs.rgb_equation, rhs.src_rgb_func, rhs.dst_rgb_func,
                            rhs.a_equation, rhs.src_a_func, rhs.dst_a_func, rhs.components);
        }
    };

    struct VertexInput {
        StaticVector<VertexBinding, Maxwell::NumVertexArrays> bindings;
        StaticVector<VertexAttribute, Maxwell::NumVertexAttributes> attributes;

        std::size_t Hash() const {
            std::size_t hash = 0;
            for (const auto& binding : bindings)
                boost::hash_combine(hash, binding.Hash());
            for (const auto& attribute : attributes)
                boost::hash_combine(hash, attribute.Hash());
            return hash;
        }

        bool operator==(const VertexInput& rhs) const {
            return std::tie(bindings, attributes) == std::tie(rhs.bindings, rhs.attributes);
        }
    } vertex_input;

    struct InputAssembly {
        Maxwell::PrimitiveTopology topology = Maxwell::PrimitiveTopology::Points;
        bool primitive_restart_enable = false;

        std::size_t Hash() const {
            std::size_t hash = 0;
            boost::hash_combine(hash, topology);
            boost::hash_combine(hash, primitive_restart_enable);
            return hash;
        }

        bool operator==(const InputAssembly& rhs) const {
            return std::tie(topology, primitive_restart_enable) ==
                   std::tie(rhs.topology, rhs.primitive_restart_enable);
        }
    } input_assembly;

    struct ViewportState {
        float x{};
        float y{};
        float width{};
        float height{};

        std::size_t Hash() const {
            std::size_t hash = 0;
            boost::hash_combine(hash, x);
            boost::hash_combine(hash, y);
            boost::hash_combine(hash, width);
            boost::hash_combine(hash, height);
            return hash;
        }

        bool operator==(const ViewportState& rhs) const {
            return std::tie(x, y, width, height) == std::tie(rhs.x, rhs.y, rhs.width, rhs.height);
        }
    } viewport_state;

    struct Rasterizer {
        bool cull_enable = false;
        Maxwell::Cull::CullFace cull_face = Maxwell::Cull::CullFace::Back;
        Maxwell::Cull::FrontFace front_face = Maxwell::Cull::FrontFace::CounterClockWise;

        std::size_t Hash() const {
            std::size_t hash = 0;
            boost::hash_combine(hash, cull_enable);
            boost::hash_combine(hash, cull_face);
            boost::hash_combine(hash, front_face);
            return hash;
        }

        bool operator==(const Rasterizer& rhs) const {
            return std::tie(cull_enable, cull_face, front_face) ==
                   std::tie(rhs.cull_enable, rhs.cull_face, rhs.front_face);
        }
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

        std::size_t Hash() const {
            std::size_t hash = 0;
            boost::hash_combine(hash, depth_test_enable);
            boost::hash_combine(hash, depth_write_enable);
            boost::hash_combine(hash, depth_bounds_enable);
            boost::hash_combine(hash, stencil_enable);
            boost::hash_combine(hash, depth_test_function);
            boost::hash_combine(hash, front_stencil.Hash());
            boost::hash_combine(hash, back_stencil.Hash());
            boost::hash_combine(hash, depth_bounds_min);
            boost::hash_combine(hash, depth_bounds_max);
            return hash;
        }

        bool operator==(const DepthStencil& rhs) const {
            return std::tie(depth_test_enable, depth_write_enable, depth_bounds_enable,
                            depth_test_function, stencil_enable, front_stencil, back_stencil,
                            depth_bounds_min, depth_bounds_max) ==
                   std::tie(rhs.depth_test_enable, rhs.depth_write_enable, rhs.depth_bounds_enable,
                            rhs.depth_test_function, rhs.stencil_enable, rhs.front_stencil,
                            rhs.back_stencil, rhs.depth_bounds_min, rhs.depth_bounds_max);
        }
    } depth_stencil;

    struct ColorBlending {
        std::array<float, 4> blend_constants{};
        std::array<BlendingAttachment, Maxwell::NumRenderTargets> attachments;
        bool independent_blend = false;

        std::size_t Hash() const {
            std::size_t hash = 0;
            boost::hash_combine(hash, blend_constants);
            for (const auto& attachment : attachments)
                boost::hash_combine(hash, attachment.Hash());
            boost::hash_combine(hash, independent_blend);
            return hash;
        }

        bool operator==(const ColorBlending& rhs) const {
            return std::tie(blend_constants, attachments, independent_blend) ==
                   std::tie(rhs.blend_constants, rhs.attachments, rhs.independent_blend);
        }
    } color_blending;

    std::size_t Hash() const {
        std::size_t hash = 0;
        boost::hash_combine(hash, vertex_input.Hash());
        boost::hash_combine(hash, input_assembly.Hash());
        boost::hash_combine(hash, viewport_state.Hash());
        boost::hash_combine(hash, rasterizer.Hash());
        // boost::hash_combine(hash, multisampling.Hash());
        boost::hash_combine(hash, depth_stencil.Hash());
        boost::hash_combine(hash, color_blending.Hash());
        return hash;
    }

    bool operator==(const PipelineParams& rhs) const {
        return std::tie(vertex_input, input_assembly, viewport_state, rasterizer, /*multisampling,*/
                        depth_stencil, color_blending) ==
               std::tie(rhs.vertex_input, rhs.input_assembly, rhs.viewport_state, rhs.rasterizer,
                        /*rhs.multisampling,*/ rhs.depth_stencil, rhs.color_blending);
    }
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
                             const VKDevice& device);

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

    UniquePipelineLayout CreatePipelineLayout(const PipelineParams& params,
                                              const Pipeline& pipeline) const;
    UniquePipeline CreatePipeline(const PipelineParams& params, const Pipeline& pipeline,
                                  vk::RenderPass renderpass) const;

    std::unordered_map<PipelineCacheKey, std::unique_ptr<CacheEntry>> cache;
    UniqueDescriptorSetLayout empty_set_layout;
};

} // namespace Vulkan