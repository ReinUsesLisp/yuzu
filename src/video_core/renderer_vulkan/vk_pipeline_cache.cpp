// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstddef>
#include <memory>
#include <vector>

#include <boost/functional/hash.hpp>

#include "common/static_vector.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"

namespace Vulkan {

// How many sets are created per descriptor pool.
static constexpr std::size_t SETS_PER_POOL = 0x400;

/// Gets the address for the specified shader stage program
static VAddr GetShaderAddress(Core::System& system, Maxwell::ShaderProgram program) {
    const auto& gpu = system.GPU().Maxwell3D();
    const auto& shader_config = gpu.regs.shader_config[static_cast<std::size_t>(program)];
    const auto cpu_addr = gpu.memory_manager.GpuToCpuAddress(gpu.regs.code_address.CodeAddress() +
                                                             shader_config.offset);
    ASSERT(cpu_addr);
    return *cpu_addr;
}

static std::size_t GetStageFromProgram(std::size_t program) {
    return program == 0 ? 0 : program - 1;
}

static Maxwell::ShaderStage GetStageFromProgram(Maxwell::ShaderProgram program) {
    return static_cast<Maxwell::ShaderStage>(
        GetStageFromProgram(static_cast<std::size_t>(program)));
}

/// Gets the shader program code from memory for the specified address
static VKShader::ProgramCode GetShaderCode(VAddr addr) {
    VKShader::ProgramCode program_code(VKShader::MAX_PROGRAM_CODE_LENGTH);
    Memory::ReadBlock(addr, program_code.data(), program_code.size() * sizeof(u64));
    return program_code;
}

static vk::StencilOpState GetStencilFaceState(const PipelineParams::StencilFace& state) {
    return vk::StencilOpState(MaxwellToVK::StencilOp(state.action_stencil_fail),
                              MaxwellToVK::StencilOp(state.action_depth_pass),
                              MaxwellToVK::StencilOp(state.action_depth_fail),
                              MaxwellToVK::ComparisonOp(state.test_func), state.test_mask,
                              state.write_mask, state.test_ref);
}

std::size_t PipelineParams::VertexBinding::Hash() const {
    std::size_t hash = 0;
    boost::hash_combine(hash, index);
    boost::hash_combine(hash, stride);
    boost::hash_combine(hash, divisor);
    return hash;
}

bool PipelineParams::VertexBinding::operator==(const VertexBinding& rhs) const {
    return std::tie(index, stride, divisor) == std::tie(rhs.index, rhs.stride, rhs.divisor);
}

std::size_t PipelineParams::VertexAttribute::Hash() const {
    std::size_t hash = 0;
    boost::hash_combine(hash, index);
    boost::hash_combine(hash, buffer);
    boost::hash_combine(hash, type);
    boost::hash_combine(hash, size);
    boost::hash_combine(hash, offset);
    return hash;
}

bool PipelineParams::VertexAttribute::operator==(const VertexAttribute& rhs) const {
    return std::tie(index, buffer, type, size, offset) ==
           std::tie(rhs.index, rhs.buffer, rhs.type, rhs.size, rhs.offset);
}

std::size_t PipelineParams::StencilFace::Hash() const {
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

bool PipelineParams::StencilFace::operator==(const StencilFace& rhs) const {
    return std::tie(action_stencil_fail, action_depth_fail, action_depth_pass, test_func, test_ref,
                    test_mask, write_mask) ==
           std::tie(rhs.action_stencil_fail, rhs.action_depth_fail, rhs.action_depth_pass,
                    rhs.test_func, rhs.test_ref, rhs.test_mask, rhs.write_mask);
}

std::size_t PipelineParams::BlendingAttachment::Hash() const {
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

bool PipelineParams::BlendingAttachment::operator==(const BlendingAttachment& rhs) const {
    return std::tie(enable, rgb_equation, src_rgb_func, dst_rgb_func, a_equation, src_a_func,
                    dst_a_func, components) ==
           std::tie(rhs.enable, rhs.rgb_equation, rhs.src_rgb_func, rhs.dst_rgb_func,
                    rhs.a_equation, rhs.src_a_func, rhs.dst_a_func, rhs.components);
}

std::size_t PipelineParams::VertexInput::Hash() const {
    std::size_t hash = 0;
    for (const auto& binding : bindings)
        boost::hash_combine(hash, binding.Hash());
    for (const auto& attribute : attributes)
        boost::hash_combine(hash, attribute.Hash());
    return hash;
}

bool PipelineParams::VertexInput::operator==(const VertexInput& rhs) const {
    return std::tie(bindings, attributes) == std::tie(rhs.bindings, rhs.attributes);
}

std::size_t PipelineParams::InputAssembly::Hash() const {
    std::size_t hash = 0;
    boost::hash_combine(hash, topology);
    boost::hash_combine(hash, primitive_restart_enable);
    return hash;
}

bool PipelineParams::InputAssembly::operator==(const InputAssembly& rhs) const {
    return std::tie(topology, primitive_restart_enable) ==
           std::tie(rhs.topology, rhs.primitive_restart_enable);
}

std::size_t PipelineParams::ViewportState::Hash() const {
    std::size_t hash = 0;
    boost::hash_combine(hash, x);
    boost::hash_combine(hash, y);
    boost::hash_combine(hash, width);
    boost::hash_combine(hash, height);
    return hash;
}

bool PipelineParams::ViewportState::operator==(const ViewportState& rhs) const {
    return std::tie(x, y, width, height) == std::tie(rhs.x, rhs.y, rhs.width, rhs.height);
}

std::size_t PipelineParams::Rasterizer::Hash() const {
    std::size_t hash = 0;
    boost::hash_combine(hash, cull_enable);
    boost::hash_combine(hash, cull_face);
    boost::hash_combine(hash, front_face);
    return hash;
}

bool PipelineParams::Rasterizer::operator==(const Rasterizer& rhs) const {
    return std::tie(cull_enable, cull_face, front_face) ==
           std::tie(rhs.cull_enable, rhs.cull_face, rhs.front_face);
}

std::size_t PipelineParams::DepthStencil::Hash() const {
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

bool PipelineParams::DepthStencil::operator==(const DepthStencil& rhs) const {
    return std::tie(depth_test_enable, depth_write_enable, depth_bounds_enable, depth_test_function,
                    stencil_enable, front_stencil, back_stencil, depth_bounds_min,
                    depth_bounds_max) ==
           std::tie(rhs.depth_test_enable, rhs.depth_write_enable, rhs.depth_bounds_enable,
                    rhs.depth_test_function, rhs.stencil_enable, rhs.front_stencil,
                    rhs.back_stencil, rhs.depth_bounds_min, rhs.depth_bounds_max);
}

std::size_t PipelineParams::ColorBlending::Hash() const {
    std::size_t hash = 0;
    boost::hash_combine(hash, blend_constants);
    for (const auto& attachment : attachments)
        boost::hash_combine(hash, attachment.Hash());
    boost::hash_combine(hash, independent_blend);
    return hash;
}

bool PipelineParams::ColorBlending::operator==(const ColorBlending& rhs) const {
    return std::tie(blend_constants, attachments, independent_blend) ==
           std::tie(rhs.blend_constants, rhs.attachments, rhs.independent_blend);
}

void PipelineParams::CalculateHash() {
    hash = 0;
    boost::hash_combine(hash, vertex_input.Hash());
    boost::hash_combine(hash, input_assembly.Hash());
    boost::hash_combine(hash, viewport_state.Hash());
    boost::hash_combine(hash, rasterizer.Hash());
    // boost::hash_combine(hash, multisampling.Hash());
    boost::hash_combine(hash, depth_stencil.Hash());
    boost::hash_combine(hash, color_blending.Hash());
}

std::size_t PipelineParams::Hash() const {
    return hash;
}

bool PipelineParams::operator==(const PipelineParams& rhs) const {
    return std::tie(vertex_input, input_assembly, viewport_state, rasterizer, /*multisampling,*/
                    depth_stencil, color_blending) ==
           std::tie(rhs.vertex_input, rhs.input_assembly, rhs.viewport_state, rhs.rasterizer,
                    /*rhs.multisampling,*/ rhs.depth_stencil, rhs.color_blending);
}

class CachedShader::DescriptorPool final : public VKFencedPool {
public:
    explicit DescriptorPool(const VKDevice& device,
                            const std::vector<vk::DescriptorPoolSize>& pool_sizes,
                            const vk::DescriptorSetLayout layout)
        : VKFencedPool(SETS_PER_POOL),
          pool_ci(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, SETS_PER_POOL,
                  static_cast<u32>(stored_pool_sizes.size()), stored_pool_sizes.data()),
          stored_pool_sizes{pool_sizes}, layout{layout}, device{device} {}

    ~DescriptorPool() = default;

    vk::DescriptorSet Commit(VKFence& fence) {
        const std::size_t index = CommitResource(fence);
        const std::size_t pool_index = index / SETS_PER_POOL;
        const std::size_t set_index = index % SETS_PER_POOL;
        return allocations[pool_index][set_index].get();
    }

protected:
    void Allocate(std::size_t begin, std::size_t end) override {
        ASSERT_MSG(begin % SETS_PER_POOL == 0 && end % SETS_PER_POOL == 0, "Not aligned.");

        const auto dev = device.GetLogical();
        const auto& dld = device.GetDispatchLoader();

        auto pool = dev.createDescriptorPoolUnique(pool_ci, nullptr, dld);
        const std::vector<vk::DescriptorSetLayout> layout_clones(SETS_PER_POOL, layout);
        const vk::DescriptorSetAllocateInfo descriptor_set_ai(*pool, SETS_PER_POOL,
                                                              layout_clones.data());

        pools.push_back(std::move(pool));

        allocations.push_back(dev.allocateDescriptorSetsUnique<std::allocator<UniqueDescriptorSet>>(
            descriptor_set_ai, dld));
    }

private:
    const VKDevice& device;
    const std::vector<vk::DescriptorPoolSize> stored_pool_sizes;
    const vk::DescriptorPoolCreateInfo pool_ci;
    const vk::DescriptorSetLayout layout;

    std::vector<UniqueDescriptorPool> pools;
    std::vector<std::vector<UniqueDescriptorSet>> allocations;
};

CachedShader::CachedShader(Core::System& system, const VKDevice& device, VAddr addr,
                           Maxwell::ShaderProgram program_type)
    : device{device}, addr{addr}, program_type{program_type}, setup{GetShaderCode(addr)} {
    VKShader::ProgramResult program_result = [&]() {
        switch (program_type) {
        case Maxwell::ShaderProgram::VertexA:
            // VertexB is always enabled, so when VertexA is enabled, we have two vertex shaders.
            // Conventional HW does not support this, so we combine VertexA and VertexB into one
            // stage here.
            setup.SetProgramB(
                GetShaderCode(GetShaderAddress(system, Maxwell::ShaderProgram::VertexB)));
        case Maxwell::ShaderProgram::VertexB:
            return VKShader::GenerateVertexShader(setup);
        case Maxwell::ShaderProgram::Fragment:
            return VKShader::GenerateFragmentShader(setup);
        default:
            LOG_CRITICAL(HW_GPU, "Unimplemented program_type={}", static_cast<u32>(program_type));
            UNREACHABLE();
        }
    }();

    entries = program_result.entries;

    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();

    const vk::ShaderModuleCreateInfo shader_module_ci(
        {}, program_result.code.size(), reinterpret_cast<const u32*>(program_result.code.data()));
    shader_module = dev.createShaderModuleUnique(shader_module_ci, nullptr, dld);

    CreateDescriptorSetLayout();
    CreateDescriptorPool();
}

vk::DescriptorSet CachedShader::CommitDescriptorSet(VKFence& fence) {
    if (descriptor_pool == nullptr) {
        // If the descriptor pool has not been initialized, it means that the shader doesn't used
        // descriptors. Return a null descriptor set.
        return nullptr;
    }
    return descriptor_pool->Commit(fence);
}

void CachedShader::CreateDescriptorSetLayout() {
    const vk::ShaderStageFlags stage = MaxwellToVK::ShaderStage(GetStageFromProgram(program_type));

    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (u32 bindpoint = 0; bindpoint < static_cast<u32>(entries.const_buffers.size());
         ++bindpoint) {
        const auto& entry = entries.const_buffers[bindpoint];
        const u32 current_binding = entries.const_buffers_base_binding + bindpoint;
        bindings.emplace_back(current_binding, vk::DescriptorType::eUniformBuffer, 1, stage,
                              nullptr);
    }
    for (u32 bindpoint = 0; bindpoint < static_cast<u32>(entries.samplers.size()); ++bindpoint) {
        const auto& entry = entries.samplers[bindpoint];
        const u32 current_binding = entries.samplers_base_binding + bindpoint;
        bindings.emplace_back(current_binding, vk::DescriptorType::eCombinedImageSampler, 1, stage,
                              nullptr);
    }

    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    descriptor_set_layout = dev.createDescriptorSetLayoutUnique(
        {{}, static_cast<u32>(bindings.size()), bindings.data()}, nullptr, dld);
}

void CachedShader::CreateDescriptorPool() {
    std::vector<vk::DescriptorPoolSize> pool_sizes;

    const auto PushSize = [&](vk::DescriptorType descriptor_type, std::size_t size) {
        if (size > 0) {
            pool_sizes.push_back({descriptor_type, static_cast<u32>(size * SETS_PER_POOL)});
        }
    };
    PushSize(vk::DescriptorType::eUniformBuffer, entries.const_buffers.size());
    PushSize(vk::DescriptorType::eInputAttachment, entries.attributes.size());
    PushSize(vk::DescriptorType::eCombinedImageSampler, entries.samplers.size());

    if (pool_sizes.size() == 0) {
        // If the shader doesn't use descriptor sets, skip the pool creation.
        return;
    }

    descriptor_pool = std::make_unique<DescriptorPool>(device, pool_sizes, *descriptor_set_layout);
}

VKPipelineCache::VKPipelineCache(Core::System& system, RasterizerVulkan& rasterizer,
                                 const VKDevice& device, VKScheduler& scheduler)
    : RasterizerCache{rasterizer}, system{system}, device{device}, scheduler{scheduler} {
    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    empty_set_layout = dev.createDescriptorSetLayoutUnique({{}, 0, nullptr}, nullptr, dld);
}

Pipeline VKPipelineCache::GetPipeline(const PipelineParams& params,
                                      const RenderPassParams& renderpass_params,
                                      vk::RenderPass renderpass) {
    const auto& gpu = system.GPU().Maxwell3D();
    Pipeline pipeline;
    PipelineCacheShaders shaders{};

    for (std::size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const auto& shader_config = gpu.regs.shader_config[index];
        const auto program{static_cast<Maxwell::ShaderProgram>(index)};

        // Skip stages that are not enabled
        if (!gpu.regs.IsShaderConfigEnabled(index)) {
            continue;
        }

        const VAddr program_addr{GetShaderAddress(system, program)};
        shaders[index] = program_addr;

        // Look up shader in the cache based on address
        Shader shader{TryGet(program_addr)};

        if (!shader) {
            // No shader found - create a new one
            shader = std::make_shared<CachedShader>(system, device, program_addr, program);
            Register(shader);
        }

        const std::size_t stage = index == 0 ? 0 : index - 1;
        pipeline.shaders[stage] = std::move(shader);

        // When VertexA is enabled, we have dual vertex shaders
        if (program == Maxwell::ShaderProgram::VertexA) {
            // VertexB was combined with VertexA, so we skip the VertexB iteration
            index++;
        }
    }

    PipelineCacheKey key;
    key.shaders = shaders;
    key.renderpass = renderpass_params;
    key.pipeline = params;
    key.pipeline.CalculateHash();
    const auto [pair, is_cache_miss] = cache.try_emplace(key);
    auto& entry = pair->second;

    if (is_cache_miss) {
        entry = std::make_unique<CacheEntry>();

        entry->layout = CreatePipelineLayout(params, pipeline);
        pipeline.layout = *entry->layout;

        entry->pipeline = CreatePipeline(params, pipeline, renderpass);
    }

    pipeline.handle = *entry->pipeline;
    pipeline.layout = *entry->layout;
    return pipeline;
}

void VKPipelineCache::ObjectInvalidated(const Shader& shader) {
    const VAddr invalidated_addr = shader->GetAddr();
    for (auto it = cache.begin(); it != cache.end();) {
        auto& entry = it->first;
        const bool has_addr = [&]() {
            const auto [shaders, renderpass_params, params] = entry;
            for (auto& shader_addr : shaders) {
                if (shader_addr == invalidated_addr) {
                    return true;
                }
            }
            return false;
        }();
        if (has_addr) {
            // TODO(Rodrigo): Instead of finishing here, wait for the fences that use this pipeline
            // and flush.
            scheduler.Finish();
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
}

UniquePipelineLayout VKPipelineCache::CreatePipelineLayout(const PipelineParams& params,
                                                           const Pipeline& pipeline) const {
    std::array<vk::DescriptorSetLayout, Maxwell::MaxShaderStage> set_layouts{};
    for (std::size_t i = 0; i < Maxwell::MaxShaderStage; ++i) {
        const auto& shader = pipeline.shaders[i];
        set_layouts[i] = shader != nullptr ? shader->GetDescriptorSetLayout() : *empty_set_layout;
    }

    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    return dev.createPipelineLayoutUnique(
        {{}, static_cast<u32>(set_layouts.size()), set_layouts.data(), 0, nullptr}, nullptr, dld);
}

UniquePipeline VKPipelineCache::CreatePipeline(const PipelineParams& params,
                                               const Pipeline& pipeline,
                                               vk::RenderPass renderpass) const {
    const auto& vi = params.vertex_input;
    const auto& ia = params.input_assembly;
    const auto& ds = params.depth_stencil;
    const auto& cd = params.color_blending;
    const auto& vs = params.viewport_state;
    const auto& rs = params.rasterizer;

    StaticVector<vk::VertexInputBindingDescription, Maxwell::NumVertexArrays> vertex_bindings;
    for (const auto& binding : vi.bindings) {
        ASSERT(binding.divisor == 0);
        vertex_bindings.Push(vk::VertexInputBindingDescription(binding.index, binding.stride));
    }

    StaticVector<vk::VertexInputAttributeDescription, Maxwell::NumVertexArrays> vertex_attributes;
    for (const auto& attribute : vi.attributes) {
        vertex_attributes.Push(vk::VertexInputAttributeDescription(
            attribute.index, attribute.buffer,
            MaxwellToVK::VertexFormat(attribute.type, attribute.size), attribute.offset));
    }

    const vk::PipelineVertexInputStateCreateInfo vertex_input_ci(
        {}, static_cast<u32>(vertex_bindings.Size()), vertex_bindings.Data(),
        static_cast<u32>(vertex_attributes.Size()), vertex_attributes.Data());

    const vk::PrimitiveTopology primitive_topology = MaxwellToVK::PrimitiveTopology(ia.topology);
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly_ci({}, primitive_topology,
                                                                     ia.primitive_restart_enable);

    const vk::Viewport viewport(vs.x, vs.y, vs.width, vs.height, 0.0f, 1.0f);
    // TODO(Rodrigo): Read scissor values instead of using viewport
    const vk::Rect2D scissor(
        {0, 0}, {static_cast<u32>(std::abs(0x1000)), static_cast<u32>(std::abs(0x1000))});
    const vk::PipelineViewportStateCreateInfo viewport_state_ci({}, 1, &viewport, 1, &scissor);

    // TODO(Rodrigo): Find out what's the default register value for front face
    const vk::PipelineRasterizationStateCreateInfo rasterizer_ci(
        {}, false, false, vk::PolygonMode::eFill,
        rs.cull_enable ? MaxwellToVK::CullFace(rs.cull_face) : vk::CullModeFlagBits::eNone,
        rs.cull_enable ? MaxwellToVK::FrontFace(rs.front_face) : vk::FrontFace::eCounterClockwise,
        false, 0.0f, 0.0f, 0.0f, 1.0f);

    const vk::PipelineMultisampleStateCreateInfo multisampling_ci(
        {}, vk::SampleCountFlagBits::e1, false, 0.0f, nullptr, false, false);

    const vk::CompareOp depth_test_compare = ds.depth_test_enable
                                                 ? MaxwellToVK::ComparisonOp(ds.depth_test_function)
                                                 : vk::CompareOp::eAlways;

    const vk::PipelineDepthStencilStateCreateInfo depth_stencil_ci(
        {}, ds.depth_test_enable, ds.depth_write_enable, depth_test_compare, ds.depth_bounds_enable,
        ds.stencil_enable, GetStencilFaceState(ds.front_stencil),
        GetStencilFaceState(ds.back_stencil), ds.depth_bounds_min, ds.depth_bounds_max);

    std::array<vk::PipelineColorBlendAttachmentState, Maxwell::NumRenderTargets> cb_attachments;
    // TODO(Rodrigo): Change this when multiple color attachments are supported
    // cd.independent_blend ? cb_attachments.size() : 1
    const std::size_t blend_attachment_count = 1;
    for (std::size_t i = 0; i < blend_attachment_count; ++i) {
        constexpr std::array<vk::ColorComponentFlagBits, 4> component_table = {
            vk::ColorComponentFlagBits::eR, vk::ColorComponentFlagBits::eG,
            vk::ColorComponentFlagBits::eB, vk::ColorComponentFlagBits::eA};
        const auto& blend = cd.attachments[i];

        vk::ColorComponentFlags color_components{};
        for (std::size_t i = 0; i < component_table.size(); ++i) {
            if (blend.components[i])
                color_components |= component_table[i];
        }

        cb_attachments[i] = vk::PipelineColorBlendAttachmentState(
            blend.enable, MaxwellToVK::BlendFactor(blend.src_rgb_func),
            MaxwellToVK::BlendFactor(blend.dst_rgb_func),
            MaxwellToVK::BlendEquation(blend.rgb_equation),
            MaxwellToVK::BlendFactor(blend.src_a_func), MaxwellToVK::BlendFactor(blend.dst_a_func),
            MaxwellToVK::BlendEquation(blend.a_equation), color_components);
    }
    const vk::PipelineColorBlendStateCreateInfo color_blending_ci(
        {}, false, vk::LogicOp::eCopy, static_cast<u32>(blend_attachment_count),
        cb_attachments.data(), cd.blend_constants);

    StaticVector<vk::PipelineShaderStageCreateInfo, Maxwell::MaxShaderStage> shader_stages;
    for (std::size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
        const auto& shader = pipeline.shaders[stage];
        if (shader == nullptr)
            continue;

        shader_stages.Push(vk::PipelineShaderStageCreateInfo(
            {}, MaxwellToVK::ShaderStage(static_cast<Maxwell::ShaderStage>(stage)),
            shader->GetHandle(primitive_topology), "main", nullptr));
    }

    const vk::GraphicsPipelineCreateInfo create_info(
        {}, static_cast<u32>(shader_stages.Size()), shader_stages.Data(), &vertex_input_ci,
        &input_assembly_ci, nullptr, &viewport_state_ci, &rasterizer_ci, &multisampling_ci,
        &depth_stencil_ci, &color_blending_ci, nullptr, pipeline.layout, renderpass, 0);

    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    return dev.createGraphicsPipelineUnique(nullptr, create_info, nullptr, dld);
}

} // namespace Vulkan
