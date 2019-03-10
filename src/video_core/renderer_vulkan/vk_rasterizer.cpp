// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <memory>
#include <vector>

#include <boost/functional/hash.hpp>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/static_vector.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_global_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_resource_manager.h"
#include "video_core/renderer_vulkan/vk_sampler_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct FramebufferInfo {
    vk::Framebuffer framebuffer;
    View zeta_view;
    u32 width;
    u32 height;
};

void PipelineState::Reset() {
    vertex_bindings.clear();
    update_entries.clear();
}

void PipelineState::AddVertexBinding(vk::Buffer buffer, vk::DeviceSize offset) {
    vertex_bindings.emplace_back(buffer, offset);
}

void PipelineState::SetIndexBinding(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) {
    index_buffer = buffer;
    index_offset = offset;
    index_type = type;
}

void PipelineState::AddDescriptor(vk::DescriptorSet descriptor_set, u32 current_binding,
                                  vk::DescriptorType descriptor_type, vk::Buffer buffer, u64 offset,
                                  std::size_t size) {
    update_entries.emplace_back(
        vk::DescriptorBufferInfo(buffer, offset, static_cast<vk::DeviceSize>(size)));
}

void PipelineState::AddDescriptor(vk::DescriptorSet descriptor_set, u32 current_binding,
                                  vk::DescriptorType descriptor_type, vk::Sampler sampler,
                                  vk::ImageView image_view, vk::ImageLayout image_layout) {
    update_entries.emplace_back(vk::DescriptorImageInfo(sampler, image_view, image_layout));
}

void PipelineState::UpdateDescriptorSet(
    const VKDevice& device, vk::DescriptorSet descriptor_set,
    vk::DescriptorUpdateTemplate descriptor_update_template) const {
    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    dev.updateDescriptorSetWithTemplate(descriptor_set, descriptor_update_template,
                                        update_entries.data(), dld);
}

void PipelineState::BindVertexBuffers(vk::CommandBuffer cmdbuf,
                                      const vk::DispatchLoaderDynamic& dld) const {
    // TODO(Rodrigo): Sort data and bindings to do this in a single call.
    for (u32 index = 0; index < static_cast<u32>(vertex_bindings.size()); ++index) {
        const auto [buffer, size] = vertex_bindings[index];
        cmdbuf.bindVertexBuffers(index, {buffer}, {size}, dld);
    }
}

void PipelineState::BindIndexBuffer(vk::CommandBuffer cmdbuf,
                                    const vk::DispatchLoaderDynamic& dld) const {
    DEBUG_ASSERT(index_buffer && index_offset != 0);
    cmdbuf.bindIndexBuffer(index_buffer, index_offset, index_type, dld);
}

RasterizerVulkan::RasterizerVulkan(Core::System& system, Core::Frontend::EmuWindow& renderer,
                                   VKScreenInfo& screen_info, const VKDevice& device,
                                   VKResourceManager& resource_manager,
                                   VKMemoryManager& memory_manager, VKScheduler& scheduler)
    : VideoCore::RasterizerInterface(), system{system}, render_window{renderer},
      screen_info{screen_info}, device{device}, resource_manager{resource_manager},
      memory_manager{memory_manager}, scheduler{scheduler},
      uniform_buffer_alignment{device.GetUniformBufferAlignment()} {
    texture_cache = std::make_unique<VKTextureCache>(system, *this, device, resource_manager,
                                                     memory_manager, scheduler);
    pipeline_cache = std::make_unique<VKPipelineCache>(system, *this, device, scheduler);
    buffer_cache = std::make_unique<VKBufferCache>(system, *this, device, memory_manager, scheduler,
                                                   STREAM_BUFFER_SIZE);
    global_cache = std::make_unique<VKGlobalCache>(system, *this, device, memory_manager);

    renderpass_cache = std::make_unique<VKRenderPassCache>(device);
    sampler_cache = std::make_unique<VKSamplerCache>(device);
}

RasterizerVulkan::~RasterizerVulkan() = default;

void RasterizerVulkan::DrawArrays() {
    auto exctx = scheduler.GetExecutionContext();

    PipelineParams params;
    PrepareDraw();

    SyncFixedPipeline(params);

    std::array<View, Maxwell::NumRenderTargets> color_attachments;
    std::tie(color_attachments, exctx) = GetColorAttachments(exctx);

    // Get renderpass parameters and get a renderpass from the cache.
    const auto renderpass_params = GetRenderPassParams();
    const auto renderpass = renderpass_cache->GetRenderPass(renderpass_params);

    SetupGeometry(params);

    const auto [shaders, shader_addresses] = pipeline_cache->GetShaders();

    const Pipeline pipeline = pipeline_cache->GetPipeline(
        params, renderpass_params, shaders, shader_addresses, renderpass, exctx.GetFence());

    exctx = SetupShaderDescriptors(exctx, shaders, pipeline.descriptor_set,
                                   pipeline.descriptor_template);

    exctx = buffer_cache->Send(exctx);

    FramebufferInfo fbinfo;
    std::tie(fbinfo, exctx) = ConfigureFramebuffers(exctx, color_attachments, renderpass);

    for (const View sampled_view : sampled_views) {
        constexpr auto pipeline_stage = vk::PipelineStageFlagBits::eAllGraphics;
        sampled_view->Transition(exctx.GetCommandBuffer(), vk::ImageLayout::eShaderReadOnlyOptimal,
                                 pipeline_stage, vk::AccessFlagBits::eShaderRead);
    }

    for (const View color_view : color_attachments) {
        if (color_view == nullptr)
            continue;
        // TODO(Rodrigo): Optimize this in some way that's not O(n^2)
        const bool texception = std::any_of(sampled_views.begin(), sampled_views.end(),
                                            [color_view](const auto& sampled_view) {
                                                return color_view->IsOverlapping(sampled_view);
                                            });
        const auto image_layout = texception ? vk::ImageLayout::eSharedPresentKHR
                                             : vk::ImageLayout::eColorAttachmentOptimal;
        color_view->Transition(exctx.GetCommandBuffer(), image_layout,
                               vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::AccessFlagBits::eColorAttachmentRead |
                                   vk::AccessFlagBits::eColorAttachmentWrite);
    }

    if (const View zeta_view = fbinfo.zeta_view; zeta_view != nullptr) {
        zeta_view->Transition(exctx.GetCommandBuffer(),
                              vk::ImageLayout::eDepthStencilAttachmentOptimal,
                              vk::PipelineStageFlagBits::eLateFragmentTests,
                              vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                  vk::AccessFlagBits::eDepthStencilAttachmentWrite);
    }

    DispatchDraw(exctx, pipeline.layout, pipeline.descriptor_set, pipeline.handle, renderpass,
                 fbinfo.framebuffer, fbinfo.width, fbinfo.height);
}

void RasterizerVulkan::Clear() {
    const auto& regs = system.GPU().Maxwell3D().regs;
    const bool use_color = regs.clear_buffers.R || regs.clear_buffers.G || regs.clear_buffers.B ||
                           regs.clear_buffers.A;
    const bool use_depth = regs.clear_buffers.Z;
    const bool use_stencil = regs.clear_buffers.S;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }

    auto exctx = scheduler.GetExecutionContext();
    const auto& dld = device.GetDispatchLoader();

    if (use_color) {
        View color_view{};
        std::tie(color_view, exctx) =
            texture_cache->GetColorBufferSurface(exctx, regs.clear_buffers.RT.Value(), false);

        const auto cmdbuf = exctx.GetCommandBuffer();
        color_view->Transition(cmdbuf, vk::ImageLayout::eTransferDstOptimal,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::AccessFlagBits::eTransferWrite);

        const vk::ClearColorValue clear(std::array<float, 4>{
            regs.clear_color[0], regs.clear_color[1], regs.clear_color[2], regs.clear_color[3]});
        cmdbuf.clearColorImage(color_view->GetImage(), vk::ImageLayout::eTransferDstOptimal, clear,
                               color_view->GetImageSubresourceRange(), dld);
    }
    if (use_depth || use_stencil) {
        View zeta_surface{};
        std::tie(zeta_surface, exctx) = texture_cache->GetDepthBufferSurface(exctx, false);

        const auto cmdbuf = exctx.GetCommandBuffer();
        zeta_surface->Transition(cmdbuf, vk::ImageLayout::eTransferDstOptimal,
                                 vk::PipelineStageFlagBits::eTransfer,
                                 vk::AccessFlagBits::eTransferWrite);

        const vk::ClearDepthStencilValue clear(regs.clear_depth,
                                               static_cast<u32>(regs.clear_stencil));
        cmdbuf.clearDepthStencilImage(zeta_surface->GetImage(),
                                      vk::ImageLayout::eTransferDstOptimal, clear,
                                      zeta_surface->GetImageSubresourceRange(), dld);
    }
}

void RasterizerVulkan::FlushAll() {}

void RasterizerVulkan::FlushRegion(Tegra::GPUVAddr addr, u64 size) {}

void RasterizerVulkan::InvalidateRegion(Tegra::GPUVAddr addr, u64 size) {
    texture_cache->InvalidateRegion(addr, size);
    pipeline_cache->InvalidateRegion(addr, size);
    buffer_cache->InvalidateRegion(addr, size);
    global_cache->InvalidateRegion(addr, size);
}

void RasterizerVulkan::FlushAndInvalidateRegion(Tegra::GPUVAddr addr, u64 size) {
    FlushRegion(addr, size);
    InvalidateRegion(addr, size);
}

bool RasterizerVulkan::AccelerateDisplay(const Tegra::FramebufferConfig& config,
                                         VAddr framebuffer_addr, u32 pixel_stride) {
    if (!framebuffer_addr) {
        return {};
    }

    const auto& surface{texture_cache->TryFindFramebufferSurface(framebuffer_addr)};
    if (!surface) {
        return {};
    }

    // Verify that the cached surface is the same size and format as the requested framebuffer
    const auto& params{surface->GetSurfaceParams()};
    const auto& pixel_format{
        VideoCore::Surface::PixelFormatFromGPUPixelFormat(config.pixel_format)};
    ASSERT_MSG(params.width == config.width, "Framebuffer width is different");
    ASSERT_MSG(params.height == config.height, "Framebuffer height is different");
    // ASSERT_MSG(params.pixel_format == pixel_format, "Framebuffer pixel_format is different");

    screen_info.image = surface;
    screen_info.width = surface->GetSurfaceParams().width;
    screen_info.height = surface->GetSurfaceParams().height;
    return true;
}

bool RasterizerVulkan::AccelerateDrawBatch(bool is_indexed_) {
    is_indexed = is_indexed_;
    DrawArrays();
    return true;
}

template <typename Map, typename Interval>
static constexpr auto RangeFromInterval(Map& map, const Interval& interval) {
    return boost::make_iterator_range(map.equal_range(interval));
}

void RasterizerVulkan::UpdatePagesCachedCount(Tegra::GPUVAddr addr, u64 size, int delta) {
    const u64 page_start{addr >> Memory::PAGE_BITS};
    const u64 page_end{(addr + size + Memory::PAGE_SIZE - 1) >> Memory::PAGE_BITS};

    // Interval maps will erase segments if count reaches 0, so if delta is negative we have to
    // subtract after iterating
    const auto pages_interval = CachedPageMap::interval_type::right_open(page_start, page_end);
    if (delta > 0)
        cached_pages.add({pages_interval, delta});

    for (const auto& pair : RangeFromInterval(cached_pages, pages_interval)) {
        const auto interval = pair.first & pages_interval;
        const int count = pair.second;

        const VAddr interval_start_addr = boost::icl::first(interval) << Memory::PAGE_BITS;
        const VAddr interval_end_addr = boost::icl::last_next(interval) << Memory::PAGE_BITS;
        const u64 interval_size = interval_end_addr - interval_start_addr;

        if (delta > 0 && count == delta)
            Memory::RasterizerMarkRegionCached(interval_start_addr, interval_size, true);
        else if (delta < 0 && count == -delta)
            Memory::RasterizerMarkRegionCached(interval_start_addr, interval_size, false);
        else
            ASSERT(count >= 0);
    }

    if (delta < 0)
        cached_pages.add({pages_interval, delta});
}

void RasterizerVulkan::PrepareDraw() {
    state.Reset();
    sampled_views.clear();
}

void RasterizerVulkan::SyncFixedPipeline(PipelineParams& params) {
    SyncDepthStencil(params);
    SyncInputAssembly(params);
    SyncColorBlending(params);
    SyncViewportState(params);
    SyncRasterizerState(params);
}

std::tuple<std::array<View, Maxwell::NumRenderTargets>, VKExecutionContext>
RasterizerVulkan::GetColorAttachments(VKExecutionContext exctx) {
    std::array<View, Maxwell::NumRenderTargets> color_attachments;
    for (std::size_t rt = 0; rt < Maxwell::NumRenderTargets; ++rt) {
        View attachment;
        std::tie(attachment, exctx) = texture_cache->GetColorBufferSurface(exctx, rt, true);
        color_attachments[rt] = attachment;
    }
    return {color_attachments, exctx};
}

std::tuple<FramebufferInfo, VKExecutionContext> RasterizerVulkan::ConfigureFramebuffers(
    VKExecutionContext exctx, const std::array<View, Maxwell::NumRenderTargets>& color_attachments,
    vk::RenderPass renderpass) {
    const auto& regs = system.GPU().Maxwell3D().regs;

    FramebufferCacheKey fbkey;
    fbkey.renderpass = renderpass;
    fbkey.width = std::numeric_limits<u32>::max();
    fbkey.height = std::numeric_limits<u32>::max();

    const auto MarkAsModifiedAndPush = [&fbkey](View view) {
        if (view == nullptr) {
            return;
        }
        view->MarkAsModified(true);
        fbkey.views.Push(view->GetHandle());
        fbkey.width = std::min(fbkey.width, view->GetWidth());
        fbkey.height = std::min(fbkey.height, view->GetHeight());
    };

    for (const auto color_attachment : color_attachments) {
        MarkAsModifiedAndPush(color_attachment);
    }

    View zeta_view{};
    std::tie(zeta_view, exctx) = texture_cache->GetDepthBufferSurface(exctx, true);
    MarkAsModifiedAndPush(zeta_view);

    const auto [fbentry, is_cache_miss] = framebuffer_cache.try_emplace(fbkey);
    auto& framebuffer = fbentry->second;
    if (is_cache_miss) {
        const vk::FramebufferCreateInfo framebuffer_ci(
            {}, fbkey.renderpass, static_cast<u32>(fbkey.views.Size()), fbkey.views.Data(),
            fbkey.width, fbkey.height, 1);
        const auto dev = device.GetLogical();
        const auto& dld = device.GetDispatchLoader();
        framebuffer = dev.createFramebufferUnique(framebuffer_ci, nullptr, dld);
    }

    FramebufferInfo info;
    info.framebuffer = *framebuffer;
    info.zeta_view = zeta_view;
    info.width = fbkey.width;
    info.height = fbkey.height;
    return {info, exctx};
}

void RasterizerVulkan::SetupGeometry(PipelineParams& params) {
    std::size_t buffer_size = CalculateVertexArraysSize();
    if (is_indexed) {
        buffer_size = Common::AlignUp<std::size_t>(buffer_size, 4) + CalculateIndexBufferSize();
    }
    buffer_size += Maxwell::MaxConstBuffers * (MaxConstbufferSize + uniform_buffer_alignment);

    buffer_cache->Reserve(buffer_size);

    SetupVertexArrays(params);
    if (is_indexed) {
        SetupIndexBuffer();
    }
}

VKExecutionContext RasterizerVulkan::SetupShaderDescriptors(
    VKExecutionContext exctx, const std::array<Shader, Maxwell::MaxShaderStage>& shaders,
    vk::DescriptorSet descriptor_set, vk::DescriptorUpdateTemplate descriptor_template) {
    if (!descriptor_set || !descriptor_template) {
        return exctx;
    }

    for (std::size_t stage = 0; stage < std::size(shaders); ++stage) {
        const auto stage_enum = static_cast<Maxwell::ShaderStage>(stage);
        const Shader& shader = shaders[stage];
        if (!shader)
            continue;

        SetupConstBuffers(shader, stage_enum, descriptor_set);
        SetupGlobalBuffers(shader, stage_enum, descriptor_set);
        exctx = SetupTextures(exctx, shader, stage_enum, descriptor_set);
    }

    state.UpdateDescriptorSet(device, descriptor_set, descriptor_template);
    return exctx;
}

void RasterizerVulkan::DispatchDraw(VKExecutionContext exctx, vk::PipelineLayout pipeline_layout,
                                    vk::DescriptorSet descriptor_set, vk::Pipeline pipeline,
                                    vk::RenderPass renderpass, vk::Framebuffer framebuffer,
                                    u32 render_width, u32 render_height) {
    const auto& gpu = system.GPU().Maxwell3D();
    const auto& regs = gpu.regs;

    const auto& dld = device.GetDispatchLoader();
    const auto cmdbuf = exctx.GetCommandBuffer();

    const vk::RenderPassBeginInfo renderpass_bi(
        renderpass, framebuffer, {{0, 0}, {render_width, render_height}}, 0, nullptr);
    cmdbuf.beginRenderPass(renderpass_bi, vk::SubpassContents::eInline, dld);
    {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline, dld);
        cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout,
                                  VKShader::DESCRIPTOR_SET, 1, &descriptor_set, 0, nullptr, dld);

        state.BindVertexBuffers(cmdbuf, dld);

        const u32 instance = gpu.state.current_instance;
        if (is_indexed) {
            state.BindIndexBuffer(cmdbuf, dld);
            cmdbuf.drawIndexed(regs.index_array.count, 1, 0, regs.vb_element_base, instance, dld);
        } else {
            cmdbuf.draw(regs.vertex_buffer.count, 1, regs.vertex_buffer.first, instance, dld);
        }
    }
    cmdbuf.endRenderPass(dld);
}

void RasterizerVulkan::SetupVertexArrays(PipelineParams& params) {
    const auto& regs = system.GPU().Maxwell3D().regs;

    for (u32 index = 0; index < static_cast<u32>(Maxwell::NumVertexAttributes); ++index) {
        const auto& attrib = regs.vertex_attrib_format[index];

        // Ignore invalid attributes.
        if (!attrib.IsValid())
            continue;

        const auto& buffer = regs.vertex_array[attrib.buffer];
        LOG_TRACE(HW_GPU, "vertex attrib {}, count={}, size={}, type={}, offset={}, normalize={}",
                  index, attrib.ComponentCount(), attrib.SizeString(), attrib.TypeString(),
                  attrib.offset.Value(), attrib.IsNormalized());

        ASSERT(buffer.IsEnabled());

        PipelineParams::VertexAttribute attribute;
        attribute.index = index;
        attribute.buffer = attrib.buffer;
        attribute.type = attrib.type;
        attribute.size = attrib.size;
        attribute.offset = attrib.offset;
        params.vertex_input.attributes.Push(attribute);
    }

    for (u32 index = 0; index < static_cast<u32>(Maxwell::NumVertexArrays); ++index) {
        const auto& vertex_array = regs.vertex_array[index];
        if (!vertex_array.IsEnabled())
            continue;

        Tegra::GPUVAddr start = vertex_array.StartAddress();
        const Tegra::GPUVAddr end = regs.vertex_array_limit[index].LimitAddress();

        ASSERT(end > start);
        const std::size_t size = end - start + 1;
        const auto offset = buffer_cache->UploadMemory(start, size);

        PipelineParams::VertexBinding binding;
        binding.index = index;
        binding.stride = vertex_array.stride;
        binding.divisor = vertex_array.divisor;
        params.vertex_input.bindings.Push(binding);

        state.AddVertexBinding(buffer_cache->GetBuffer(), offset);
    }
}

void RasterizerVulkan::SetupIndexBuffer() {
    const auto& regs = system.GPU().Maxwell3D().regs;
    const auto offset =
        buffer_cache->UploadMemory(regs.index_array.IndexStart(), CalculateIndexBufferSize());
    state.SetIndexBinding(buffer_cache->GetBuffer(), offset,
                          MaxwellToVK::IndexFormat(regs.index_array.format));
}

void RasterizerVulkan::SetupConstBuffers(const Shader& shader, Maxwell::ShaderStage stage,
                                         vk::DescriptorSet descriptor_set) {
    const auto& gpu = system.GPU().Maxwell3D();
    const auto& shader_stage = gpu.state.shader_stages[static_cast<std::size_t>(stage)];
    const auto& entries = shader->GetEntries().const_buffers;
    const u32 base_binding = shader->GetEntries().const_buffers_base_binding;

    for (u32 bindpoint = 0; bindpoint < static_cast<u32>(entries.size()); ++bindpoint) {
        const auto& used_buffer = entries[bindpoint];
        const auto& buffer = shader_stage.const_buffers[used_buffer.GetIndex()];
        const u32 current_binding = base_binding + bindpoint;

        std::size_t size = 0;

        if (used_buffer.IsIndirect()) {
            // Buffer is accessed indirectly, so upload the entire thing
            size = buffer.size;
        } else {
            // Buffer is accessed directly, upload just what we use
            size = used_buffer.GetSize() * sizeof(float);
        }
        ASSERT(size <= MaxConstbufferSize);

        // Align the actual size so it ends up being a multiple of vec4 to meet the OpenGL
        // std140 UBO alignment requirements.
        // ???
        size = Common::AlignUp(size, 4 * sizeof(float));
        ASSERT_MSG(size <= MaxConstbufferSize, "Constant buffer is too big");

        const auto offset =
            buffer_cache->UploadMemory(buffer.address, size, uniform_buffer_alignment);

        state.AddDescriptor(descriptor_set, current_binding, vk::DescriptorType::eUniformBuffer,
                            buffer_cache->GetBuffer(), offset, size);
    }
}

void RasterizerVulkan::SetupGlobalBuffers(const Shader& shader, Maxwell::ShaderStage stage,
                                          vk::DescriptorSet descriptor_set) {
    const auto& entries = shader->GetEntries().global_buffers;
    for (u32 bindpoint = 0; bindpoint < static_cast<u32>(entries.size()); ++bindpoint) {
        const auto& entry = entries[bindpoint];
        const u32 current_bindpoint = shader->GetEntries().global_buffers_base_binding;
        const auto region = global_cache->GetGlobalRegion(entry, stage);

        state.AddDescriptor(descriptor_set, current_bindpoint, vk::DescriptorType::eStorageBuffer,
                            region->GetBufferHandle(), 0, region->GetSizeInBytes());
    }
}

VKExecutionContext RasterizerVulkan::SetupTextures(VKExecutionContext exctx, const Shader& shader,
                                                   Maxwell::ShaderStage stage,
                                                   vk::DescriptorSet descriptor_set) {
    const auto& gpu = system.GPU().Maxwell3D();
    const auto& entries = shader->GetEntries().samplers;
    const u32 base_binding = shader->GetEntries().samplers_base_binding;

    for (std::size_t bindpoint = 0; bindpoint < entries.size(); ++bindpoint) {
        const auto& entry = entries[bindpoint];
        const auto texture = gpu.GetStageTexture(stage, entry.GetOffset());

        View view;
        std::tie(view, exctx) = texture_cache->GetTextureSurface(exctx, texture);
        UNIMPLEMENTED_IF(view == nullptr);
        sampled_views.push_back(view);

        const vk::ImageView image_view =
            view->GetHandle(entry.GetType(), texture.tic.x_source, texture.tic.y_source,
                            texture.tic.z_source, texture.tic.w_source, entry.IsArray());

        const u32 current_binding = base_binding + static_cast<u32>(bindpoint);
        state.AddDescriptor(descriptor_set, current_binding,
                            vk::DescriptorType::eCombinedImageSampler,
                            sampler_cache->GetSampler(texture.tsc), image_view,
                            vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    return exctx;
}

std::size_t RasterizerVulkan::CalculateVertexArraysSize() const {
    const auto& regs = system.GPU().Maxwell3D().regs;

    std::size_t size = 0;
    for (u32 index = 0; index < Maxwell::NumVertexArrays; ++index) {
        if (!regs.vertex_array[index].IsEnabled())
            continue;
        // This implementation assumes that all attributes are used.

        const Tegra::GPUVAddr start = regs.vertex_array[index].StartAddress();
        const Tegra::GPUVAddr end = regs.vertex_array_limit[index].LimitAddress();

        ASSERT(end > start);
        size += end - start + 1;
    }
    return size;
}

std::size_t RasterizerVulkan::CalculateIndexBufferSize() const {
    const auto& regs = system.GPU().Maxwell3D().regs;
    return static_cast<std::size_t>(regs.index_array.count) *
           static_cast<std::size_t>(regs.index_array.FormatSizeInBytes());
}

RenderPassParams RasterizerVulkan::GetRenderPassParams() const {
    using namespace VideoCore::Surface;
    const auto& regs = system.GPU().Maxwell3D().regs;

    RenderPassParams renderpass_params;
    if ((renderpass_params.has_zeta = regs.zeta_enable)) {
        renderpass_params.zeta_component_type = ComponentTypeFromDepthFormat(regs.zeta.format);
        renderpass_params.zeta_pixel_format = PixelFormatFromDepthFormat(regs.zeta.format);
    }

    for (std::size_t rt = 0; rt < static_cast<std::size_t>(regs.rt_control.count); ++rt) {
        const auto& rendertarget = regs.rt[rt];
        if (rendertarget.Address() == 0 || rendertarget.format == Tegra::RenderTargetFormat::NONE)
            continue;
        RenderPassParams::ColorAttachment attachment;
        attachment.index = static_cast<u32>(rt);
        attachment.pixel_format = PixelFormatFromRenderTargetFormat(rendertarget.format);
        attachment.component_type = ComponentTypeFromRenderTarget(rendertarget.format);
        renderpass_params.color_map.Push(attachment);
    }

    return renderpass_params;
}

void RasterizerVulkan::SyncDepthStencil(PipelineParams& params) {
    const auto& regs = system.GPU().Maxwell3D().regs;
    auto& ds = params.depth_stencil;

    ds.depth_test_function = regs.depth_test_func;
    ds.depth_test_enable = regs.depth_test_enable == 1;
    ds.depth_write_enable = regs.depth_write_enabled == 1;

    ds.stencil_enable = regs.stencil_enable == 1;

    auto& front = ds.front_stencil;
    front.test_func = regs.stencil_front_func_func;
    front.test_ref = regs.stencil_front_func_ref;
    front.test_mask = regs.stencil_front_func_mask;
    front.action_stencil_fail = regs.stencil_front_op_fail;
    front.action_depth_fail = regs.stencil_front_op_zfail;
    front.action_depth_pass = regs.stencil_front_op_zpass;
    front.write_mask = regs.stencil_front_mask;

    if (regs.stencil_two_side_enable) {
        auto& back = ds.back_stencil;
        back.test_func = regs.stencil_back_func_func;
        back.test_ref = regs.stencil_back_func_ref;
        back.test_mask = regs.stencil_back_func_mask;
        back.action_stencil_fail = regs.stencil_back_op_fail;
        back.action_depth_fail = regs.stencil_back_op_zfail;
        back.action_depth_pass = regs.stencil_back_op_zpass;
        back.write_mask = regs.stencil_back_mask;
    }

    // TODO(Rodrigo): Read from registers, luckily this is core in Vulkan unlike OpenGL
    ds.depth_bounds_enable = false;
    ds.depth_bounds_min = 0.0f;
    ds.depth_bounds_max = 0.0f;
}

void RasterizerVulkan::SyncInputAssembly(PipelineParams& params) {
    const auto& regs = system.GPU().Maxwell3D().regs;
    auto& ia = params.input_assembly;

    ia.topology = regs.draw.topology;
    // ia.primitive_restart_enable = ;
}

void RasterizerVulkan::SyncColorBlending(PipelineParams& params) {
    const auto& regs = system.GPU().Maxwell3D().regs;
    auto& cd = params.color_blending;

    cd.blend_constants = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                          regs.blend_color.a};
    cd.attachments_count = regs.rt_control.count;

    for (std::size_t rt = 0; rt < regs.rt_control.count; ++rt) {
        auto& blend = cd.attachments[rt];
        if (regs.independent_blend_enable != 0) {
            blend.enable = regs.blend.enable[rt] != 0;
            if (!blend.enable)
                continue;
            const auto& src = regs.independent_blend[rt];
            blend.rgb_equation = src.equation_rgb;
            blend.src_rgb_func = src.factor_source_rgb;
            blend.dst_rgb_func = src.factor_dest_rgb;
            blend.a_equation = src.equation_a;
            blend.src_a_func = src.factor_source_a;
            blend.dst_a_func = src.factor_dest_a;
            // TODO(Rodrigo): Read from registers
            blend.components = {true, true, true, true};
        } else {
            const auto& src = regs.blend;
            blend.enable = src.enable[rt] != 0;
            if (!blend.enable)
                continue;
            blend.rgb_equation = src.equation_rgb;
            blend.src_rgb_func = src.factor_source_rgb;
            blend.dst_rgb_func = src.factor_dest_rgb;
            blend.a_equation = src.equation_a;
            blend.src_a_func = src.factor_source_a;
            blend.dst_a_func = src.factor_dest_a;
            // TODO(Rodrigo): Read from registers
            blend.components = {true, true, true, true};
        }
    }
}

void RasterizerVulkan::SyncViewportState(PipelineParams& params) {
    const auto& regs = system.GPU().Maxwell3D().regs;
    const auto& viewport = regs.viewport_transform[0];

    float scale_y = viewport.scale_y;
    if (regs.view_volume_clip_control.flip_y == 1) {
        scale_y = -scale_y;
    }

    auto& vs = params.viewport_state;
    vs.x = viewport.translate_x - viewport.scale_x;
    vs.y = viewport.translate_y - scale_y;
    vs.width = viewport.translate_x + viewport.scale_x - vs.x;
    vs.height = viewport.translate_y + scale_y - vs.y;
}

void RasterizerVulkan::SyncRasterizerState(PipelineParams& params) {
    const auto& regs = system.GPU().Maxwell3D().regs;
    auto& rs = params.rasterizer;

    rs.cull_enable = regs.cull.enabled != 0;
    rs.cull_face = regs.cull.cull_face;

    rs.front_face = regs.cull.front_face;

    // If the GPU is configured to flip the rasterizer triangles, then we need to flip the front and
    // back.
    /*if (regs.screen_y_control.triangle_rast_flip == 1) {
        if (rs.front_face == Maxwell::Cull::FrontFace::CounterClockWise)
            rs.front_face = Maxwell::Cull::FrontFace::ClockWise;
        else if (rs.front_face == Maxwell::Cull::FrontFace::ClockWise)
            rs.front_face = Maxwell::Cull::FrontFace::CounterClockWise;
    }*/
}

} // namespace Vulkan
