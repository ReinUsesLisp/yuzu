// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <memory>

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
    std::array<View, Maxwell::NumRenderTargets> color_views;
    View zeta_view;
    u32 width;
    u32 height;
};

void PipelineState::Reset() {
    vertex_bindings.Clear();
    descriptor_bindings_count = 0;
    buffer_info_count = 0;
    image_info_count = 0;
}

void PipelineState::AssignDescriptorSet(u32 stage, vk::DescriptorSet descriptor_set) {
    // A null descriptor set means that the stage is not using descriptors, it must be skipped.
    if (descriptor_set) {
        descriptor_sets[stage] = descriptor_set;
    }
}

void PipelineState::AddVertexBinding(vk::Buffer buffer, vk::DeviceSize offset) {
    vertex_bindings.Push({buffer, offset});
}

void PipelineState::SetIndexBinding(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) {
    index_buffer = buffer;
    index_offset = offset;
    index_type = type;
}

std::tuple<vk::WriteDescriptorSet&, vk::DescriptorBufferInfo&>
PipelineState::CaptureDescriptorWriteBuffer() {
    const u32 desc_index = descriptor_bindings_count++;
    const u32 info_index = buffer_info_count++;
    ASSERT(desc_index < static_cast<u32>(MAX_DESCRIPTOR_WRITES));
    ASSERT(info_index < static_cast<u32>(MAX_DESCRIPTOR_BUFFERS));

    return {descriptor_bindings[desc_index], buffer_infos[info_index]};
}

std::tuple<vk::WriteDescriptorSet&, vk::DescriptorImageInfo&>
PipelineState::CaptureDescriptorWriteImage() {
    const u32 desc_index = descriptor_bindings_count++;
    const u32 info_index = image_info_count++;
    ASSERT(desc_index < static_cast<u32>(MAX_DESCRIPTOR_WRITES));
    ASSERT(info_index < static_cast<u32>(MAX_DESCRIPTOR_IMAGES));

    return {descriptor_bindings[desc_index], image_infos[info_index]};
}

void PipelineState::UpdateDescriptorSets(const VKDevice& device) const {
    const auto dev = device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    dev.updateDescriptorSets(descriptor_bindings_count, descriptor_bindings.data(), 0, nullptr,
                             dld);
}

void PipelineState::BindDescriptors(vk::CommandBuffer cmdbuf, vk::PipelineLayout layout,
                                    const vk::DispatchLoaderDynamic& dld) const {
    for (std::size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
        if (const auto descriptor_set = descriptor_sets[stage]; descriptor_set) {
            cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout,
                                      static_cast<u32>(stage), 1, &descriptor_set, 0, nullptr, dld);
        }
    }
}

void PipelineState::BindVertexBuffers(vk::CommandBuffer cmdbuf,
                                      const vk::DispatchLoaderDynamic& dld) const {
    // TODO(Rodrigo): Sort data and bindings to do this in a single call.
    for (u32 index = 0; index < vertex_bindings.Size(); ++index) {
        const auto [buffer, size] = vertex_bindings.Data()[index];
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
                                   VKMemoryManager& memory_manager, VKScheduler& sched)
    : VideoCore::RasterizerInterface(), system{system}, render_window{renderer},
      screen_info{screen_info}, device{device}, resource_manager{resource_manager},
      memory_manager{memory_manager}, sched{sched}, uniform_buffer_alignment{
                                                        device.GetUniformBufferAlignment()} {
    texture_cache = std::make_unique<VKTextureCache>(system, *this, device, resource_manager,
                                                     memory_manager, sched);
    pipeline_cache = std::make_unique<VKPipelineCache>(system, *this, device, sched);
    buffer_cache = std::make_unique<VKBufferCache>(system, *this, device,
                                                   memory_manager, sched, STREAM_BUFFER_SIZE);
    renderpass_cache = std::make_unique<VKRenderPassCache>(device);
    sampler_cache = std::make_unique<VKSamplerCache>(device);
}

RasterizerVulkan::~RasterizerVulkan() = default;

void RasterizerVulkan::DrawArrays() {
    if (accelerate_draw == AccelDraw::Disabled)
        return;

    const auto& gpu = system.GPU().Maxwell3D();
    const auto& regs = gpu.regs;
    const bool is_indexed = accelerate_draw == AccelDraw::Indexed;

    PipelineParams params;
    state.Reset();

    // Get renderpass parameters and get a draw renderpass from the cache
    const RenderPassParams renderpass_params = GetRenderPassParams();
    const vk::RenderPass renderpass = renderpass_cache->GetDrawRenderPass(renderpass_params);

    SyncDepthStencil(params);
    SyncInputAssembly(params);
    SyncColorBlending(params);
    SyncViewportState(params);
    SyncRasterizerState(params);

    // Calculate buffer size.
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

    const Pipeline pipeline = pipeline_cache->GetPipeline(params, renderpass_params, renderpass);

    const auto& dld = device.GetDispatchLoader();
    auto exctx = sched.GetExecutionContext();

    for (std::size_t stage = 0; stage < pipeline.shaders.size(); ++stage) {
        const Shader& shader = pipeline.shaders[stage];
        if (shader == nullptr)
            continue;

        auto& fence = exctx.GetFence();
        const auto descriptor_set = shader->CommitDescriptorSet(fence);
        SetupConstBuffers(state, shader, static_cast<Maxwell::ShaderStage>(stage), descriptor_set);
        exctx =
            SetupTextures(exctx, shader, static_cast<Maxwell::ShaderStage>(stage), descriptor_set);
        state.AssignDescriptorSet(static_cast<u32>(stage), descriptor_set);
    }

    FramebufferInfo fb_info;
    std::tie(fb_info, exctx) = ConfigureFramebuffers(exctx, renderpass);
    const View color_view = fb_info.color_views[0];
    const View zeta_view = fb_info.zeta_view;

    state.UpdateDescriptorSets(device);

    exctx = buffer_cache->Send(exctx);

    const auto cmdbuf = exctx.GetCommandBuffer();
    color_view->Transition(cmdbuf, vk::ImageLayout::eColorAttachmentOptimal,
                           vk::PipelineStageFlagBits::eColorAttachmentOutput,
                           vk::AccessFlagBits::eColorAttachmentRead |
                               vk::AccessFlagBits::eColorAttachmentWrite);

    if (zeta_view != nullptr) {
        zeta_view->Transition(cmdbuf, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                              vk::PipelineStageFlagBits::eLateFragmentTests,
                              vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                  vk::AccessFlagBits::eDepthStencilAttachmentWrite);
    }

    const vk::RenderPassBeginInfo renderpass_bi(
        renderpass, fb_info.framebuffer, {{0, 0}, {fb_info.width, fb_info.height}}, 0, nullptr);
    cmdbuf.beginRenderPass(renderpass_bi, vk::SubpassContents::eInline, dld);
    {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.handle, dld);
        state.BindDescriptors(cmdbuf, pipeline.layout, dld);
        state.BindVertexBuffers(cmdbuf, dld);

        if (is_indexed) {
            state.BindIndexBuffer(cmdbuf, dld);
            cmdbuf.drawIndexed(regs.index_array.count, 1, 0, regs.vb_element_base, 0, dld);
        } else {
            cmdbuf.draw(regs.vertex_buffer.count, 1, regs.vertex_buffer.first, 0, dld);
        }
    }
    cmdbuf.endRenderPass(dld);
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

    auto exctx = sched.GetExecutionContext();
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

bool RasterizerVulkan::AccelerateDrawBatch(bool is_indexed) {
    accelerate_draw = is_indexed ? AccelDraw::Indexed : AccelDraw::Arrays;
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

std::tuple<FramebufferInfo, VKExecutionContext> RasterizerVulkan::ConfigureFramebuffers(
    VKExecutionContext exctx, vk::RenderPass renderpass, bool using_color_fb, bool using_zeta_fb,
    bool preserve_contents) {
    const auto& regs = system.GPU().Maxwell3D().regs;

    View color_view{}, zeta_view{};
    if (using_color_fb) {
        std::tie(color_view, exctx) =
            texture_cache->GetColorBufferSurface(exctx, 0, preserve_contents);
    }
    if (using_zeta_fb) {
        std::tie(zeta_view, exctx) = texture_cache->GetDepthBufferSurface(exctx, preserve_contents);
    }

    FramebufferCacheKey fbkey;
    fbkey.renderpass = renderpass;
    fbkey.width = std::numeric_limits<u32>::max();
    fbkey.height = std::numeric_limits<u32>::max();

    if (color_view != nullptr) {
        color_view->MarkAsModified(true);
        fbkey.views.Push(color_view->GetHandle());
        fbkey.width = std::min(fbkey.width, color_view->GetWidth());
        fbkey.height = std::min(fbkey.height, color_view->GetHeight());
    }
    if (zeta_view != nullptr) {
        zeta_view->MarkAsModified(true);
        fbkey.views.Push(zeta_view->GetHandle());
        fbkey.width = std::min(fbkey.width, zeta_view->GetWidth());
        fbkey.height = std::min(fbkey.height, zeta_view->GetHeight());
    }

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
    info.color_views[0] = color_view;
    info.zeta_view = zeta_view;
    info.width = fbkey.width;
    info.height = fbkey.height;
    return {info, exctx};
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

void RasterizerVulkan::SetupConstBuffers(PipelineState& state, const Shader& shader,
                                         Maxwell::ShaderStage stage,
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

            if (size > MaxConstbufferSize) {
                LOG_CRITICAL(HW_GPU, "indirect constbuffer size {} exceeds maximum {}", size,
                             MaxConstbufferSize);
                size = MaxConstbufferSize;
            }
        } else {
            // Buffer is accessed directly, upload just what we use
            size = used_buffer.GetSize() * sizeof(float);
        }

        // Align the actual size so it ends up being a multiple of vec4 to meet the OpenGL
        // std140 UBO alignment requirements.
        size = Common::AlignUp(size, 4 * sizeof(float));
        ASSERT_MSG(size <= MaxConstbufferSize, "Constant buffer is too big");

        const auto offset =
            buffer_cache->UploadMemory(buffer.address, size, uniform_buffer_alignment);

        auto [write, buffer_info] = state.CaptureDescriptorWriteBuffer();
        buffer_info = vk::DescriptorBufferInfo(buffer_cache->GetBuffer(), offset,
                                               static_cast<vk::DeviceSize>(size));
        write = vk::WriteDescriptorSet(descriptor_set, current_binding, 0, 1,
                                       vk::DescriptorType::eUniformBuffer, nullptr, &buffer_info,
                                       nullptr);
    }
}

VKExecutionContext RasterizerVulkan::SetupTextures(VKExecutionContext exctx, const Shader& shader,
                                                   Maxwell::ShaderStage stage,
                                                   vk::DescriptorSet descriptor_set) {
    const auto& gpu = system.GPU().Maxwell3D();
    const auto& entries = shader->GetEntries().samplers;
    const u32 base_binding = shader->GetEntries().samplers_base_binding;

    for (u32 bindpoint = 0; bindpoint < static_cast<u32>(entries.size()); ++bindpoint) {
        const auto& entry = entries[bindpoint];
        const auto texture = gpu.GetStageTexture(stage, entry.GetOffset());
        const u32 current_binding = base_binding + bindpoint;

        View view;
        std::tie(view, exctx) = texture_cache->GetTextureSurface(exctx, texture);
        UNIMPLEMENTED_IF(view == nullptr);
        const vk::ImageView image_view =
            view->GetHandle(entry.GetType(), texture.tic.x_source, texture.tic.y_source,
                            texture.tic.z_source, texture.tic.w_source, entry.IsArray());

        constexpr auto pipeline_stage = vk::PipelineStageFlagBits::eAllGraphics;
        view->Transition(exctx.GetCommandBuffer(), vk::ImageLayout::eShaderReadOnlyOptimal,
                         pipeline_stage, vk::AccessFlagBits::eShaderRead);

        const auto [write, image_info] = state.CaptureDescriptorWriteImage();
        image_info = vk::DescriptorImageInfo(sampler_cache->GetSampler(texture.tsc), image_view,
                                             vk::ImageLayout::eShaderReadOnlyOptimal);
        write = vk::WriteDescriptorSet(descriptor_set, current_binding, 0, 1,
                                       vk::DescriptorType::eCombinedImageSampler, &image_info,
                                       nullptr, nullptr);
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
    const auto& regs = system.GPU().Maxwell3D().regs;

    RenderPassParams renderpass_params;
    if ((renderpass_params.has_zeta = regs.zeta_enable)) {
        renderpass_params.zeta_component_type =
            VideoCore::Surface::ComponentTypeFromDepthFormat(regs.zeta.format);
        renderpass_params.zeta_pixel_format =
            VideoCore::Surface::PixelFormatFromDepthFormat(regs.zeta.format);
    }

    // TODO(Rodrigo): Support multiple attachments
    RenderPassParams::ColorAttachment attachment;
    attachment.index = 0;
    attachment.pixel_format =
        VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[0].format);
    attachment.component_type =
        VideoCore::Surface::ComponentTypeFromRenderTarget(regs.rt[0].format);
    renderpass_params.color_map.Push(attachment);

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
    ds.depth_bounds_min = 0.f;
    ds.depth_bounds_max = 0.f;
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
    cd.independent_blend = regs.independent_blend_enable == 1;

    if (!cd.independent_blend) {
        auto& blend = cd.attachments[0];
        const auto& src = regs.blend;

        blend.enable = src.enable[0] != 0;
        if (blend.enable) {
            blend.rgb_equation = src.equation_rgb;
            blend.src_rgb_func = src.factor_source_rgb;
            blend.dst_rgb_func = src.factor_dest_rgb;
            blend.a_equation = src.equation_a;
            blend.src_a_func = src.factor_source_a;
            blend.dst_a_func = src.factor_dest_a;
            // TODO(Rodrigo): Read from registers
            blend.components = {true, true, true, true};
        }
        return;
    }

    for (std::size_t i = 0; i < Tegra::Engines::Maxwell3D::Regs::NumRenderTargets; i++) {
        auto& blend = cd.attachments[i];
        const auto& src = regs.independent_blend[i];

        blend.enable = regs.blend.enable[i] != 0;
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
