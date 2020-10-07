// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/wrapper.h"

namespace Vulkan {

using Tegra::Texture::TextureMipmapFilter;

namespace {

constexpr std::array ATTACHMENT_REFERENCES{
    VkAttachmentReference{0, VK_IMAGE_LAYOUT_GENERAL},
    VkAttachmentReference{1, VK_IMAGE_LAYOUT_GENERAL},
    VkAttachmentReference{2, VK_IMAGE_LAYOUT_GENERAL},
    VkAttachmentReference{3, VK_IMAGE_LAYOUT_GENERAL},
    VkAttachmentReference{4, VK_IMAGE_LAYOUT_GENERAL},
    VkAttachmentReference{5, VK_IMAGE_LAYOUT_GENERAL},
    VkAttachmentReference{6, VK_IMAGE_LAYOUT_GENERAL},
    VkAttachmentReference{7, VK_IMAGE_LAYOUT_GENERAL},
    VkAttachmentReference{8, VK_IMAGE_LAYOUT_GENERAL},
};

constexpr VkBorderColor ConvertBorderColor(const std::array<float, 4>& color) {
    if (color == std::array<float, 4>{0, 0, 0, 0}) {
        return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    } else if (color == std::array<float, 4>{0, 0, 0, 1}) {
        return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    } else if (color == std::array<float, 4>{1, 1, 1, 1}) {
        return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    }
    if (color[0] + color[1] + color[2] > 1.35f) {
        // If color elements are brighter than roughly 0.5 average, use white border
        return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    } else if (color[3] > 0.5f) {
        return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    } else {
        return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    }
}

[[nodiscard]] VkImageType ImageType(const VideoCommon::ImageType type) {
    switch (type) {
    case VideoCommon::ImageType::e1D:
        return VK_IMAGE_TYPE_1D;
    case VideoCommon::ImageType::e2D:
    case VideoCommon::ImageType::Linear:
    case VideoCommon::ImageType::Rect:
        return VK_IMAGE_TYPE_2D;
    case VideoCommon::ImageType::e3D:
        return VK_IMAGE_TYPE_3D;
    }
    UNREACHABLE_MSG("Invalid image type={}", static_cast<int>(type));
    return {};
}

[[nodiscard]] VkImageCreateInfo MakeImageCreateInfo(const VKDevice& device,
                                                    const VideoCommon::ImageInfo& info) {
    const auto format_info = MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, info.format);
    VkImageCreateFlags flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    if (info.type == VideoCommon::ImageType::e2D && info.resources.layers >= 6) {
        flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }
    if (info.type == VideoCommon::ImageType::e3D) {
        flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    }
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT;
    if (format_info.attachable) {
        switch (VideoCore::Surface::GetFormatType(info.format)) {
        case VideoCore::Surface::SurfaceType::ColorTexture:
            usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            break;
        case VideoCore::Surface::SurfaceType::Depth:
        case VideoCore::Surface::SurfaceType::DepthStencil:
            usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            break;
        default:
            UNREACHABLE_MSG("Invalid surface type");
        }
    }
    if (format_info.storage) {
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    return VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
        .imageType = ImageType(info.type),
        .format = format_info.format,
        .extent =
            {
                .width = info.size.width,
                .height = info.size.height,
                .depth = info.size.depth,
            },
        .mipLevels = info.resources.mipmaps,
        .arrayLayers = info.resources.layers,
        .samples = VK_SAMPLE_COUNT_1_BIT, // TODO
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
}

[[nodiscard]] VkImageAspectFlags ImageAspectMask(const VideoCommon::ImageInfo& info) {
    switch (VideoCore::Surface::GetFormatType(info.format)) {
    case VideoCore::Surface::SurfaceType::ColorTexture:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    case VideoCore::Surface::SurfaceType::Depth:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VideoCore::Surface::SurfaceType::DepthStencil:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
        UNREACHABLE_MSG("Invalid surface type");
        return {};
    }
}

[[nodiscard]] VkAttachmentDescription AttachmentDescription(const VKDevice& device,
                                                            const ImageView* image_view) {
    const auto pixel_format = image_view->format;
    return VkAttachmentDescription{
        .flags = VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT,
        .format = MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, pixel_format).format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
        .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
        .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
}

[[nodiscard]] VkComponentSwizzle ComponentSwizzle(u8 swizzle) {
    switch (static_cast<Tegra::Texture::SwizzleSource>(swizzle)) {
    case Tegra::Texture::SwizzleSource::Zero:
        return VK_COMPONENT_SWIZZLE_ZERO;
    case Tegra::Texture::SwizzleSource::R:
        return VK_COMPONENT_SWIZZLE_R;
    case Tegra::Texture::SwizzleSource::G:
        return VK_COMPONENT_SWIZZLE_G;
    case Tegra::Texture::SwizzleSource::B:
        return VK_COMPONENT_SWIZZLE_B;
    case Tegra::Texture::SwizzleSource::A:
        return VK_COMPONENT_SWIZZLE_A;
    case Tegra::Texture::SwizzleSource::OneFloat:
    case Tegra::Texture::SwizzleSource::OneInt:
        return VK_COMPONENT_SWIZZLE_ONE;
    }
    UNREACHABLE_MSG("Invalid swizzle={}", static_cast<int>(swizzle));
}

[[nodiscard]] VkImageViewType ImageViewType(VideoCommon::ImageViewType type) {
    switch (type) {
    case VideoCommon::ImageViewType::e1D:
        return VK_IMAGE_VIEW_TYPE_1D;
    case VideoCommon::ImageViewType::e2D:
        return VK_IMAGE_VIEW_TYPE_2D;
    case VideoCommon::ImageViewType::Cube:
        return VK_IMAGE_VIEW_TYPE_CUBE;
    case VideoCommon::ImageViewType::e3D:
        return VK_IMAGE_VIEW_TYPE_3D;
    case VideoCommon::ImageViewType::e1DArray:
        return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case VideoCommon::ImageViewType::e2DArray:
        return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case VideoCommon::ImageViewType::CubeArray:
        return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    case VideoCommon::ImageViewType::Rect:
        LOG_WARNING(Render_Vulkan, "Unnormalized image view type not supported");
        return VK_IMAGE_VIEW_TYPE_2D;
    case VideoCommon::ImageViewType::Buffer:
        UNREACHABLE_MSG("Texture buffers can't be image views");
        return VK_IMAGE_VIEW_TYPE_1D;
    }
    UNREACHABLE_MSG("Invalid image view type={}", static_cast<int>(type));
    return VK_IMAGE_VIEW_TYPE_2D;
}

void TransitionToGeneral(vk::CommandBuffer cmdbuf, VkImage image, VkImageAspectFlags aspect_mask) {
    const VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange =
            {
                .aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, barrier);
}

[[nodiscard]] VkImageSubresourceLayers MakeImageSubresourceLayers(
    VideoCommon::SubresourceLayers subresource, VkImageAspectFlags aspect_mask) {
    return VkImageSubresourceLayers{
        .aspectMask = aspect_mask,
        .mipLevel = subresource.base_mipmap,
        .baseArrayLayer = subresource.base_layer,
        .layerCount = subresource.num_layers,
    };
}

[[nodiscard]] VkOffset3D MakeOffset3D(VideoCommon::Offset3D offset3d) {
    return VkOffset3D{
        .x = static_cast<s32>(offset3d.x),
        .y = static_cast<s32>(offset3d.y),
        .z = static_cast<s32>(offset3d.z),
    };
}

[[nodiscard]] VkExtent3D MakeExtent3D(VideoCommon::Extent3D extent3d) {
    return VkExtent3D{
        .width = extent3d.width,
        .height = extent3d.height,
        .depth = extent3d.depth,
    };
}

[[nodiscard]] VkImageCopy MakeImageCopy(const VideoCommon::ImageCopy& copy,
                                        VkImageAspectFlags aspect_mask) noexcept {
    return VkImageCopy{
        .srcSubresource = MakeImageSubresourceLayers(copy.src_subresource, aspect_mask),
        .srcOffset = MakeOffset3D(copy.src_offset),
        .dstSubresource = MakeImageSubresourceLayers(copy.dst_subresource, aspect_mask),
        .dstOffset = MakeOffset3D(copy.dst_offset),
        .extent = MakeExtent3D(copy.extent),
    };
}

[[nodiscard]] std::vector<VkBufferImageCopy> TransformBufferImageCopies(
    std::span<const VideoCommon::BufferImageCopy> copies, VkImageAspectFlags aspect_mask) {
    struct Maker {
        constexpr VkBufferImageCopy operator()(const VideoCommon::BufferImageCopy& copy) const {
            return VkBufferImageCopy{
                .bufferOffset = copy.buffer_offset,
                .bufferRowLength = copy.buffer_row_length,
                .bufferImageHeight = copy.buffer_image_height,
                .imageSubresource =
                    {
                        .aspectMask = aspect_mask,
                        .mipLevel = copy.image_subresource.base_mipmap,
                        .baseArrayLayer = copy.image_subresource.base_layer,
                        .layerCount = copy.image_subresource.num_layers,
                    },
                .imageOffset =
                    {
                        .x = static_cast<s32>(copy.image_offset.x),
                        .y = static_cast<s32>(copy.image_offset.y),
                        .z = static_cast<s32>(copy.image_offset.z),
                    },
                .imageExtent =
                    {
                        .width = copy.image_extent.width,
                        .height = copy.image_extent.height,
                        .depth = copy.image_extent.depth,
                    },
            };
        }
        VkImageAspectFlags aspect_mask;
    };
    if (aspect_mask == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
        std::vector<VkBufferImageCopy> result(copies.size() * 2);
        std::ranges::transform(copies, result.begin(), Maker{VK_IMAGE_ASPECT_DEPTH_BIT});
        std::ranges::transform(copies, result.begin() + copies.size(),
                               Maker{VK_IMAGE_ASPECT_STENCIL_BIT});
        return result;
    } else {
        std::vector<VkBufferImageCopy> result(copies.size());
        std::ranges::transform(copies, result.begin(), Maker{aspect_mask});
        return result;
    }
}

} // Anonymous namespace

ImageBufferMap TextureCacheRuntime::MapUploadBuffer(size_t size) {
    const auto& buffer = staging_buffer_pool.GetUnusedBuffer(size, true);
    return ImageBufferMap{
        .handle = *buffer.handle,
        .map = buffer.commit->Map(size),
    };
}

void TextureCacheRuntime::CopyImage(Image& dst, Image& src,
                                    std::span<const VideoCommon::ImageCopy> copies) {
    std::vector<VkImageCopy> vk_copies(copies.size());
    const VkImageAspectFlags aspect_mask = dst.AspectMask();
    std::ranges::transform(copies, vk_copies.begin(), [aspect_mask](const auto& copy) {
        return MakeImageCopy(copy, aspect_mask);
    });
    const VkImage dst_image = dst.Handle();
    const VkImage src_image = src.Handle();
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([dst_image, src_image, aspect_mask, vk_copies](vk::CommandBuffer cmdbuf) {
        // TODO: Support ranged barriers
        const VkImageSubresourceRange subresource_range{
            .aspectMask = aspect_mask,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        };
        const std::array read_barriers{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = src_image,
                .subresourceRange = subresource_range,
            },
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst_image,
                .subresourceRange = subresource_range,
            },
        };
        const VkImageMemoryBarrier write_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst_image,
            .subresourceRange = subresource_range,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, {}, {}, read_barriers);
        cmdbuf.CopyImage(src_image, VK_IMAGE_LAYOUT_GENERAL, dst_image, VK_IMAGE_LAYOUT_GENERAL,
                         vk_copies);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               0, write_barrier);
    });
}

void TextureCacheRuntime::InsertUploadMemoryBarrier() {
    // scheduler.Record([](vk::CommandBuffer cmdbuf) {});
}

Image::Image(TextureCacheRuntime& runtime, const VideoCommon::ImageInfo& info, GPUVAddr gpu_addr,
             VAddr cpu_addr)
    : VideoCommon::ImageBase(info, gpu_addr, cpu_addr), scheduler{&runtime.scheduler},
      image(runtime.device.GetLogical().CreateImage(MakeImageCreateInfo(runtime.device, info))),
      commit(runtime.memory_manager.Commit(image, false)), aspect_mask(ImageAspectMask(info)) {}

void Image::UploadMemory(const ImageBufferMap& map, size_t buffer_offset,
                         std::span<const VideoCommon::BufferImageCopy> copies) {
    std::vector<VkBufferImageCopy> vk_copies = TransformBufferImageCopies(copies, aspect_mask);

    // TODO: Move this to another API
    scheduler->RequestOutsideRenderPassOperationContext();
    if (!initialized) {
        initialized = true;
        scheduler->Record([image = *image, aspect_mask = aspect_mask](vk::CommandBuffer cmdbuf) {
            TransitionToGeneral(cmdbuf, image, aspect_mask);
        });
    }
    scheduler->Record([buffer = map.handle, image = *image, aspect_mask = aspect_mask,
                       vk_copies](vk::CommandBuffer cmdbuf) {
        // TODO: Move this to another API
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        cmdbuf.CopyBufferToImage(buffer, image, VK_IMAGE_LAYOUT_GENERAL, vk_copies);

        // TODO: Move this to another API
        cmdbuf.PipelineBarrier(
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask =
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = 0,
                .dstQueueFamilyIndex = 0,
                .image = image,
                .subresourceRange =
                    {
                        .aspectMask = aspect_mask,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
            });
    });
}

void Image::DownloadMemory(const ImageBufferMap& map, size_t buffer_offset,
                           std::span<const VideoCommon::BufferImageCopy> copies) {
    std::vector<VkBufferImageCopy> vk_copies = TransformBufferImageCopies(copies, aspect_mask);
    scheduler->Record([buffer = map.handle, image = *image, aspect_mask = aspect_mask,
                       vk_copies](vk::CommandBuffer cmdbuf) {
        cmdbuf.CopyImageToBuffer(image, VK_IMAGE_LAYOUT_GENERAL, buffer, vk_copies);
    });
    scheduler->Finish();
}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageViewInfo& info,
                     ImageId image_id, Image& image)
    : VideoCommon::ImageViewBase{info, image.info, image_id} {
    const VkFormat format =
        MaxwellToVK::SurfaceFormat(runtime.device, FormatType::Optimal, info.format).format;
    const VkImageViewCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = image.Handle(),
        .viewType = VkImageViewType{},
        .format = format,
        .components =
            {
                .r = ComponentSwizzle(info.x_source),
                .g = ComponentSwizzle(info.y_source),
                .b = ComponentSwizzle(info.z_source),
                .a = ComponentSwizzle(info.w_source),
            },
        .subresourceRange =
            {
                .aspectMask = ImageAspectMask(image.info),
                .baseMipLevel = info.range.base.mipmap,
                .levelCount = info.range.extent.mipmaps,
                .baseArrayLayer = info.range.base.layer,
                .layerCount = info.range.extent.layers,
            },
    };
    const vk::Device& device = runtime.device.GetLogical();
    const auto create = [this, &device, &create_info](VideoCommon::ImageViewType type,
                                                      std::optional<u32> num_layers) {
        VkImageViewCreateInfo ci{create_info};
        ci.viewType = ImageViewType(type);
        if (num_layers) {
            ci.subresourceRange.layerCount = *num_layers;
        }
        views[static_cast<size_t>(type)] = device.CreateImageView(ci);
    };
    switch (info.type) {
    case VideoCommon::ImageViewType::e1D:
    case VideoCommon::ImageViewType::e1DArray:
        create(VideoCommon::ImageViewType::e1D, 1);
        create(VideoCommon::ImageViewType::e1DArray, std::nullopt);
        render_target = Handle(VideoCommon::ImageViewType::e1DArray);
        break;
    case VideoCommon::ImageViewType::e2D:
    case VideoCommon::ImageViewType::e2DArray:
        create(VideoCommon::ImageViewType::e2D, 1);
        create(VideoCommon::ImageViewType::e2DArray, std::nullopt);
        render_target = Handle(VideoCommon::ImageViewType::e2DArray);
        break;
    case VideoCommon::ImageViewType::e3D:
        create(VideoCommon::ImageViewType::e3D, std::nullopt);
        render_target = Handle(VideoCommon::ImageViewType::e3D);
        break;
    case VideoCommon::ImageViewType::Cube:
    case VideoCommon::ImageViewType::CubeArray:
        create(VideoCommon::ImageViewType::Cube, 6);
        create(VideoCommon::ImageViewType::CubeArray, std::nullopt);
        break;
    case VideoCommon::ImageViewType::Rect:
        UNIMPLEMENTED();
        break;
    case VideoCommon::ImageViewType::Buffer:
        UNIMPLEMENTED();
        break;
    }
}

ImageView::ImageView(TextureCacheRuntime&, const VideoCommon::NullImageParams& params)
    : VideoCommon::ImageViewBase{params} {}

Sampler::Sampler(TextureCacheRuntime& runtime, const Tegra::Texture::TSCEntry& tsc) {
    const auto& device = runtime.device;
    const bool arbitrary_borders = runtime.device.IsExtCustomBorderColorSupported();
    const std::array<float, 4> color = tsc.BorderColor();
    const VkSamplerCustomBorderColorCreateInfoEXT border{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
        .pNext = nullptr,
        .customBorderColor = std::bit_cast<VkClearColorValue>(color),
        .format = VK_FORMAT_UNDEFINED,
    };
    const VkSamplerReductionModeCreateInfoEXT reduction_ci{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT,
        .pNext = arbitrary_borders ? &border : nullptr,
        .reductionMode = MaxwellToVK::SamplerReduction(tsc.reduction_filter),
    };
    sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = &reduction_ci,
        .flags = 0,
        .magFilter = MaxwellToVK::Sampler::Filter(tsc.mag_filter),
        .minFilter = MaxwellToVK::Sampler::Filter(tsc.min_filter),
        .mipmapMode = MaxwellToVK::Sampler::MipmapMode(tsc.mipmap_filter),
        .addressModeU = MaxwellToVK::Sampler::WrapMode(device, tsc.wrap_u, tsc.mag_filter),
        .addressModeV = MaxwellToVK::Sampler::WrapMode(device, tsc.wrap_v, tsc.mag_filter),
        .addressModeW = MaxwellToVK::Sampler::WrapMode(device, tsc.wrap_p, tsc.mag_filter),
        .mipLodBias = tsc.LodBias(),
        .anisotropyEnable = static_cast<VkBool32>(tsc.MaxAnisotropy() > 1.0f ? VK_TRUE : VK_FALSE),
        .maxAnisotropy = tsc.MaxAnisotropy(),
        .compareEnable = tsc.depth_compare_enabled,
        .compareOp = MaxwellToVK::Sampler::DepthCompareFunction(tsc.depth_compare_func),
        .minLod = tsc.mipmap_filter == TextureMipmapFilter::None ? 0.0f : tsc.MinLod(),
        .maxLod = tsc.mipmap_filter == TextureMipmapFilter::None ? 0.25f : tsc.MaxLod(),
        .borderColor =
            arbitrary_borders ? VK_BORDER_COLOR_INT_CUSTOM_EXT : ConvertBorderColor(color),
        .unnormalizedCoordinates = VK_FALSE,
    });
}

Framebuffer::Framebuffer(TextureCacheRuntime& runtime, std::span<ImageView*, NUM_RT> color_buffers,
                         ImageView* depth_buffer, std::array<u8, NUM_RT> draw_buffers,
                         VideoCommon::Extent2D size) {
    std::vector<VkAttachmentDescription> descriptions;
    std::vector<VkImageView> attachments;
    RenderPassKey renderpass_key{};
    u32 num_layers = 0;

    for (size_t index = 0; index < NUM_RT; ++index) {
        if (const ImageView* const image_view = color_buffers[index]; image_view) {
            descriptions.push_back(AttachmentDescription(runtime.device, image_view));
            attachments.push_back(image_view->RenderTarget());
            renderpass_key.color_formats[index] = image_view->format;
            num_layers = std::max(num_layers, image_view->range.extent.layers);
        } else {
            renderpass_key.color_formats[index] = PixelFormat::Invalid;
        }
    }

    const size_t num_colors = attachments.size();
    const VkAttachmentReference* depth_attachment =
        depth_buffer ? &ATTACHMENT_REFERENCES[num_colors] : nullptr;
    if (depth_buffer) {
        descriptions.push_back(AttachmentDescription(runtime.device, depth_buffer));
        attachments.push_back(depth_buffer->RenderTarget());
        renderpass_key.depth_format = depth_buffer->format;
        num_layers = std::max(num_layers, depth_buffer->range.extent.layers);
    } else {
        renderpass_key.depth_format = PixelFormat::Invalid;
    }
    renderpass_key.samples = MsaaMode::Msaa1x1; // TODO

    const auto& device = runtime.device.GetLogical();
    const auto [cache_pair, is_new] = runtime.renderpass_cache.try_emplace(renderpass_key);
    if (is_new) {
        const VkSubpassDescription subpass{
            .flags = 0,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = static_cast<u32>(num_colors),
            .pColorAttachments = num_colors != 0 ? ATTACHMENT_REFERENCES.data() : nullptr,
            .pResolveAttachments = nullptr,
            .pDepthStencilAttachment = depth_attachment,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = nullptr,
        };
        cache_pair->second = device.CreateRenderPass(VkRenderPassCreateInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .attachmentCount = static_cast<u32>(descriptions.size()),
            .pAttachments = descriptions.data(),
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 0,
            .pDependencies = nullptr,
        });
    }

    renderpass = *cache_pair->second;
    render_area = {
        .width = size.width,
        .height = size.height,
    };
    num_color_buffers = static_cast<u32>(num_colors);
    framebuffer = device.CreateFramebuffer(VkFramebufferCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderPass = renderpass,
        .attachmentCount = static_cast<u32>(attachments.size()),
        .pAttachments = attachments.data(),
        .width = size.width,
        .height = size.height,
        .layers = num_layers,
    });
}

} // namespace Vulkan
