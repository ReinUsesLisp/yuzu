// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include "video_core/engines/fermi_2d.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/wrapper.h"

namespace Vulkan {

using Tegra::Engines::Fermi2D;
using Tegra::Texture::SwizzleSource;
using Tegra::Texture::TextureMipmapFilter;
using VideoCommon::BufferImageCopy;
using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCore::Surface::IsPixelFormatASTC;

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

[[nodiscard]] VkImageType ConvertImageType(const ImageType type) {
    switch (type) {
    case ImageType::e1D:
        return VK_IMAGE_TYPE_1D;
    case ImageType::e2D:
    case ImageType::Linear:
    case ImageType::Rect:
        return VK_IMAGE_TYPE_2D;
    case ImageType::e3D:
        return VK_IMAGE_TYPE_3D;
    case ImageType::Buffer:
        break;
    }
    UNREACHABLE_MSG("Invalid image type={}", static_cast<int>(type));
    return {};
}

[[nodiscard]] VkImageCreateInfo MakeImageCreateInfo(const VKDevice& device, const ImageInfo& info) {
    const auto format_info = MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, info.format);
    VkImageCreateFlags flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    if (info.type == ImageType::e2D && info.resources.layers >= 6 &&
        info.size.width == info.size.height) {
        flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }
    if (info.type == ImageType::e3D) {
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
        .imageType = ConvertImageType(info.type),
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

[[nodiscard]] vk::Image MakeImage(const VKDevice& device, const ImageInfo& info) {
    if (info.type == ImageType::Buffer) {
        return vk::Image{};
    }
    return device.GetLogical().CreateImage(MakeImageCreateInfo(device, info));
}

[[nodiscard]] vk::Buffer MakeBuffer(const VKDevice& device, const ImageInfo& info) {
    if (info.type != ImageType::Buffer) {
        return vk::Buffer{};
    }
    const size_t bytes_per_block = VideoCore::Surface::BytesPerBlock(info.format);
    return device.GetLogical().CreateBuffer(VkBufferCreateInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = info.size.width * bytes_per_block,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
                 VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    });
}

[[nodiscard]] VkImageAspectFlags ImageAspectMask(PixelFormat format) {
    switch (VideoCore::Surface::GetFormatType(format)) {
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

[[nodiscard]] VkImageAspectFlags ImageViewAspectMask(const VideoCommon::ImageViewInfo& info) {
    const bool is_first = info.Swizzle()[0] == SwizzleSource::R;
    switch (info.format) {
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::D32_FLOAT_S8_UINT:
        return is_first ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT;
    case PixelFormat::S8_UINT_D24_UNORM:
        return is_first ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    case PixelFormat::D16_UNORM:
    case PixelFormat::D32_FLOAT:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
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

[[nodiscard]] VkComponentSwizzle ComponentSwizzle(SwizzleSource swizzle) {
    switch (swizzle) {
    case SwizzleSource::Zero:
        return VK_COMPONENT_SWIZZLE_ZERO;
    case SwizzleSource::R:
        return VK_COMPONENT_SWIZZLE_R;
    case SwizzleSource::G:
        return VK_COMPONENT_SWIZZLE_G;
    case SwizzleSource::B:
        return VK_COMPONENT_SWIZZLE_B;
    case SwizzleSource::A:
        return VK_COMPONENT_SWIZZLE_A;
    case SwizzleSource::OneFloat:
    case SwizzleSource::OneInt:
        return VK_COMPONENT_SWIZZLE_ONE;
    }
    UNREACHABLE_MSG("Invalid swizzle={}", static_cast<int>(swizzle));
    return VK_COMPONENT_SWIZZLE_ZERO;
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

[[nodiscard]] std::vector<VkBufferCopy> TransformBufferCopies(
    std::span<const VideoCommon::BufferCopy> copies, size_t buffer_offset) {
    std::vector<VkBufferCopy> result(copies.size());
    std::ranges::transform(
        copies, result.begin(), [buffer_offset](const VideoCommon::BufferCopy& copy) {
            return VkBufferCopy{
                .srcOffset = static_cast<VkDeviceSize>(copy.src_offset + buffer_offset),
                .dstOffset = static_cast<VkDeviceSize>(copy.dst_offset),
                .size = static_cast<VkDeviceSize>(copy.size),
            };
        });
    return result;
}

[[nodiscard]] std::vector<VkBufferImageCopy> TransformBufferImageCopies(
    std::span<const BufferImageCopy> copies, size_t buffer_offset, VkImageAspectFlags aspect_mask) {
    struct Maker {
        VkBufferImageCopy operator()(const BufferImageCopy& copy) const {
            return VkBufferImageCopy{
                .bufferOffset = copy.buffer_offset + buffer_offset,
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
        size_t buffer_offset;
        VkImageAspectFlags aspect_mask;
    };
    if (aspect_mask == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
        std::vector<VkBufferImageCopy> result(copies.size() * 2);
        std::ranges::transform(copies, result.begin(),
                               Maker{buffer_offset, VK_IMAGE_ASPECT_DEPTH_BIT});
        std::ranges::transform(copies, result.begin() + copies.size(),
                               Maker{buffer_offset, VK_IMAGE_ASPECT_STENCIL_BIT});
        return result;
    } else {
        std::vector<VkBufferImageCopy> result(copies.size());
        std::ranges::transform(copies, result.begin(), Maker{buffer_offset, aspect_mask});
        return result;
    }
}

[[nodiscard]] constexpr VkImageSubresourceRange MakeSubresourceRange(
    VkImageAspectFlags aspect_mask, const VideoCommon::SubresourceRange& range) {
    return VkImageSubresourceRange{
        .aspectMask = aspect_mask,
        .baseMipLevel = range.base.mipmap,
        .levelCount = range.extent.mipmaps,
        .baseArrayLayer = range.base.layer,
        .layerCount = range.extent.layers,
    };
}

[[nodiscard]] VkImageSubresourceRange MakeSubresourceRange(const ImageView* image_view) {
    return MakeSubresourceRange(ImageAspectMask(image_view->format), image_view->range);
}

[[nodiscard]] VkImageSubresourceLayers MakeSubresourceLayers(const ImageView* image_view) {
    return VkImageSubresourceLayers{
        .aspectMask = ImageAspectMask(image_view->format),
        .mipLevel = image_view->range.base.mipmap,
        .baseArrayLayer = image_view->range.base.layer,
        .layerCount = image_view->range.extent.layers,
    };
}

[[nodiscard]] constexpr SwizzleSource SwapRedGreen(SwizzleSource value) {
    switch (value) {
    case SwizzleSource::R:
        return SwizzleSource::G;
    case SwizzleSource::G:
        return SwizzleSource::R;
    default:
        return value;
    }
}

void CopyBufferToImage(vk::CommandBuffer cmdbuf, VkBuffer src_buffer, VkImage image,
                       VkImageAspectFlags aspect_mask, bool is_initialized,
                       std::span<const VkBufferImageCopy> copies) {
    static constexpr VkAccessFlags ACCESS_FLAGS =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkImageMemoryBarrier read_barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = ACCESS_FLAGS,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = is_initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
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
    const VkImageMemoryBarrier write_barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = ACCESS_FLAGS,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
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
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                           read_barrier);
    cmdbuf.CopyBufferToImage(src_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copies);
    // TODO: Move this to another API
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                           write_barrier);
}

[[nodiscard]] VkImageBlit MakeImageBlit(const Fermi2D::Config& copy,
                                        const VkImageSubresourceLayers& dst_layers,
                                        const VkImageSubresourceLayers& src_layers) {
    // Using an aggregate constructor here generates an internal compiler error on MSVC
    VkImageBlit result;
    result.srcSubresource = src_layers;
    result.srcOffsets[0] = {
        .x = copy.src_x0,
        .y = copy.src_y0,
        .z = 0,
    };
    result.srcOffsets[1] = {
        .x = copy.src_x1,
        .y = copy.src_y1,
        .z = 1,
    };
    result.dstSubresource = dst_layers;
    result.dstOffsets[0] = {
        .x = copy.dst_x0,
        .y = copy.dst_y0,
        .z = 0,
    };
    result.dstOffsets[1] = {
        .x = copy.dst_x1,
        .y = copy.dst_y1,
        .z = 1,
    };
    return result;
}

} // Anonymous namespace

ImageBufferMap TextureCacheRuntime::MapUploadBuffer(size_t size) {
    const auto& buffer = staging_buffer_pool.GetUnusedBuffer(size, true);
    return ImageBufferMap{
        .handle = *buffer.handle,
        .map = buffer.commit->Map(size),
    };
}

void TextureCacheRuntime::BlitImage(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& src,
                                    const Fermi2D::Config& copy) {
    const VkImageAspectFlags aspect_mask = ImageAspectMask(src.format);
    ASSERT(aspect_mask == ImageAspectMask(dst.format));
    if (aspect_mask == VK_IMAGE_ASPECT_COLOR_BIT) {
        blit_image_helper.BlitColor(dst_framebuffer, src, copy);
        return;
    }
    ASSERT(src.format == dst.format);
    ASSERT(copy.operation == Fermi2D::Operation::SrcCopy);
    const VkImage dst_image = dst.ImageHandle();
    const VkImage src_image = src.ImageHandle();
    const VkImageSubresourceLayers dst_layers = MakeSubresourceLayers(&dst);
    const VkImageSubresourceLayers src_layers = MakeSubresourceLayers(&src);
    scheduler.Record(
        [copy, dst_image, src_image, dst_layers, src_layers](vk::CommandBuffer cmdbuf) {
            // TODO: Barriers
            const bool is_linear = copy.filter == Fermi2D::Filter::Bilinear;
            const VkFilter filter = is_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
            cmdbuf.BlitImage(src_image, VK_IMAGE_LAYOUT_GENERAL, dst_image, VK_IMAGE_LAYOUT_GENERAL,
                             MakeImageBlit(copy, dst_layers, src_layers), filter);
        });
}

void TextureCacheRuntime::ConvertImage(Framebuffer* dst, ImageView& dst_view, ImageView& src_view) {
    switch (dst_view.format) {
    case PixelFormat::R32_FLOAT:
        switch (src_view.format) {
        case PixelFormat::D32_FLOAT:
            return blit_image_helper.ConvertD32ToR32(dst, src_view);
        default:
            break;
        }
        break;
    case PixelFormat::D32_FLOAT:
        switch (src_view.format) {
        case PixelFormat::R32_FLOAT:
            return blit_image_helper.ConvertR32ToD32(dst, src_view);
        default:
            break;
        }
        break;
    default:
        break;
    }
    UNIMPLEMENTED_MSG("Unimplemented format copy from {} to {}", src_view.format, dst_view.format);
}

void TextureCacheRuntime::CopyImage(Image& dst, Image& src,
                                    std::span<const VideoCommon::ImageCopy> copies) {
    std::vector<VkImageCopy> vk_copies(copies.size());
    const VkImageAspectFlags aspect_mask = dst.AspectMask();
    ASSERT(aspect_mask == src.AspectMask());

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

Image::Image(TextureCacheRuntime& runtime, const ImageInfo& info, GPUVAddr gpu_addr, VAddr cpu_addr)
    : VideoCommon::ImageBase(info, gpu_addr, cpu_addr), scheduler{&runtime.scheduler},
      image(MakeImage(runtime.device, info)), buffer(MakeBuffer(runtime.device, info)),
      aspect_mask(ImageAspectMask(info.format)) {
    if (image) {
        commit = runtime.memory_manager.Commit(image, false);
    } else {
        commit = runtime.memory_manager.Commit(buffer, false);
    }

    if (image) {
        auto vkSetDebugUtilsObjectNameEXT =
            (PFN_vkSetDebugUtilsObjectNameEXT)runtime.device.GetDispatchLoader()
                .vkGetDeviceProcAddr(*runtime.device.GetLogical(), "vkSetDebugUtilsObjectNameEXT");
        auto c = fmt::format("Image 0x{:x} {}", static_cast<u64>(gpu_addr), info.layer_stride);
        const VkDebugUtilsObjectNameInfoEXT tag{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = VK_OBJECT_TYPE_IMAGE,
            .objectHandle = reinterpret_cast<u64>(*image),
            .pObjectName = c.c_str(),
        };
        vkSetDebugUtilsObjectNameEXT(*runtime.device.GetLogical(), &tag);
    }
    if (IsPixelFormatASTC(info.format) && !runtime.device.IsOptimalAstcSupported()) {
        flags |= VideoCommon::ImageFlagBits::Converted;
    }
}

void Image::UploadMemory(const ImageBufferMap& map, size_t buffer_offset,
                         std::span<const BufferImageCopy> copies) {
    // TODO: Move this to another API
    scheduler->RequestOutsideRenderPassOperationContext();
    std::vector vk_copies = TransformBufferImageCopies(copies, buffer_offset, aspect_mask);
    const VkBuffer src_buffer = map.handle;
    const VkImage vk_image = *image;
    const VkImageAspectFlags vk_aspect_mask = aspect_mask;
    const bool is_initialized = std::exchange(initialized, true);
    scheduler->Record([src_buffer, vk_image, vk_aspect_mask, is_initialized,
                       vk_copies](vk::CommandBuffer cmdbuf) {
        CopyBufferToImage(cmdbuf, src_buffer, vk_image, vk_aspect_mask, is_initialized, vk_copies);
    });
}

void Image::UploadMemory(const ImageBufferMap& map, size_t buffer_offset,
                         std::span<const VideoCommon::BufferCopy> copies) {
    // TODO: Move this to another API
    scheduler->RequestOutsideRenderPassOperationContext();
    std::vector vk_copies = TransformBufferCopies(copies, buffer_offset);
    const VkBuffer src_buffer = map.handle;
    const VkBuffer dst_buffer = *buffer;
    scheduler->Record([src_buffer, dst_buffer, vk_copies](vk::CommandBuffer cmdbuf) {
        // TODO: Barriers
        cmdbuf.CopyBuffer(src_buffer, dst_buffer, vk_copies);
    });
}

void Image::DownloadMemory(const ImageBufferMap& map, size_t buffer_offset,
                           std::span<const BufferImageCopy> copies) {
    std::vector vk_copies = TransformBufferImageCopies(copies, buffer_offset, aspect_mask);
    scheduler->Record([buffer = map.handle, image = *image, aspect_mask = aspect_mask,
                       vk_copies](vk::CommandBuffer cmdbuf) {
        // TODO: Barriers
        cmdbuf.CopyImageToBuffer(image, VK_IMAGE_LAYOUT_GENERAL, buffer, vk_copies);
    });
    scheduler->Finish();
}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageViewInfo& info,
                     ImageId image_id, Image& image)
    : VideoCommon::ImageViewBase{info, image.info, image_id}, image_handle{image.Handle()} {
    const VkFormat format =
        MaxwellToVK::SurfaceFormat(runtime.device, FormatType::Optimal, info.format).format;
    const std::array swizzle = info.Swizzle();
    const VkImageViewCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = image.Handle(),
        .viewType = VkImageViewType{},
        .format = format,
        .components =
            {
                .r = ComponentSwizzle(swizzle[0]),
                .g = ComponentSwizzle(swizzle[1]),
                .b = ComponentSwizzle(swizzle[2]),
                .a = ComponentSwizzle(swizzle[3]),
            },
        .subresourceRange = MakeSubresourceRange(ImageViewAspectMask(info), info.range),
    };
    const vk::Device& device = runtime.device.GetLogical();
    const auto create = [this, &device, &create_info](VideoCommon::ImageViewType type,
                                                      std::optional<u32> num_layers) {
        VkImageViewCreateInfo ci{create_info};
        ci.viewType = ImageViewType(type);
        if (num_layers) {
            ci.subresourceRange.layerCount = *num_layers;
        }
        image_views[static_cast<size_t>(type)] = device.CreateImageView(ci);
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
        buffer_view = device.CreateBufferView(VkBufferViewCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .buffer = image.Buffer(),
            .format = format,
            .offset = 0, // TODO: Redesign buffer cache to support this
            .range = image.guest_size_bytes,
        });
        break;
    }
}

ImageView::ImageView(TextureCacheRuntime&, const VideoCommon::NullImageParams& params)
    : VideoCommon::ImageViewBase{params} {}

Sampler::Sampler(TextureCacheRuntime& runtime, const Tegra::Texture::TSCEntry& tsc) {
    const auto& device = runtime.device;
    const bool arbitrary_borders = runtime.device.IsExtCustomBorderColorSupported();
    const std::array<float, 4> color = tsc.BorderColor();
    // C++20 bit_cast
    VkClearColorValue border_color;
    std::memcpy(&border_color, &color, sizeof(color));
    const VkSamplerCustomBorderColorCreateInfoEXT border{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
        .pNext = nullptr,
        .customBorderColor = border_color,
        .format = VK_FORMAT_UNDEFINED,
    };
    const VkSamplerReductionModeCreateInfoEXT reduction_ci{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT,
        .pNext = arbitrary_borders ? &border : nullptr,
        .reductionMode = MaxwellToVK::SamplerReduction(tsc.reduction_filter),
    };
    // Some games have samplers with garbage. Sanitize them here.
    const float max_anisotropy = std::clamp(tsc.MaxAnisotropy(), 1.0f, 16.0f);
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
        .anisotropyEnable = static_cast<VkBool32>(max_anisotropy > 1.0f ? VK_TRUE : VK_FALSE),
        .maxAnisotropy = max_anisotropy,
        .compareEnable = tsc.depth_compare_enabled,
        .compareOp = MaxwellToVK::Sampler::DepthCompareFunction(tsc.depth_compare_func),
        .minLod = tsc.mipmap_filter == TextureMipmapFilter::None ? 0.0f : tsc.MinLod(),
        .maxLod = tsc.mipmap_filter == TextureMipmapFilter::None ? 0.25f : tsc.MaxLod(),
        .borderColor =
            arbitrary_borders ? VK_BORDER_COLOR_INT_CUSTOM_EXT : ConvertBorderColor(color),
        .unnormalizedCoordinates = VK_FALSE,
    });
}

Framebuffer::Framebuffer(TextureCacheRuntime& runtime,
                         const VideoCommon::SlotVector<Image>& slot_images,
                         std::span<ImageView*, NUM_RT> color_buffers, ImageView* depth_buffer,
                         std::array<u8, NUM_RT> draw_buffers, VideoCommon::Extent2D size) {
    std::vector<VkAttachmentDescription> descriptions;
    std::vector<VkImageView> attachments;
    RenderPassKey renderpass_key{};
    u32 num_layers = 0;

    for (size_t index = 0; index < NUM_RT; ++index) {
        const ImageView* const image_view = color_buffers[index];
        if (!image_view) {
            renderpass_key.color_formats[index] = PixelFormat::Invalid;
            continue;
        }
        descriptions.push_back(AttachmentDescription(runtime.device, image_view));
        attachments.push_back(image_view->RenderTarget());
        renderpass_key.color_formats[index] = image_view->format;
        num_layers = std::max(num_layers, image_view->range.extent.layers);
        images[num_images] = slot_images[image_view->image_id].Handle();
        image_ranges[num_images] = MakeSubresourceRange(image_view);
        ++num_images;
    }

    const size_t num_colors = attachments.size();
    const VkAttachmentReference* depth_attachment =
        depth_buffer ? &ATTACHMENT_REFERENCES[num_colors] : nullptr;
    if (depth_buffer) {
        descriptions.push_back(AttachmentDescription(runtime.device, depth_buffer));
        attachments.push_back(depth_buffer->RenderTarget());
        renderpass_key.depth_format = depth_buffer->format;
        num_layers = std::max(num_layers, depth_buffer->range.extent.layers);
        images[num_images] = slot_images[depth_buffer->image_id].Handle();
        image_ranges[num_images] = MakeSubresourceRange(depth_buffer);
        ++num_images;
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
