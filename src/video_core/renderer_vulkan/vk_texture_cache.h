// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <compare>
#include <span>

#include "video_core/renderer_vulkan/vk_memory_manager.h"
#include "video_core/renderer_vulkan/wrapper.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

using Tegra::Texture::MsaaMode;
using VideoCommon::ImageId;
using VideoCommon::NUM_RT;
using VideoCommon::RenderTargets;
using VideoCore::Surface::PixelFormat;

class VKDevice;
class VKScheduler;
class VKStagingBufferPool;

class BlitImageHelper;
class Image;
class ImageView;
class Framebuffer;

struct RenderPassKey {
    constexpr auto operator<=>(const RenderPassKey&) const noexcept = default;

    std::array<PixelFormat, NUM_RT> color_formats;
    PixelFormat depth_format;
    MsaaMode samples;
};

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::RenderPassKey> {
    [[nodiscard]] constexpr size_t operator()(const Vulkan::RenderPassKey& key) const noexcept {
        size_t hash = static_cast<size_t>(key.depth_format) << 48;
        hash ^= static_cast<size_t>(key.samples) << 52;
        for (size_t i = 0; i < key.color_formats.size(); ++i) {
            hash ^= static_cast<size_t>(key.color_formats[i]) << (i * 6);
        }
        return hash;
    }
};
} // namespace std

namespace Vulkan {

struct ImageBufferMap {
    [[nodiscard]] VkBuffer Handle() const noexcept {
        return handle;
    }

    [[nodiscard]] std::span<u8> Span() const noexcept {
        return map.Span();
    }

    VkBuffer handle;
    MemoryMap map;
};

struct TextureCacheRuntime {
    const VKDevice& device;
    VKScheduler& scheduler;
    VKMemoryManager& memory_manager;
    VKStagingBufferPool& staging_buffer_pool;
    BlitImageHelper& blit_image_helper;
    std::unordered_map<RenderPassKey, vk::RenderPass> renderpass_cache;

    [[nodiscard]] ImageBufferMap MapUploadBuffer(size_t size);

    [[nodiscard]] ImageBufferMap MapDownloadBuffer(size_t size) {
        // TODO: Have a special function for this
        return MapUploadBuffer(size);
    }

    void BlitImage(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& src,
                   const Tegra::Engines::Fermi2D::Config& copy);

    void CopyImage(Image& dst, Image& src, std::span<const VideoCommon::ImageCopy> copies);

    void ConvertImage(Framebuffer* dst, ImageView& dst_view, ImageView& src_view);

    [[nodiscard]] bool CanAccelerateImageUpload(Image&) const noexcept {
        return false;
    }

    void AccelerateImageUpload(Image&, const ImageBufferMap&, size_t,
                               std::span<const VideoCommon::SwizzleParameters>) {
        UNREACHABLE();
    }

    void InsertUploadMemoryBarrier();
};

class Image : public VideoCommon::ImageBase {
public:
    explicit Image(TextureCacheRuntime&, const VideoCommon::ImageInfo& info, GPUVAddr gpu_addr,
                   VAddr cpu_addr);

    void UploadMemory(const ImageBufferMap& map, size_t buffer_offset,
                      std::span<const VideoCommon::BufferImageCopy> copies);

    void UploadMemory(const ImageBufferMap& map, size_t buffer_offset,
                      std::span<const VideoCommon::BufferCopy> copies);

    void DownloadMemory(const ImageBufferMap& map, size_t buffer_offset,
                        std::span<const VideoCommon::BufferImageCopy> copies);

    [[nodiscard]] VkImage Handle() const noexcept {
        return *image;
    }

    [[nodiscard]] VkBuffer Buffer() const noexcept {
        return *buffer;
    }

    [[nodiscard]] VkImageCreateFlags AspectMask() const noexcept {
        return aspect_mask;
    }

private:
    VKScheduler* scheduler;
    vk::Image image;
    vk::Buffer buffer;
    VKMemoryCommit commit;
    VkImageAspectFlags aspect_mask = 0;
    bool initialized = false;
};

class ImageView : public VideoCommon::ImageViewBase {
public:
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::ImageViewInfo&, ImageId, Image&);
    explicit ImageView(TextureCacheRuntime&, const VideoCommon::NullImageParams&);

    [[nodiscard]] VkImageView Handle(VideoCommon::ImageViewType type) const noexcept {
        return *image_views[static_cast<size_t>(type)];
    }

    [[nodiscard]] VkBufferView BufferView() const noexcept {
        return *buffer_view;
    }

    [[nodiscard]] VkImage ImageHandle() const noexcept {
        return image_handle;
    }

    [[nodiscard]] VkImageView RenderTarget() const noexcept {
        return render_target;
    }

private:
    std::array<vk::ImageView, VideoCommon::NUM_IMAGE_VIEW_TYPES> image_views;
    vk::BufferView buffer_view;
    VkImage image_handle = VK_NULL_HANDLE;
    VkImageView render_target = VK_NULL_HANDLE;
};

class ImageAlloc : public VideoCommon::ImageAllocBase {};

class Sampler {
public:
    explicit Sampler(TextureCacheRuntime&, const Tegra::Texture::TSCEntry&);

    [[nodiscard]] VkSampler Handle() const noexcept {
        return *sampler;
    }

private:
    vk::Sampler sampler;
};

class Framebuffer {
public:
    explicit Framebuffer(TextureCacheRuntime&, const VideoCommon::SlotVector<Image>& slot_images,
                         std::span<ImageView*, NUM_RT> color_buffers, ImageView* depth_buffer,
                         std::array<u8, NUM_RT> draw_buffers, VideoCommon::Extent2D size);

    [[nodiscard]] VkFramebuffer Handle() const noexcept {
        return *framebuffer;
    }

    [[nodiscard]] VkRenderPass RenderPass() const noexcept {
        return renderpass;
    }

    [[nodiscard]] VkExtent2D RenderArea() const noexcept {
        return render_area;
    }

    [[nodiscard]] u32 NumColorBuffers() const noexcept {
        return num_color_buffers;
    }

    [[nodiscard]] u32 NumImages() const noexcept {
        return num_images;
    }

    [[nodiscard]] const std::array<VkImage, 9>& Images() const noexcept {
        return images;
    }

    [[nodiscard]] const std::array<VkImageSubresourceRange, 9>& ImageRanges() const noexcept {
        return image_ranges;
    }

private:
    vk::Framebuffer framebuffer;
    VkRenderPass renderpass{};
    VkExtent2D render_area{};
    u32 num_color_buffers = 0;
    u32 num_images = 0;
    std::array<VkImage, 9> images{};
    std::array<VkImageSubresourceRange, 9> image_ranges{};
};

struct TextureCacheParams {
    static constexpr bool ENABLE_VALIDATION = true;
    static constexpr bool FRAMEBUFFER_BLITS = false;

    using Runtime = Vulkan::TextureCacheRuntime;
    using Image = Vulkan::Image;
    using ImageAlloc = Vulkan::ImageAlloc;
    using ImageView = Vulkan::ImageView;
    using Sampler = Vulkan::Sampler;
    using Framebuffer = Vulkan::Framebuffer;
};

using TextureCache = VideoCommon::TextureCache<TextureCacheParams>;

} // namespace Vulkan
