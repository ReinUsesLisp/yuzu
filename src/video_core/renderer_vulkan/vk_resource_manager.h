// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <vector>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/declarations.h"

namespace Vulkan {

class VKDevice;
class VKFence;
class VKResourceManager;

class CommandBufferPool;

/// Interface for a vulkan resource
class VKResource {
public:
    explicit VKResource();
    virtual ~VKResource() = 0;

    /**
     * Signals the object that an owning fence has been signaled.
     * @param signaling_fence Fence that signals its usage end.
     */
    virtual void OnFenceRemoval(VKFence* signaling_fence) = 0;
};

/**
 * Fences take ownership of objects, protecting them from GPU-side or driver-side race conditions.
 * They must be commited from the resouce manager. Their use flow is: commit the fence from the
 * manager, protect resources with it and use them, send the fence to a execution queue and Wait for
 * it if needed and then call Release. Used resources will automatically be signaled by the manager
 * when they are free to be reused.
 * @brief Protects are resource for concurrent usage and signals its release
 */
class VKFence {
    friend class VKResourceManager;

public:
    explicit VKFence(const VKDevice& device, UniqueFence handle);
    ~VKFence();

    /**
     * Waits for the fence to be signaled.
     * @warning You must have ownership of the fence and it has to be previously sent to a queue to
     * call this function.
     */
    void Wait();

    /**
     * Releases ownership of the fence. Pass after it has been sent to an execution queue.
     * Unmanaged usage of the fence after the call will result in undefined behavior because it may
     * be being used for something else.
     */
    void Release();

    /**
     * Protects a resource with this fence.
     * @param resource Resource to protect.
     */
    void Protect(VKResource* resource);

    /**
     * Removes protection for a resource.
     * @param resource Resource to unprotect.
     */
    void Unprotect(VKResource* resource);

    /// Retreives the fence.
    operator vk::Fence() const {
        return *handle;
    }

private:
    /// Take ownership of the fence.
    void Commit();

    /**
     * Updates the fence status.
     * @warning Waiting for the owner might soft lock the execution.
     * @param gpu_wait Wait for the fence to be signaled by the driver.
     * @param owner_wait Wait for the owner to signal its freedom.
     * @returns True if the fence is free. Waiting for gpu and owner will always return true.
     */
    bool Tick(bool gpu_wait, bool owner_wait);

    const VKDevice& device;                       ///< Device handler
    UniqueFence handle;                           ///< Vulkan fence
    std::vector<VKResource*> protected_resources; ///< List of resources protected by this fence
    bool is_owned = false; ///< The fence has been commited but not released yet.
    bool is_used = false;  ///< The fence has been commited but it has not been checked to be free.
};

/**
 * A fence watch is used to keep track of the usage a fence and protect a resource or set of
 * resources.
 */
class VKFenceWatch final : public VKResource {
public:
    explicit VKFenceWatch();
    ~VKFenceWatch();

    /// Waits for the fence to be released.
    void Wait();

    /**
     * Waits for a previous fence and watches a new one.
     * @param new_fence New fence to wait to.
     */
    void Watch(VKFence& new_fence);

    /**
     * Checks if it's currently being watched and starts watching it if it's available.
     * @returns True if a watch has started, false if it's being watched.
     */
    bool TryWatch(VKFence& new_fence);

    void OnFenceRemoval(VKFence* signaling_fence) override;

private:
    VKFence* fence{}; ///< Fence watching this resource. nullptr when the watch is free.
};

/**
 * Handles a pool of resources protected by fences. Handles resources overflow allocating more
 * resources.
 */
class VKFencedPool {
public:
    explicit VKFencedPool(std::size_t grow_step);
    virtual ~VKFencedPool();

protected:
    /**
     * Commits a free resource and protects it with a fence. It may allocate new resources.
     * @param fence Fence that protects the commited resource.
     * @returns Index of the resource commited.
     */
    std::size_t CommitResource(VKFence& fence);

    /// Called when a chunk of resources have to be allocated.
    virtual void Allocate(std::size_t begin, std::size_t end) = 0;

private:
    /// Manages pool overflow allocating new resources.
    std::size_t ManageOverflow();

    /// Allocates a new page of resources.
    void Grow();

    std::size_t grow_step{};       ///< Number of new resources created after an overflow
    std::size_t free_iterator = 0; ///< Hint to where the next free resources is likely to be found
    std::vector<std::unique_ptr<VKFenceWatch>> watches; ///< Set of watched resources
};

/**
 * The resource manager handles all resources that can be protected with a fence avoiding
 * driver-side or GPU-side race conditions. Use flow is documented in VKFence.
 */
class VKResourceManager final {
public:
    explicit VKResourceManager(const VKDevice& device);
    ~VKResourceManager();

    /// Commits a fence. It has to be sent to a queue and released.
    VKFence& CommitFence();

    /// Commits an unused command buffer and protects it with a fence.
    vk::CommandBuffer CommitCommandBuffer(VKFence& fence);

private:
    /// Allocates new fences.
    void GrowFences(std::size_t new_fences_count);

    const VKDevice& device;          ///< Device handler.
    std::size_t fences_iterator = 0; ///< Index where a free fence is likely to be found.
    std::vector<std::unique_ptr<VKFence>> fences;           ///< Pool of fences.
    std::unique_ptr<CommandBufferPool> command_buffer_pool; ///< Pool of command buffers.
};

} // namespace Vulkan
