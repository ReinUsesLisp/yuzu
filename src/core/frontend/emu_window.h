// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>
#include "common/common_types.h"
#include "common/dynamic_library.h"
#include "core/frontend/framebuffer_layout.h"

namespace Core::Frontend {

/// Information for the Graphics Backends signifying what type of screen pointer is in
/// WindowInformation
enum class WindowSystemType {
    Uninitialized,
    Windows,
    X11,
    Wayland,
};

/// List of enum
/// This isn't meant to be an exhaustive list of renderer backends, rather this represents the
/// different API's that the backends can use for rendering
enum class APIType {
    Nothing = -1,
    OpenGL = 0,
    Vulkan = 1,
};

/// Information for the backends that the frontend should hold a reference to. This information is
/// "static" and can be persisted between emulator runs (hence why it is part of the EmuWindow and
/// not the RendererBackend)
struct BackendInfo {
    /// Name of the renderer backend (usually the same as the API type)
    std::string name;
    /// Which graphics API this backend uses
    APIType api_type;
    /// Reference to the shared library that powers this backend.
    Common::DynamicLibrary dl;
    /// List of display adapaters that this backend supports rendering to. Empty if this isn't
    /// modifiable
    std::vector<std::string> adapters;
    // TODO: add supported feature detection here (extensions)
};

/**
 * Represents a graphics context that can be used for background computation or drawing. If the
 * graphics backend doesn't require the context, then the implementation of these methods can be
 * stubs
 */
class GraphicsContext {
public:
    virtual ~GraphicsContext();

    /// Makes the graphics context current for the caller thread
    virtual void MakeCurrent() = 0;

    /// Releases (dunno if this is the "right" word) the context from the caller thread
    virtual void DoneCurrent() = 0;
};

/**
 * Abstraction class used to provide an interface between emulation code and the frontend
 * (e.g. SDL, QGLWidget, GLFW, etc...).
 *
 * Design notes on the interaction between EmuWindow and the emulation core:
 * - Generally, decisions on anything visible to the user should be left up to the GUI.
 *   For example, the emulation core should not try to dictate some window title or size.
 *   This stuff is not the core's business and only causes problems with regards to thread-safety
 *   anyway.
 * - Under certain circumstances, it may be desirable for the core to politely request the GUI
 *   to set e.g. a minimum window size. However, the GUI should always be free to ignore any
 *   such hints.
 * - EmuWindow may expose some of its state as read-only to the emulation core, however care
 *   should be taken to make sure the provided information is self-consistent. This requires
 *   some sort of synchronization (most of this is still a TODO).
 * - DO NOT TREAT THIS CLASS AS A GUI TOOLKIT ABSTRACTION LAYER. That's not what it is. Please
 *   re-read the upper points again and think about it if you don't see this.
 */
class EmuWindow : public GraphicsContext {
public:
    /// Data structure to store emuwindow configuration
    struct WindowSystemInfo {
        WindowSystemInfo() = default;
        WindowSystemInfo(WindowSystemType type_, void* display_connection_, void* render_surface_)
            : type(type_), display_connection(display_connection_),
              render_surface(render_surface_) {}

        // Window system type. Determines which GL context or Vulkan WSI is used.
        WindowSystemType type = WindowSystemType::Uninitialized;

        // Connection to a display server. This is used on X11 and Wayland platforms.
        void* display_connection = nullptr;

        // Render surface. This is a pointer to the native window handle, which depends
        // on the platform. e.g. HWND for Windows, Window for X11. If the surface is
        // set to nullptr, the video backend will run in headless mode.
        void* render_surface = nullptr;

        // Scale of the render surface. For hidpi systems, this will be >1.
        float render_surface_scale = 1.0f;
    };

    /// Polls window events
    virtual void PollEvents() = 0;

    /**
     * Returns a GraphicsContext that the frontend provides that is shared with the emu window. This
     * context can be used from other threads for background graphics computation. If the frontend
     * is using a graphics backend that doesn't need anything specific to run on a different thread,
     * then it can use a stubbed implemenation for GraphicsContext.
     *
     * If the return value is null, then the core should assume that the frontend cannot provide a
     * Shared Context
     */
    virtual std::unique_ptr<GraphicsContext> CreateSharedContext() const {
        return nullptr;
    }

    /// Returns if window is shown (not minimized)
    virtual bool IsShown() const = 0;

    /**
     * Signal that a touch pressed event has occurred (e.g. mouse click pressed)
     * @param framebuffer_x Framebuffer x-coordinate that was pressed
     * @param framebuffer_y Framebuffer y-coordinate that was pressed
     */
    void TouchPressed(unsigned framebuffer_x, unsigned framebuffer_y);

    /// Signal that a touch released event has occurred (e.g. mouse click released)
    void TouchReleased();

    /**
     * Signal that a touch movement event has occurred (e.g. mouse was moved over the emu window)
     * @param framebuffer_x Framebuffer x-coordinate
     * @param framebuffer_y Framebuffer y-coordinate
     */
    void TouchMoved(unsigned framebuffer_x, unsigned framebuffer_y);

    /**
     * Returns system information about the drawing area.
     */
    const WindowSystemInfo& GetWindowInfo() const {
        return window_info;
    }

    /**
     * Gets the framebuffer layout (width, height, and screen regions)
     * @note This method is thread-safe
     */
    const Layout::FramebufferLayout& GetFramebufferLayout() const {
        return framebuffer_layout;
    }

    /**
     * Convenience method to update the current frame layout
     * Read from the current settings to determine which layout to use.
     */
    void UpdateCurrentFramebufferLayout(unsigned width, unsigned height);

    /**
     * Retrieves the current backend info for a renderer backend by api type
     */
    BackendInfo* GetBackendInfo(APIType);

protected:
    explicit EmuWindow(WindowSystemInfo);
    virtual ~EmuWindow();

    /**
     * Update framebuffer layout with the given parameter.
     * @note EmuWindow implementations will usually use this in window resize event handlers.
     */
    void NotifyFramebufferLayoutChanged(const Layout::FramebufferLayout& layout) {
        framebuffer_layout = layout;
    }

    WindowSystemInfo window_info;
    std::vector<BackendInfo> possible_backends;

private:
    Layout::FramebufferLayout framebuffer_layout; ///< Current framebuffer layout

    class TouchState;
    std::shared_ptr<TouchState> touch_state;

    /**
     * Clip the provided coordinates to be inside the touchscreen area.
     */
    std::tuple<unsigned, unsigned> ClipToTouchScreen(unsigned new_x, unsigned new_y) const;
};

} // namespace Core::Frontend
