// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <glad/glad.h>

#include <QApplication>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QMessageBox>
#include <QOffscreenSurface>
#include <QOpenGLWindow>
#include <QPainter>
#include <QScreen>
#include <QStringList>
#include <QWindow>
#ifdef HAS_VULKAN
#include <QVulkanWindow>
#endif

#include <fmt/format.h>

#include "common/assert.h"
#include "common/microprofile.h"
#include "common/scm_rev.h"
#include "core/core.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/frontend/scope_acquire_window_context.h"
#include "core/settings.h"
#include "input_common/keyboard.h"
#include "input_common/main.h"
#include "input_common/motion_emu.h"
#include "video_core/renderer_base.h"
#include "video_core/video_core.h"
#include "yuzu/bootmanager.h"
#include "yuzu/main.h"

EmuThread::EmuThread(GRenderWindow* render_window) : render_window(render_window) {}

void EmuThread::run() {
    render_window->MakeCurrent();

    MicroProfileOnThreadCreate("EmuThread");

    stop_run = false;

    emit LoadProgress(VideoCore::LoadCallbackStage::Prepare, 0, 0);

    Core::System::GetInstance().Renderer().Rasterizer().LoadDiskResources(
        stop_run, [this](VideoCore::LoadCallbackStage stage, std::size_t value, std::size_t total) {
            emit LoadProgress(stage, value, total);
        });

    emit LoadProgress(VideoCore::LoadCallbackStage::Complete, 0, 0);

    if (Settings::values.use_asynchronous_gpu_emulation) {
        // Release OpenGL context for the GPU thread
        render_window->DoneCurrent();
    }

    // holds whether the cpu was running during the last iteration,
    // so that the DebugModeLeft signal can be emitted before the
    // next execution step
    bool was_active = false;
    while (!stop_run) {
        if (running) {
            if (!was_active)
                emit DebugModeLeft();

            Core::System::ResultStatus result = Core::System::GetInstance().RunLoop();
            if (result != Core::System::ResultStatus::Success) {
                this->SetRunning(false);
                emit ErrorThrown(result, Core::System::GetInstance().GetStatusDetails());
            }

            was_active = running || exec_step;
            if (!was_active && !stop_run)
                emit DebugModeEntered();
        } else if (exec_step) {
            if (!was_active)
                emit DebugModeLeft();

            exec_step = false;
            Core::System::GetInstance().SingleStep();
            emit DebugModeEntered();
            yieldCurrentThread();

            was_active = false;
        } else {
            std::unique_lock<std::mutex> lock(running_mutex);
            running_cv.wait(lock, [this] { return IsRunning() || exec_step || stop_run; });
        }
    }

    // Shutdown the core emulation
    Core::System::GetInstance().Shutdown();

#if MICROPROFILE_ENABLED
    MicroProfileOnThreadExit();
#endif

    render_window->moveContext();
}

class GGLContext : public Core::Frontend::GraphicsContext {
public:
    explicit GGLContext(QOpenGLContext* shared_context) : surface() {
        context = std::make_unique<QOpenGLContext>(shared_context);
        surface.setFormat(shared_context->format());
        surface.create();
    }

    void MakeCurrent() override {
        context->makeCurrent(&surface);
    }

    void DoneCurrent() override {
        context->doneCurrent();
    }

    void SwapBuffers() override {}

private:
    std::unique_ptr<QOpenGLContext> context;
    QOffscreenSurface surface;
};

class GWidgetInternal : public QWindow {
public:
    GWidgetInternal(GRenderWindow* parent) : parent(parent) {}

    void resizeEvent(QResizeEvent* ev) override {
        parent->OnClientAreaResized(ev->size().width(), ev->size().height());
        parent->OnFramebufferSizeChanged();
    }

    void DisablePainting() {
        do_painting = false;
    }

    void EnablePainting() {
        do_painting = true;
    }

    virtual std::pair<unsigned, unsigned> GetSize() const = 0;

protected:
    bool IsPaintingEnabled() const {
        return do_painting;
    }

private:
    GRenderWindow* parent;
    bool do_painting = false;
};

// This class overrides paintEvent and resizeEvent to prevent the GUI thread from stealing GL
// context.
// The corresponding functionality is handled in EmuThread instead
class GGLWidgetInternal final : public GWidgetInternal, public QOpenGLWindow {
public:
    GGLWidgetInternal(GRenderWindow* parent, QOpenGLContext* shared_context)
        : GWidgetInternal(parent), QOpenGLWindow(shared_context) {}

    void paintEvent(QPaintEvent* ev) override {
        if (IsPaintingEnabled()) {
            QPainter painter(this);
        }
    }

    std::pair<unsigned, unsigned> GetSize() const override {
        return std::make_pair(QWindow::width(), QWindow::height());
    }
};

class GVKWidgetInternal final : public GWidgetInternal {
public:
    GVKWidgetInternal(GRenderWindow* parent, QVulkanInstance* instance) : GWidgetInternal(parent) {
        setSurfaceType(QSurface::SurfaceType::VulkanSurface);
        setVulkanInstance(instance);
    }

    std::pair<unsigned, unsigned> GetSize() const override {
        return std::make_pair(width(), height());
    }
};

GRenderWindow::GRenderWindow(QWidget* parent, EmuThread* emu_thread)
    : QWidget(parent), emu_thread(emu_thread) {
    setWindowTitle(QStringLiteral("yuzu %1 | %2-%3")
                       .arg(Common::g_build_name, Common::g_scm_branch, Common::g_scm_desc));
    setAttribute(Qt::WA_AcceptTouchEvents);

    InputCommon::Init();
    connect(this, &GRenderWindow::FirstFrameDisplayed, static_cast<GMainWindow*>(parent),
            &GMainWindow::OnLoadComplete);
}

GRenderWindow::~GRenderWindow() {
    InputCommon::Shutdown();
}

void GRenderWindow::moveContext() {
    if (!context) {
        return;
    }
    DoneCurrent();

    // If the thread started running, move the GL Context to the new thread. Otherwise, move it
    // back.
    auto thread = (QThread::currentThread() == qApp->thread() && emu_thread != nullptr)
                      ? emu_thread
                      : qApp->thread();
    context->moveToThread(thread);
}

void GRenderWindow::SwapBuffers() {
    // In our multi-threaded QWidget use case we shouldn't need to call `makeCurrent`,
    // since we never call `doneCurrent` in this thread.
    // However:
    // - The Qt debug runtime prints a bogus warning on the console if `makeCurrent` wasn't called
    // since the last time `swapBuffers` was executed;
    // - On macOS, if `makeCurrent` isn't called explicitely, resizing the buffer breaks.
    if (context) {
        context->makeCurrent(child);
        context->swapBuffers(child);
    }
    if (!first_frame) {
        emit FirstFrameDisplayed();
        first_frame = true;
    }
}

void GRenderWindow::MakeCurrent() {
    if (context) {
        context->makeCurrent(child);
    }
}

void GRenderWindow::DoneCurrent() {
    if (context) {
        context->doneCurrent();
    }
}

void GRenderWindow::PollEvents() {}

bool GRenderWindow::IsShown() const {
    return !isMinimized();
}

void GRenderWindow::RetrieveVulkanHandlers(void** get_instance_proc_addr, void** instance,
                                           void** surface) const {
#ifdef HAS_VULKAN
    *get_instance_proc_addr =
        static_cast<void*>(this->instance->getInstanceProcAddr("vkGetInstanceProcAddr"));
    *instance = static_cast<void*>(this->instance->vkInstance());
    *surface = this->instance->surfaceForWindow(child);
#else
    UNREACHABLE_MSG("Executing Vulkan code without compiling Vulkan");
#endif
}

// On Qt 5.0+, this correctly gets the size of the framebuffer (pixels).
//
// Older versions get the window size (density independent pixels),
// and hence, do not support DPI scaling ("retina" displays).
// The result will be a viewport that is smaller than the extent of the window.
void GRenderWindow::OnFramebufferSizeChanged() {
    // Screen changes potentially incur a change in screen DPI, hence we should update the
    // framebuffer size
    const qreal pixelRatio = windowPixelRatio();
    const auto size = child->GetSize();
    UpdateCurrentFramebufferLayout(size.first * pixelRatio, size.second * pixelRatio);
}

void GRenderWindow::BackupGeometry() {
    geometry = ((QWidget*)this)->saveGeometry();
}

void GRenderWindow::RestoreGeometry() {
    // We don't want to back up the geometry here (obviously)
    QWidget::restoreGeometry(geometry);
}

void GRenderWindow::restoreGeometry(const QByteArray& geometry) {
    // Make sure users of this class don't need to deal with backing up the geometry themselves
    QWidget::restoreGeometry(geometry);
    BackupGeometry();
}

QByteArray GRenderWindow::saveGeometry() {
    // If we are a top-level widget, store the current geometry
    // otherwise, store the last backup
    if (parent() == nullptr)
        return ((QWidget*)this)->saveGeometry();
    else
        return geometry;
}

qreal GRenderWindow::windowPixelRatio() const {
    // windowHandle() might not be accessible until the window is displayed to screen.
    return windowHandle() ? windowHandle()->screen()->devicePixelRatio() : 1.0f;
}

std::pair<unsigned, unsigned> GRenderWindow::ScaleTouch(const QPointF pos) const {
    const qreal pixel_ratio = windowPixelRatio();
    return {static_cast<unsigned>(std::max(std::round(pos.x() * pixel_ratio), qreal{0.0})),
            static_cast<unsigned>(std::max(std::round(pos.y() * pixel_ratio), qreal{0.0}))};
}

void GRenderWindow::closeEvent(QCloseEvent* event) {
    emit Closed();
    QWidget::closeEvent(event);
}

void GRenderWindow::keyPressEvent(QKeyEvent* event) {
    InputCommon::GetKeyboard()->PressKey(event->key());
}

void GRenderWindow::keyReleaseEvent(QKeyEvent* event) {
    InputCommon::GetKeyboard()->ReleaseKey(event->key());
}

void GRenderWindow::mousePressEvent(QMouseEvent* event) {
    if (event->source() == Qt::MouseEventSynthesizedBySystem)
        return; // touch input is handled in TouchBeginEvent

    auto pos = event->pos();
    if (event->button() == Qt::LeftButton) {
        const auto [x, y] = ScaleTouch(pos);
        this->TouchPressed(x, y);
    } else if (event->button() == Qt::RightButton) {
        InputCommon::GetMotionEmu()->BeginTilt(pos.x(), pos.y());
    }
}

void GRenderWindow::mouseMoveEvent(QMouseEvent* event) {
    if (event->source() == Qt::MouseEventSynthesizedBySystem)
        return; // touch input is handled in TouchUpdateEvent

    auto pos = event->pos();
    const auto [x, y] = ScaleTouch(pos);
    this->TouchMoved(x, y);
    InputCommon::GetMotionEmu()->Tilt(pos.x(), pos.y());
}

void GRenderWindow::mouseReleaseEvent(QMouseEvent* event) {
    if (event->source() == Qt::MouseEventSynthesizedBySystem)
        return; // touch input is handled in TouchEndEvent

    if (event->button() == Qt::LeftButton)
        this->TouchReleased();
    else if (event->button() == Qt::RightButton)
        InputCommon::GetMotionEmu()->EndTilt();
}

void GRenderWindow::TouchBeginEvent(const QTouchEvent* event) {
    // TouchBegin always has exactly one touch point, so take the .first()
    const auto [x, y] = ScaleTouch(event->touchPoints().first().pos());
    this->TouchPressed(x, y);
}

void GRenderWindow::TouchUpdateEvent(const QTouchEvent* event) {
    QPointF pos;
    int active_points = 0;

    // average all active touch points
    for (const auto tp : event->touchPoints()) {
        if (tp.state() & (Qt::TouchPointPressed | Qt::TouchPointMoved | Qt::TouchPointStationary)) {
            active_points++;
            pos += tp.pos();
        }
    }

    pos /= active_points;

    const auto [x, y] = ScaleTouch(pos);
    this->TouchMoved(x, y);
}

void GRenderWindow::TouchEndEvent() {
    this->TouchReleased();
}

bool GRenderWindow::event(QEvent* event) {
    if (event->type() == QEvent::TouchBegin) {
        TouchBeginEvent(static_cast<QTouchEvent*>(event));
        return true;
    } else if (event->type() == QEvent::TouchUpdate) {
        TouchUpdateEvent(static_cast<QTouchEvent*>(event));
        return true;
    } else if (event->type() == QEvent::TouchEnd || event->type() == QEvent::TouchCancel) {
        TouchEndEvent();
        return true;
    }

    return QWidget::event(event);
}

void GRenderWindow::focusOutEvent(QFocusEvent* event) {
    QWidget::focusOutEvent(event);
    InputCommon::GetKeyboard()->ReleaseAllKeys();
}

void GRenderWindow::OnClientAreaResized(unsigned width, unsigned height) {
    NotifyClientAreaSizeChanged(std::make_pair(width, height));
}

std::unique_ptr<Core::Frontend::GraphicsContext> GRenderWindow::CreateSharedContext() const {
    return std::make_unique<GGLContext>(shared_context.get());
}

bool GRenderWindow::InitRenderTarget() {
    if (shared_context) {
        shared_context.reset();
    }

    if (context) {
        context.reset();
    }

    if (child) {
        delete child;
    }

    if (container) {
        delete container;
    }

    if (layout()) {
        delete layout();
    }

    first_frame = false;

    switch (Settings::values.renderer_backend) {
    case Settings::RendererBackend::OpenGL:
        if (!InitializeOpenGL())
            return false;
        break;
    case Settings::RendererBackend::Vulkan:
        if (!InitializeVulkan())
            return false;
        break;
    }

    container = QWidget::createWindowContainer(child, this);
    QBoxLayout* layout = new QHBoxLayout(this);

    layout->addWidget(container);
    layout->setMargin(0);
    setLayout(layout);

    // Show causes the window to actually be created and the gl context as well, but we don't want
    // the widget to be shown yet, so immediately hide it.
    show();
    hide();

    resize(Layout::ScreenUndocked::Width, Layout::ScreenUndocked::Height);
    child->resize(Layout::ScreenUndocked::Width, Layout::ScreenUndocked::Height);
    container->resize(Layout::ScreenUndocked::Width, Layout::ScreenUndocked::Height);

    OnMinimalClientAreaChangeRequest(GetActiveConfig().min_client_area_size);

    OnFramebufferSizeChanged();
    NotifyClientAreaSizeChanged(child->GetSize());

    BackupGeometry();

    if (Settings::values.renderer_backend == Settings::RendererBackend::OpenGL) {
        if (!LoadOpenGL()) {
            return false;
        }
    }

    return true;
}

void GRenderWindow::CaptureScreenshot(u16 res_scale, const QString& screenshot_path) {
    auto& renderer = Core::System::GetInstance().Renderer();

    if (!res_scale)
        res_scale = VideoCore::GetResolutionScaleFactor(renderer);

    const Layout::FramebufferLayout layout{Layout::FrameLayoutFromResolutionScale(res_scale)};
    screenshot_image = QImage(QSize(layout.width, layout.height), QImage::Format_RGB32);
    renderer.RequestScreenshot(screenshot_image.bits(),
                               [=] {
                                   screenshot_image.mirrored(false, true).save(screenshot_path);
                                   LOG_INFO(Frontend, "The screenshot is saved.");
                               },
                               layout);
}

void GRenderWindow::OnMinimalClientAreaChangeRequest(
    const std::pair<unsigned, unsigned>& minimal_size) {
    setMinimumSize(minimal_size.first, minimal_size.second);
}

bool GRenderWindow::InitializeOpenGL() {
    // TODO: One of these flags might be interesting: WA_OpaquePaintEvent, WA_NoBackground,
    // WA_DontShowOnScreen, WA_DeleteOnClose
    QSurfaceFormat fmt;
    fmt.setVersion(4, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    // TODO: Expose a setting for buffer value (ie default/single/double/triple)
    fmt.setSwapBehavior(QSurfaceFormat::DefaultSwapBehavior);
    shared_context = std::make_unique<QOpenGLContext>();
    shared_context->setFormat(fmt);
    shared_context->create();
    context = std::make_unique<QOpenGLContext>();
    context->setShareContext(shared_context.get());
    context->setFormat(fmt);
    context->create();
    fmt.setSwapInterval(false);

    child = new GGLWidgetInternal(this, shared_context.get());
    return true;
}

bool GRenderWindow::InitializeVulkan() {
#ifdef HAS_VULKAN
    instance = new QVulkanInstance();
    instance->setApiVersion(QVersionNumber(1, 1, 0));
    instance->setFlags(QVulkanInstance::Flag::NoDebugOutputRedirect);
    if (Settings::values.renderer_debug) {
        const auto supported_layers{instance->supportedLayers()};
        const bool found =
            std::find_if(supported_layers.begin(), supported_layers.end(), [](const auto& layer) {
                constexpr const char searched_layer[] = "VK_LAYER_LUNARG_standard_validation";
                return layer.name == searched_layer;
            });
        if (found) {
            instance->setLayers(QByteArrayList() << "VK_LAYER_LUNARG_standard_validation");
            instance->setExtensions(QByteArrayList() << VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }
    }
    if (!instance->create()) {
        QMessageBox::critical(
            this, tr("Error while initializing Vulkan 1.1!"),
            tr("Your OS doesn't seem to support Vulkan 1.1 instances, or you do not have the "
               "latest graphics drivers."));
        return false;
    }

    child = new GVKWidgetInternal(this, instance);
    return true;
#else
    QMessageBox::critical(this, tr("Vulkan not available!"),
                          tr("yuzu has not been compiled with Vulkan support."));
    return false;
#endif
}

bool GRenderWindow::LoadOpenGL() {
    Core::Frontend::ScopeAcquireWindowContext acquire_context{*this};
    if (!gladLoadGL()) {
        QMessageBox::critical(this, tr("Error while initializing OpenGL 4.3 Core!"),
                              tr("Your GPU may not support OpenGL 4.3, or you do not have the "
                                 "latest graphics driver."));
        return false;
    }

    QStringList unsupported_gl_extensions = GetUnsupportedGLExtensions();
    if (!unsupported_gl_extensions.empty()) {
        QMessageBox::critical(
            this, tr("Error while initializing OpenGL Core!"),
            tr("Your GPU may not support one or more required OpenGL extensions. Please ensure you "
               "have the latest graphics driver.<br><br>Unsupported extensions:<br>") +
                unsupported_gl_extensions.join("<br>"));
        return false;
    }
    return true;
}

QStringList GRenderWindow::GetUnsupportedGLExtensions() const {
    QStringList unsupported_ext;

    if (!GLAD_GL_ARB_direct_state_access)
        unsupported_ext.append("ARB_direct_state_access");
    if (!GLAD_GL_ARB_vertex_type_10f_11f_11f_rev)
        unsupported_ext.append("ARB_vertex_type_10f_11f_11f_rev");
    if (!GLAD_GL_ARB_texture_mirror_clamp_to_edge)
        unsupported_ext.append("ARB_texture_mirror_clamp_to_edge");
    if (!GLAD_GL_ARB_multi_bind)
        unsupported_ext.append("ARB_multi_bind");

    // Extensions required to support some texture formats.
    if (!GLAD_GL_EXT_texture_compression_s3tc)
        unsupported_ext.append("EXT_texture_compression_s3tc");
    if (!GLAD_GL_ARB_texture_compression_rgtc)
        unsupported_ext.append("ARB_texture_compression_rgtc");
    if (!GLAD_GL_ARB_depth_buffer_float)
        unsupported_ext.append("ARB_depth_buffer_float");

    for (const QString& ext : unsupported_ext)
        LOG_CRITICAL(Frontend, "Unsupported GL extension: {}", ext.toStdString());

    return unsupported_ext;
}

void GRenderWindow::OnEmulationStarting(EmuThread* emu_thread) {
    this->emu_thread = emu_thread;
    child->DisablePainting();
}

void GRenderWindow::OnEmulationStopping() {
    emu_thread = nullptr;
    child->EnablePainting();
}

void GRenderWindow::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);

    // windowHandle() is not initialized until the Window is shown, so we connect it here.
    connect(windowHandle(), &QWindow::screenChanged, this, &GRenderWindow::OnFramebufferSizeChanged,
            Qt::UniqueConnection);
}
