// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <utility>
#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/declarations.h"
#include "video_core/renderer_vulkan/vk_device.h"
#include "video_core/surface.h"

namespace Vulkan::MaxwellToVK {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using PixelFormat = VideoCore::Surface::PixelFormat;
using ComponentType = VideoCore::Surface::ComponentType;

std::pair<vk::Format, bool> SurfaceFormat(const VKDevice& device, FormatType format_type,
                                          PixelFormat pixel_format, ComponentType component_type);

vk::ShaderStageFlagBits ShaderStage(Maxwell::ShaderStage stage);

vk::PrimitiveTopology PrimitiveTopology(Maxwell::PrimitiveTopology topology);

vk::Format VertexFormat(Maxwell::VertexAttribute::Type type, Maxwell::VertexAttribute::Size size);

vk::CompareOp ComparisonOp(Maxwell::ComparisonOp comparison);

vk::IndexType IndexFormat(Maxwell::IndexFormat index_format);

vk::StencilOp StencilOp(Maxwell::StencilOp stencil_op);

vk::BlendOp BlendEquation(Maxwell::Blend::Equation equation);

vk::BlendFactor BlendFactor(Maxwell::Blend::Factor factor);

vk::FrontFace FrontFace(Maxwell::Cull::FrontFace front_face);

vk::CullModeFlags CullFace(Maxwell::Cull::CullFace cull_face);

} // namespace Vulkan::MaxwellToVK