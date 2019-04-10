// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"
#include "video_core/surface.h"

namespace Tegra::Texture {

void ConvertFromGuestToHost(u8* data, VideoCore::Surface::PixelFormat pixel_format,
                            std::size_t width, std::size_t height, std::size_t depth,
                            bool convert_astc, bool convert_s8z24);

void ConvertFromHostToGuest(u8* data, VideoCore::Surface::PixelFormat pixel_format,
                            std::size_t width, std::size_t height, std::size_t depth,
                            bool convert_astc, bool convert_s8z24);

} // namespace Tegra::Texture