// Copyright 2020 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstddef>

namespace VideoCommon {

enum class HostBufferType {
    Upload,
    Download,
    DeviceLocal,
};
constexpr size_t NUM_HOST_BUFFER_TYPES = 3;

} // namespace VideoCommon
