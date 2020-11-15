// Copyright 2020 yuzu emulator team
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstddef>
#include <type_traits>

namespace Common {

template <typename N, typename D>
requires std::is_integral_v<N>&& std::is_unsigned_v<D>[[nodiscard]] constexpr auto DivCeil(
    N number, D divisor) {
    return (number + divisor - 1) / divisor;
}

template <typename N, typename D>
requires std::is_integral_v<N>&& std::is_unsigned_v<D>[[nodiscard]] constexpr auto DivCeilLog2(
    N value, D alignment_log2) {
    return (value + (N(1) << alignment_log2) - 1) >> alignment_log2;
}

} // namespace Common
