// Copyright 2018 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include "common/assert.h"

/*
 * Creates a vector with a size limit.
 *
 *
 * It's useful to avoid the following pattern without using std::vector<>:
 *
 * int foo_count = 0;
 * T foos[N];
 * foos[foo_count++] = foo;
 * foos[foo_count++] = foo;
 *
 *
 * Usage example:
 *
 * StaticVector<int, SOME_SIZE_LIMIT> static_vector;
 * static_vector.Push(20);
 * static_vector.Push(40);
 * static_vector.Push(60);
 * Foo(static_vector.Size(), static_vector.Data());
 */
template <typename T, std::size_t N>
class StaticVector {
public:
    void Push(const T&& value) {
        const std::size_t index = count++;
        DEBUG_ASSERT_MSG(index < N, "Static vector overflow");

        array[index] = std::move(value);
    }

    void Push(const T& value) {
        const std::size_t index = count++;
        DEBUG_ASSERT_MSG(index < N, "Static vector overflow");

        array[index] = value;
    }

    /// Gets current size of the vector.
    std::size_t Size() const {
        return count;
    }

    /// Gets a pointer to the data.
    T* Data() {
        return array.data();
    }

    /// Gets a pointer to the data.
    const auto* Data() const {
        return array.data();
    }

    T& operator[](std::size_t i) {
        DEBUG_ASSERT(i < N);
        return array[i];
    }

    const T& operator[](std::size_t i) const {
        DEBUG_ASSERT(i < N);
        return array[i];
    }

    auto begin() const {
        return array.begin();
    }

    auto end() const {
        return array.begin() + count;
    }

    auto begin() {
        return array.begin();
    }

    auto end() {
        return array.begin() + count;
    }

    /// Returns the capacity of the vector.
    static constexpr std::size_t Capacity() {
        return N;
    }

    [[nodiscard]] bool operator<(const StaticVector<T, N>& rhs) const {
        return std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
    }

    [[nodiscard]] bool operator==(const StaticVector<T, N>& rhs) const {
        if (count != rhs.count)
            return false;
        for (std::size_t i = 0; i < count; ++i) {
            if (array[i] != rhs.array[i])
                return false;
        }
        return true;
    }

    [[nodiscard]] bool operator!=(const StaticVector<T, N>& rhs) const {
        return !operator==(rhs);
    }

private:
    std::size_t count = 0;
    std::array<T, N> array{};
};
