#pragma once

#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace sort {

namespace kernel {
template <typename T>
SYCL_EXTERNAL inline void heap_sort(T* const arr, const size_t size) {
    if (size <= 1) return;

    std::make_heap(arr, arr + size);

    for (size_t i = size; i > 1; --i) {
        std::pop_heap(arr, arr + i);
    }
}

template <typename T>
SYCL_EXTERNAL inline void heap_sort_descending(T* const arr, const size_t size) {
    if (size <= 1) return;

    std::make_heap(arr, arr + size, std::greater<T>());

    for (size_t i = size; i > 1; --i) {
        std::pop_heap(arr, arr + i, std::greater<T>());
    }
}

template <typename T, size_t MAX_STACK_SIZE = 32>
SYCL_EXTERNAL inline bool quick_sort(T* arr, int32_t start, int32_t end) {
    const int32_t size = end - start;
    if (size <= 1) return true;

    if (MAX_STACK_SIZE < 2 * static_cast<size_t>(sycl::log2(static_cast<float>(size))) + 1) {
        return false;
    }

    // stack base
    int32_t stack[MAX_STACK_SIZE][2];
    int32_t top = 0;

    stack[top][0] = start;
    stack[top][1] = end;
    ++top;

    while (top > 0) {
        --top;
        const auto l = stack[top][0];
        const auto r = stack[top][1];

        if (l >= r) continue;

        // pivot
        const int32_t mid = l + (r - l) / 2;
        if (arr[mid] < arr[l]) {
            std::swap(arr[mid], arr[l]);
        }
        if (arr[r] < arr[l]) {
            std::swap(arr[r], arr[l]);
        }
        if (arr[mid] < arr[r]) {
            std::swap(arr[mid], arr[r]);
        }

        const auto pivot = arr[r];
        // partition (TODO: parallel partition)
        int32_t i = l - 1;
        for (int32_t j = l; j < r; ++j) {
            if (arr[j] <= pivot) {
                ++i;
                std::swap(arr[i], arr[j]);
            }
        }
        ++i;

        std::swap(arr[i], arr[r]);

        if (i - 1 > l) {
            stack[top][0] = l;
            stack[top][1] = i - 1;
            ++top;
        }
        if (i + 1 < r) {
            stack[top][0] = i + 1;
            stack[top][1] = r;
            ++top;
        }
    }
    return true;
}

template <typename T, size_t MAX_STACK_SIZE = 32>
SYCL_EXTERNAL inline bool quick_sort_descending(T* arr, int32_t start, int32_t end) {
    const int32_t size = end - start;
    if (size <= 1) return true;

    if (MAX_STACK_SIZE < 2 * static_cast<size_t>(sycl::log2(static_cast<float>(size))) + 1) {
        return false;
    }

    // stack base
    int32_t stack[MAX_STACK_SIZE][2];
    int32_t top = 0;

    stack[top][0] = start;
    stack[top][1] = end;
    ++top;

    while (top > 0) {
        --top;
        const auto l = stack[top][0];
        const auto r = stack[top][1];

        if (l >= r) continue;

        // pivot with median-of-three (reversed comparison for descending order)
        const int32_t mid = l + (r - l) / 2;
        if (arr[mid] > arr[l]) {  // Changed comparison
            std::swap(arr[mid], arr[l]);
        }
        if (arr[r] > arr[l]) {  // Changed comparison
            std::swap(arr[r], arr[l]);
        }
        if (arr[mid] > arr[r]) {  // Changed comparison
            std::swap(arr[mid], arr[r]);
        }

        const auto pivot = arr[r];
        // partition (reversed comparison for descending order)
        int32_t i = l - 1;
        for (int32_t j = l; j < r; ++j) {
            if (arr[j] >= pivot) {  // Changed comparison
                ++i;
                std::swap(arr[i], arr[j]);
            }
        }
        ++i;

        std::swap(arr[i], arr[r]);

        if (i - 1 > l) {
            stack[top][0] = l;
            stack[top][1] = i - 1;
            ++top;
        }
        if (i + 1 < r) {
            stack[top][0] = i + 1;
            stack[top][1] = r;
            ++top;
        }
    }
    return true;
}

}  // namespace kernel

}  // namespace sort

}  // namespace algorithms

}  // namespace sycl_points
