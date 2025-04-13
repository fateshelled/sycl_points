#pragma once

#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace core {

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
SYCL_EXTERNAL inline bool quick_sort(T* arr, int start, int end) {
    const int size = end - start;
    if (size <= 1) return true;

    if (MAX_STACK_SIZE < 2 * static_cast<size_t>(std::log2f(static_cast<float>(size))) + 1) {
        return false;
    }

    // stack base
    int stack[MAX_STACK_SIZE][2];
    int top = 0;

    stack[top][0] = start;
    stack[top][1] = end;
    ++top;

    while (top > 0) {
        --top;
        const auto l = stack[top][0];
        const auto r = stack[top][1];

        if (l >= r) continue;

        // pivot
        const int mid = l + (r - l) / 2;
        if (arr[mid] < arr[l]) {
            const auto temp = arr[l];
            arr[l] = arr[mid];
            arr[mid] = temp;
        }
        if (arr[r] < arr[l]) {
            const auto temp = arr[l];
            arr[l] = arr[r];
            arr[r] = temp;
        }
        if (arr[mid] < arr[r]) {
            const auto temp = arr[mid];
            arr[mid] = arr[r];
            arr[r] = temp;
        }

        const auto pivot = arr[r];
        // partition
        int i = l - 1;
        for (int j = l; j < r; ++j) {
            if (arr[j] <= pivot) {
                ++i;
                const auto temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        ++i;

        {
            const auto temp = arr[i];
            arr[i] = arr[r];
            arr[r] = temp;
        }

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
SYCL_EXTERNAL inline bool quick_sort_descending(T* arr, int start, int end) {
    const int size = end - start;
    if (size <= 1) return true;

    if (MAX_STACK_SIZE < 2 * static_cast<size_t>(std::log2f(static_cast<float>(size))) + 1) {
        return false;
    }

    // stack base
    int stack[MAX_STACK_SIZE][2];
    int top = 0;

    stack[top][0] = start;
    stack[top][1] = end;
    ++top;

    while (top > 0) {
        --top;
        const auto l = stack[top][0];
        const auto r = stack[top][1];

        if (l >= r) continue;

        // pivot with median-of-three (reversed comparison for descending order)
        const int mid = l + (r - l) / 2;
        if (arr[mid] > arr[l]) {  // Changed comparison
            const auto temp = arr[l];
            arr[l] = arr[mid];
            arr[mid] = temp;
        }
        if (arr[r] > arr[l]) {  // Changed comparison
            const auto temp = arr[l];
            arr[l] = arr[r];
            arr[r] = temp;
        }
        if (arr[mid] > arr[r]) {  // Changed comparison
            const auto temp = arr[mid];
            arr[mid] = arr[r];
            arr[r] = temp;
        }

        const auto pivot = arr[r];
        // partition (reversed comparison for descending order)
        int i = l - 1;
        for (int j = l; j < r; ++j) {
            if (arr[j] >= pivot) {  // Changed comparison
                ++i;
                const auto temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        ++i;

        {
            const auto temp = arr[i];
            arr[i] = arr[r];
            arr[r] = temp;
        }

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

}  // namespace core

}  // namespace algorithms

}  // namespace sycl_points
