#pragma once

#include <algorithm>
#include <array>
#include <sycl/sycl.hpp>

namespace sycl_points {
namespace algorithms {
namespace common {

/// @brief Perform a bitonic sort on local work-group data using virtual padding for incomplete groups.
/// @tparam LocalData Structure type stored in local memory.
/// @tparam KeyType Key type returned by @p key_of.
/// @tparam KeyFunc Functor that returns the sort key for a given entry.
/// @tparam CompareFunc Strict weak ordering comparator that defines ascending order.
/// @param data Pointer to the local memory buffer containing the elements to sort.
/// @param size Number of valid entries present in the buffer for the current work-group.
/// @param size_power_of_2 Power-of-two value equal to or larger than the work-group size.
/// @param item Work-item descriptor providing local identifiers and synchronization primitives.
/// @param invalid_key Sentinel key used for inactive entries.
/// @param key_of Functor returning the key from a local entry.
/// @param compare Comparator defining strict weak ordering for the keys.
template <typename LocalData, typename KeyType, typename KeyFunc, typename CompareFunc>
SYCL_EXTERNAL void bitonic_sort_local_data(LocalData* data, const size_t size, const size_t size_power_of_2,
                                           const sycl::nd_item<1>& item, const KeyType invalid_key, KeyFunc key_of,
                                           CompareFunc compare) {
    const size_t local_id = item.get_local_id(0);
    const size_t local_range = item.get_local_range(0);

    if (size <= 1) {
        return;
    }

    for (size_t k = 2; k <= size_power_of_2; k *= 2) {
        for (size_t j = k / 2; j > 0; j /= 2) {
            for (size_t i = local_id; i < size_power_of_2; i += local_range) {
                const size_t ixj = i ^ j;

                if (ixj > i) {
                    const bool ascending = ((i & k) == 0);

                    const KeyType val_i = (i < size) ? key_of(data[i]) : invalid_key;
                    const KeyType val_ixj = (ixj < size) ? key_of(data[ixj]) : invalid_key;

                    const bool should_swap = ascending ? compare(val_ixj, val_i) : compare(val_i, val_ixj);

                    if (should_swap && i < size && ixj < size) {
                        std::swap(data[i], data[ixj]);
                    }
                }
            }

            item.barrier(sycl::access::fence_space::local_space);
        }
    }
}

/// @brief Reduce consecutive entries sharing the same key within a sorted local buffer.
/// @tparam PointsPerThread Number of local entries handled by each work-item.
/// @tparam LocalData Structure type stored in local memory.
/// @tparam KeyType Key type returned by @p key_of.
/// @tparam KeyFunc Functor that extracts the key from a local entry.
/// @tparam EqualFunc Functor that returns true when two keys belong to the same group.
/// @tparam CombineFunc Functor merging the payload of two LocalData entries.
/// @tparam ResetFunc Functor clearing a LocalData entry after it is merged.
/// @param data Pointer to the sorted local memory buffer.
/// @param size Number of active elements in the buffer for the current work-group.
/// @param item Work-item descriptor providing local identifiers and synchronization primitives.
/// @param invalid_key Sentinel key used for inactive entries.
/// @param key_of Functor returning the key from a local entry.
/// @param equal Equality predicate that identifies entries belonging to the same group.
///
/// The algorithm performs a parallel tree reduction by iteratively accumulating contributions from
/// neighbours located at exponentially increasing offsets. After @p log2(size) steps the first
/// element of each segment holds the aggregated value, while all other entries belonging to that
/// segment are reset.
template <size_t PointsPerThread = 1, typename LocalData, typename KeyType, typename KeyFunc, typename EqualFunc,
          typename CombineFunc, typename ResetFunc>
SYCL_EXTERNAL void reduction_sorted_local_data(LocalData* data, const size_t size, const sycl::nd_item<1>& item,
                                               const KeyType invalid_key, KeyFunc key_of, EqualFunc equal,
                                               CombineFunc combine, ResetFunc reset) {
    const size_t local_id = item.get_local_id(0);
    const size_t local_range = item.get_local_range(0);

    std::array<bool, PointsPerThread> should_add{};
    std::array<size_t, PointsPerThread> indices{};
    std::array<LocalData, PointsPerThread> contributions{};

    for (size_t stride = 1; stride < size; stride *= 2) {
        item.barrier(sycl::access::fence_space::local_space);

        size_t processed = 0;
        for (size_t idx = local_id; idx < size; idx += local_range) {
            indices[processed] = idx;
            should_add[processed] = false;

            const KeyType current_key = key_of(data[idx]);

            if (current_key != invalid_key) {
                const size_t neighbor_index = idx + stride;
                if (neighbor_index < size) {
                    const KeyType neighbor_key = key_of(data[neighbor_index]);

                    if (equal(current_key, neighbor_key)) {
                        contributions[processed] = data[neighbor_index];
                        should_add[processed] = true;
                    }
                }
            }

            ++processed;
        }

        item.barrier(sycl::access::fence_space::local_space);

        for (size_t i = 0; i < processed; ++i) {
            if (should_add[i]) {
                combine(data[indices[i]], contributions[i]);
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    for (size_t idx = local_id; idx < size; idx += local_range) {
        const KeyType current_key = key_of(data[idx]);

        bool should_reset = (current_key == invalid_key);

        if (!should_reset && idx > 0) {
            const KeyType prev_key = key_of(data[idx - 1]);
            should_reset = equal(prev_key, current_key);
        }

        if (should_reset) {
            reset(data[idx]);
        }
    }

    item.barrier(sycl::access::fence_space::local_space);
}

/// @brief Populate local work-group data and optionally aggregate voxels using the shared utilities.
/// @tparam Aggregate When true the routine performs bitonic sort and reduction after loading the data.
/// @tparam PointsPerThread Number of points processed by each work-item.
/// @tparam LocalData Structure type stored in local memory.
/// @tparam LoaderFunc Functor that populates a LocalData entry from a global point index.
/// @tparam CombineFunc Functor merging the payload of two LocalData entries.
/// @tparam ResetFunc Functor clearing a LocalData entry.
/// @tparam KeyType Key type returned by @p key_of.
/// @tparam KeyFunc Functor returning the key from a local entry.
/// @tparam CompareFunc Comparator defining ascending order for the keys.
/// @tparam EqualFunc Equality predicate that identifies entries belonging to the same group.
/// @param local_data Pointer to the local memory buffer that stores temporary reductions.
/// @param point_num Total number of points processed by the kernel.
/// @param wg_size Size of the work-group executing the kernel.
/// @param wg_size_power_of_2 Power-of-two value equal to or larger than the number of local entries.
/// @param item Work-item descriptor providing local identifiers and synchronization primitives.
/// @param loader Functor responsible for writing the LocalData entry when the global id is valid.
/// @param combine Functor merging two LocalData entries when aggregation is required.
/// @param reset Functor clearing a LocalData entry when it becomes invalid.
/// @param invalid_key Sentinel key used for inactive entries.
/// @param key_of Functor returning the key from a local entry.
/// @param compare Comparator defining ascending order for the keys.
/// @param equal Equality predicate that identifies entries belonging to the same group.
template <bool Aggregate, size_t PointsPerThread = 1, typename LocalData, typename LoaderFunc, typename CombineFunc,
          typename ResetFunc, typename KeyType, typename KeyFunc, typename CompareFunc, typename EqualFunc>
SYCL_EXTERNAL void local_reduction(LocalData* local_data, const size_t point_num, const size_t wg_size,
                                   const size_t wg_size_power_of_2, const sycl::nd_item<1>& item, LoaderFunc loader,
                                   CombineFunc combine, ResetFunc reset, const KeyType invalid_key, KeyFunc key_of,
                                   CompareFunc compare, EqualFunc equal) {
    const size_t local_id = item.get_local_id(0);
    const size_t global_id = item.get_global_id(0);

    const size_t local_offset = local_id * PointsPerThread;
    const size_t global_offset = global_id * PointsPerThread;

    // Populate a contiguous block in local memory so that aggregation can cover all entries.
    for (size_t i = 0; i < PointsPerThread; ++i) {
        const size_t local_index = local_offset + i;
        const size_t global_index = global_offset + i;
        if (global_index < point_num) {
            loader(local_data[local_index], global_index);
        } else {
            reset(local_data[local_index]);
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    if constexpr (Aggregate) {
        const size_t group_id = item.get_group(0);
        const size_t group_offset = group_id * wg_size * PointsPerThread;
        const size_t remaining_points = (point_num > group_offset) ? point_num - group_offset : 0;
        const size_t active_size = std::min(wg_size * PointsPerThread, remaining_points);

        bitonic_sort_local_data(local_data, active_size, wg_size_power_of_2, item, invalid_key, key_of, compare);
        reduction_sorted_local_data<PointsPerThread>(local_data, active_size, item, invalid_key, key_of, equal,
                                                     combine, reset);
    }
}

}  // namespace common
}  // namespace algorithms
}  // namespace sycl_points

