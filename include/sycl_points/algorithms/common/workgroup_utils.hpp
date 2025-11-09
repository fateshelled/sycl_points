#pragma once

#include <algorithm>
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

    if (size <= 1) {
        return;
    }

    for (size_t k = 2; k <= size_power_of_2; k *= 2) {
        for (size_t j = k / 2; j > 0; j /= 2) {
            const size_t i = local_id;
            const size_t ixj = i ^ j;

            if (ixj > i && i < size_power_of_2) {
                const bool ascending = ((i & k) == 0);

                const KeyType val_i = (i < size) ? key_of(data[i]) : invalid_key;
                const KeyType val_ixj = (ixj < size) ? key_of(data[ixj]) : invalid_key;

                const bool should_swap = ascending ? compare(val_ixj, val_i) : compare(val_i, val_ixj);

                if (should_swap && i < size && ixj < size) {
                    std::swap(data[i], data[ixj]);
                }
            }

            item.barrier(sycl::access::fence_space::local_space);
        }
    }
}

/// @brief Reduce consecutive entries sharing the same key within a sorted local buffer.
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
template <typename LocalData, typename KeyType, typename KeyFunc, typename EqualFunc, typename CombineFunc,
          typename ResetFunc>
SYCL_EXTERNAL void reduction_sorted_local_data(LocalData* data, const size_t size, const sycl::nd_item<1>& item,
                                               const KeyType invalid_key, KeyFunc key_of, EqualFunc equal,
                                               CombineFunc combine, ResetFunc reset) {
    const size_t local_id = item.get_local_id(0);
    const KeyType current_key = (local_id < size) ? key_of(data[local_id]) : invalid_key;

    const bool has_prev = (local_id > 0) && (local_id - 1 < size);
    const KeyType previous_key = has_prev ? key_of(data[local_id - 1]) : invalid_key;

    const bool is_segment_start = (current_key != invalid_key) && (!has_prev || !equal(previous_key, current_key));

    if (is_segment_start) {
        for (size_t i = local_id + 1; i < size; ++i) {
            const KeyType next_key = key_of(data[i]);
            if (!equal(next_key, current_key)) {
                break;
            }

            combine(data[local_id], data[i]);
            reset(data[i]);
        }
    }

    item.barrier(sycl::access::fence_space::local_space);
}

/// @brief Populate local work-group data and optionally aggregate voxels using the shared utilities.
/// @tparam Aggregate When true the routine performs bitonic sort and reduction after loading the data.
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
/// @param wg_size_power_of_2 Power-of-two value equal to or larger than the work-group size.
/// @param item Work-item descriptor providing local identifiers and synchronization primitives.
/// @param loader Functor responsible for writing the LocalData entry when the global id is valid.
/// @param combine Functor merging two LocalData entries when aggregation is required.
/// @param reset Functor clearing a LocalData entry when it becomes invalid.
/// @param invalid_key Sentinel key used for inactive entries.
/// @param key_of Functor returning the key from a local entry.
/// @param compare Comparator defining ascending order for the keys.
/// @param equal Equality predicate that identifies entries belonging to the same group.
template <bool Aggregate, typename LocalData, typename LoaderFunc, typename CombineFunc, typename ResetFunc,
          typename KeyType, typename KeyFunc, typename CompareFunc, typename EqualFunc>
SYCL_EXTERNAL void local_reduction(LocalData* local_data, const size_t point_num, const size_t wg_size,
                                   const size_t wg_size_power_of_2, const sycl::nd_item<1>& item, LoaderFunc loader,
                                   CombineFunc combine, ResetFunc reset, const KeyType invalid_key, KeyFunc key_of,
                                   CompareFunc compare, EqualFunc equal) {
    const size_t local_id = item.get_local_id(0);
    const size_t global_id = item.get_global_id(0);

    if (global_id < point_num) {
        loader(local_data[local_id], global_id);
    } else {
        reset(local_data[local_id]);
    }

    item.barrier(sycl::access::fence_space::local_space);

    if constexpr (Aggregate) {
        const size_t group_id = item.get_group(0);
        const size_t group_offset = group_id * wg_size;
        const size_t remaining_points = (point_num > group_offset) ? point_num - group_offset : 0;
        const size_t active_size = std::min(wg_size, remaining_points);

        bitonic_sort_local_data(local_data, active_size, wg_size_power_of_2, item, invalid_key, key_of, compare);
        reduction_sorted_local_data(local_data, active_size, item, invalid_key, key_of, equal, combine, reset);
    }
}

}  // namespace common
}  // namespace algorithms
}  // namespace sycl_points

