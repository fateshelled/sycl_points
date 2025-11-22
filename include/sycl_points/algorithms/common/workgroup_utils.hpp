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

/// @brief Perform a sub-group segmented reduction on sorted local data to shrink identical keys.
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
/// The routine keeps only the head element of each identical-key segment within a sub-group and
/// accumulates contributions from following lanes. Non-head entries are reset so that subsequent
/// work-group reductions process fewer items and require fewer barriers.
template <typename LocalData, typename KeyType, typename KeyFunc, typename EqualFunc, typename CombineFunc,
          typename ResetFunc>
SYCL_EXTERNAL void subgroup_reduction_sorted_local_data(LocalData* data, const size_t size,
                                                         const sycl::nd_item<1>& item, const KeyType invalid_key,
                                                         KeyFunc key_of, EqualFunc equal, CombineFunc combine,
                                                         ResetFunc reset) {
    const size_t local_id = item.get_local_id(0);
    const bool is_active = (local_id < size);

    auto sub_group = item.get_sub_group();
    const size_t sg_local_id = sub_group.get_local_id()[0];
    const size_t sg_size = sub_group.get_local_range()[0];

    LocalData value{};
    KeyType key = invalid_key;

    if (is_active) {
        value = data[local_id];
        key = key_of(value);
    }

    // All work-items in a sub-group must execute the same collective operations.
    const KeyType prev_key = sycl::shift_group_right(sub_group, key, 1);
    const bool is_valid = is_active && (key != invalid_key);
    const bool is_head = is_valid && ((sg_local_id == 0) || !equal(key, prev_key));

    // Perform a manual segmented prefix reduction because sycl::inclusive_scan_over_group only supports
    // arithmetic types. This loop accumulates contributions from lanes with the same key that appear
    // earlier in the sub-group, so the tail lane of each segment holds the full reduction.
    LocalData prefix = value;
    for (size_t offset = 1; offset < sg_size; offset <<= 1) {
        const KeyType neighbor_key = sycl::shift_group_right(sub_group, key, offset);
        const LocalData neighbor_value = sycl::shift_group_right(sub_group, prefix, offset);

        const bool neighbor_in_range = (sg_local_id >= offset) && (local_id >= offset);
        if (is_valid && neighbor_in_range && equal(key, neighbor_key)) {
            combine(prefix, neighbor_value);
        }
    }

    // Identify the tail lane for each contiguous segment and broadcast its reduction back to the head.
    const KeyType next_key = sycl::shift_group_left(sub_group, key, 1);
    const bool has_neighbor = (sg_local_id + 1 < sg_size) && ((local_id + 1) < size);
    const bool is_tail = is_valid && (!has_neighbor || !equal(key, next_key));

    size_t tail_lane = is_tail ? sg_local_id : sg_size;
    for (size_t offset = 1; offset < sg_size; offset <<= 1) {
        const size_t neighbor_lane = sycl::shift_group_left(sub_group, tail_lane, offset);
        const KeyType neighbor_key = sycl::shift_group_left(sub_group, key, offset);

        const bool neighbor_in_range = (sg_local_id + offset < sg_size) && (local_id + offset < size);
        if (is_valid && neighbor_in_range && equal(key, neighbor_key)) {
            tail_lane = std::min(tail_lane, neighbor_lane);
        }
    }

    const size_t safe_tail_lane = (tail_lane < sg_size) ? tail_lane : sg_local_id;
    const LocalData segment_total = sycl::select_from_group(sub_group, prefix, safe_tail_lane);

    if (is_head) {
        data[local_id] = segment_total;
    } else if (is_active) {
        reset(data[local_id]);
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
///
/// The algorithm performs a parallel tree reduction by iteratively accumulating contributions from
/// neighbours located at exponentially increasing offsets. After @p log2(size) steps the first
/// element of each segment holds the aggregated value, while all other entries belonging to that
/// segment are reset.
template <typename LocalData, typename KeyType, typename KeyFunc, typename EqualFunc, typename CombineFunc,
          typename ResetFunc>
SYCL_EXTERNAL void reduction_sorted_local_data(LocalData* data, const size_t size, const sycl::nd_item<1>& item,
                                               const KeyType invalid_key, KeyFunc key_of, EqualFunc equal,
                                               CombineFunc combine, ResetFunc reset) {
    const size_t local_id = item.get_local_id(0);
    for (size_t stride = 1; stride < size; stride *= 2) {
        item.barrier(sycl::access::fence_space::local_space);

        bool should_add = false;
        LocalData contribution{};

        if (local_id < size) {
            const KeyType current_key = key_of(data[local_id]);

            if (current_key != invalid_key) {
                const size_t neighbor_index = local_id + stride;
                if (neighbor_index < size) {
                    const KeyType neighbor_key = key_of(data[neighbor_index]);

                    if (equal(current_key, neighbor_key)) {
                        contribution = data[neighbor_index];
                        should_add = true;
                    }
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (should_add) {
            combine(data[local_id], contribution);
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    KeyType current_key = invalid_key;
    if (local_id < size) {
        current_key = key_of(data[local_id]);
    }

    KeyType previous_valid_key = invalid_key;
    if (local_id > 0 && local_id < size) {
        previous_valid_key = key_of(data[local_id - 1]);
    }

    if (local_id < size) {
        const bool is_valid = (current_key != invalid_key);
        const bool has_previous_valid = (previous_valid_key != invalid_key);
        const bool should_reset = !is_valid || (has_previous_valid && equal(previous_valid_key, current_key));

        if (should_reset) {
            reset(data[local_id]);
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
        subgroup_reduction_sorted_local_data(local_data, active_size, item, invalid_key, key_of, equal, combine,
                                             reset);
        item.barrier(sycl::access::fence_space::local_space);
        bitonic_sort_local_data(local_data, active_size, wg_size_power_of_2, item, invalid_key, key_of, compare);
        reduction_sorted_local_data(local_data, active_size, item, invalid_key, key_of, equal, combine, reset);
    }
}

}  // namespace common
}  // namespace algorithms
}  // namespace sycl_points

