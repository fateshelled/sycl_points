#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sycl_points/algorithms/common/transform.hpp"
#include "sycl_points/algorithms/common/voxel_constants.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points::algorithms::localization {

namespace voxel_detail {

/// @brief Mix a voxel key for open addressing.
SYCL_EXTERNAL inline uint64_t hash_voxel_key(uint64_t key) {
    key ^= key >> 30;
    key *= 0xbf58476d1ce4e5b9ULL;
    key ^= key >> 27;
    key *= 0x94d049bb133111ebULL;
    key ^= key >> 31;
    return key;
}

/// @brief Test whether a voxel key is in the table.
/// @note The table has at least twice as many slots as occupied voxels.
SYCL_EXTERNAL inline bool is_voxel_occupied(const uint64_t* keys, uint64_t mask, uint64_t key) {
    constexpr int max_probe = 32;
    uint64_t slot = hash_voxel_key(key) & mask;
    for (int probe = 0; probe < max_probe; ++probe) {
        const uint64_t stored = keys[slot];
        if (stored == key) {
            return true;
        }
        if (stored == VoxelConstants::invalid_coord) {
            return false;
        }
        slot = (slot + 1) & mask;
    }
    return false;
}

/// @brief Voxel coordinate in the offset encoding that the device uses.
struct HostVoxelCoord {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;
};

/// @brief Compute a voxel coordinate on the host with the device encoding.
/// @return False if the point is not finite or the coordinate is out of range.
inline bool compute_voxel_coord_host(const PointType& point, float voxel_size_inv, HostVoxelCoord& coord) {
    if (!std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z())) {
        return false;
    }
    coord.x = static_cast<int64_t>(std::floor(point.x() * voxel_size_inv)) + VoxelConstants::coord_offset;
    coord.y = static_cast<int64_t>(std::floor(point.y() * voxel_size_inv)) + VoxelConstants::coord_offset;
    coord.z = static_cast<int64_t>(std::floor(point.z() * voxel_size_inv)) + VoxelConstants::coord_offset;
    return 0 <= coord.x && coord.x <= VoxelConstants::coord_bit_mask &&  //
           0 <= coord.y && coord.y <= VoxelConstants::coord_bit_mask &&  //
           0 <= coord.z && coord.z <= VoxelConstants::coord_bit_mask;
}

/// @brief Encode a voxel coordinate as the device key.
inline uint64_t encode_voxel_key(int64_t x, int64_t y, int64_t z) {
    return (static_cast<uint64_t>(x) << (VoxelConstants::coord_bit_size * CartesianCoordComponent::X)) |
           (static_cast<uint64_t>(y) << (VoxelConstants::coord_bit_size * CartesianCoordComponent::Y)) |
           (static_cast<uint64_t>(z) << (VoxelConstants::coord_bit_size * CartesianCoordComponent::Z));
}

inline size_t next_power_of_two(size_t value) {
    size_t result = 16;
    while (result < value) {
        if (result > std::numeric_limits<size_t>::max() / 2) {
            throw std::overflow_error("[VoxelOccupancyMap] capacity overflow");
        }
        result *= 2;
    }
    return result;
}

}  // namespace voxel_detail

/// @brief Multi-resolution voxel occupancy map that scores poses by occupied-voxel hits.
/// @note The score of a pose is the number of query points whose transformed voxel is occupied.
///       Each occupied voxel also marks its seven negative neighbors, as in 3D-BBS.
///       Thus a coarser level keeps an upper bound of the scores of its child nodes.
class VoxelOccupancyMap {
public:
    using Ptr = std::shared_ptr<VoxelOccupancyMap>;

    /// @brief Construct an empty map with one voxel size for each level.
    /// @param queue SYCL queue.
    /// @param voxel_sizes Voxel size of each level.
    VoxelOccupancyMap(const sycl_utils::DeviceQueue& queue, std::vector<float> voxel_sizes)
        : queue_(queue), voxel_sizes_(std::move(voxel_sizes)) {
        if (this->voxel_sizes_.empty()) {
            throw std::invalid_argument("[VoxelOccupancyMap] empty voxel size list");
        }
        this->voxel_size_inv_.resize(this->voxel_sizes_.size());
        for (size_t level = 0; level < this->voxel_sizes_.size(); ++level) {
            const float size = this->voxel_sizes_[level];
            if (!std::isfinite(size) || size <= 0.0f) {
                throw std::invalid_argument("[VoxelOccupancyMap] invalid voxel size");
            }
            this->voxel_size_inv_[level] = 1.0f / size;
        }
        this->capacities_.assign(this->voxel_sizes_.size(), 0);
        this->masks_.assign(this->voxel_sizes_.size(), 0);
        this->keys_.reserve(this->voxel_sizes_.size());
        for (size_t level = 0; level < this->voxel_sizes_.size(); ++level) {
            this->keys_.emplace_back(0, *this->queue_.ptr);
        }
    }

    /// @brief Build the occupancy tables from the target points.
    /// @throws std::invalid_argument if the cloud is empty or contains non-finite points.
    void set_target(const PointCloudShared& cloud) {
        if (!cloud.points || cloud.size() == 0) {
            throw std::invalid_argument("[VoxelOccupancyMap] empty cloud");
        }
        std::vector<uint64_t> keys;
        if (cloud.size() > std::numeric_limits<size_t>::max() / 8) {
            throw std::overflow_error("[VoxelOccupancyMap] key count overflow");
        }
        keys.reserve(cloud.size() * 8);
        for (const auto& point : *cloud.points) {
            if (!point.allFinite()) {
                throw std::invalid_argument("[VoxelOccupancyMap] non-finite point");
            }
        }
        for (size_t level = 0; level < this->voxel_sizes_.size(); ++level) {
            keys.clear();
            const float inv = this->voxel_size_inv_[level];
            for (const auto& point : *cloud.points) {
                voxel_detail::HostVoxelCoord coord;
                if (!voxel_detail::compute_voxel_coord_host(point, inv, coord)) {
                    continue;
                }
                // 3D-BBS inflates each occupied voxel by one cell toward the negative side.
                // The branch offsets are non-negative, so this keeps the parent upper bound valid.
                for (int dx = 0; dx >= -1; --dx) {
                    if (coord.x + dx < 0) {
                        continue;
                    }
                    for (int dy = 0; dy >= -1; --dy) {
                        if (coord.y + dy < 0) {
                            continue;
                        }
                        for (int dz = 0; dz >= -1; --dz) {
                            if (coord.z + dz < 0) {
                                continue;
                            }
                            keys.push_back(voxel_detail::encode_voxel_key(coord.x + dx, coord.y + dy, coord.z + dz));
                        }
                    }
                }
            }
            std::sort(keys.begin(), keys.end());
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
            if (keys.size() > std::numeric_limits<size_t>::max() / 2) {
                throw std::overflow_error("[VoxelOccupancyMap] table capacity overflow");
            }
            const size_t capacity = voxel_detail::next_power_of_two(std::max<size_t>(16, keys.size() * 2));
            const size_t mask = capacity - 1;
            auto& table = this->keys_[level];
            table.assign(capacity, VoxelConstants::invalid_coord);
            for (const uint64_t key : keys) {
                size_t slot = static_cast<size_t>(voxel_detail::hash_voxel_key(key)) & mask;
                while (table[slot] != VoxelConstants::invalid_coord && table[slot] != key) {
                    slot = (slot + 1) & mask;
                }
                table[slot] = key;
            }
            this->capacities_[level] = capacity;
            this->masks_[level] = mask;
        }
        this->target_size_ = cloud.size();
    }

    /// @brief Count query points that fall in an occupied voxel for each pose.
    /// @param queries Query points.
    /// @param transforms Candidate transforms from the query frame to the map frame.
    /// @param level Level index that selects the voxel size.
    /// @param counts One output count for each candidate.
    /// @param depends Events that must complete before the kernel starts.
    /// @return Events for kernel completion.
    sycl_utils::events score_async(const PointCloudShared& queries, const shared_vector<TransformMatrix>& transforms,
                                   size_t level, shared_vector<uint32_t>& counts,
                                   const std::vector<sycl::event>& depends = std::vector<sycl::event>()) const {
        if (level >= this->keys_.size()) {
            throw std::out_of_range("[VoxelOccupancyMap] level out of range");
        }
        if (transforms.size() != counts.size()) {
            throw std::invalid_argument("[VoxelOccupancyMap] output size mismatch");
        }
        if (counts.empty()) {
            return {};
        }
        const size_t group_size =
            std::max<size_t>(1, std::min<size_t>(64, this->queue_.get_work_group_size_for_parallel_reduction()));
        const size_t candidates = transforms.size();
        if (candidates > std::numeric_limits<size_t>::max() / group_size ||
            queries.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("[VoxelOccupancyMap] count or launch size overflow");
        }
        const size_t query_size = queries.size();
        const auto* query_ptr = queries.points_ptr();
        const auto* matrix_ptr = transforms.data();
        const auto* key_ptr = this->keys_[level].data();
        const uint64_t mask = this->masks_[level];
        const float voxel_size_inv = this->voxel_size_inv_[level];
        auto* count_ptr = counts.data();
        const bool search = query_size != 0 && this->capacities_[level] != 0;

        sycl_utils::events events;
        events += this->queue_.ptr->submit([&](sycl::handler& h) {
            h.depends_on(depends);
            sycl::local_accessor<uint32_t, 1> partial(sycl::range<1>(group_size), h);
            h.parallel_for(sycl::nd_range<1>(candidates * group_size, group_size), [=](sycl::nd_item<1> item) {
                const size_t candidate = item.get_group(0);
                const size_t lane = item.get_local_id(0);
                uint32_t local_count = 0;
                if (search) {
                    const auto transform_vec = eigen_utils::to_sycl_vec(matrix_ptr[candidate]);
                    for (size_t q = lane; q < query_size; q += group_size) {
                        PointType point;
                        transform::kernel::transform_point(query_ptr[q], point, transform_vec);
                        const uint64_t key = filter::kernel::compute_voxel_bit(point, voxel_size_inv);
                        if (key != VoxelConstants::invalid_coord) {
                            local_count += static_cast<uint32_t>(voxel_detail::is_voxel_occupied(key_ptr, mask, key));
                        }
                    }
                }
                partial[lane] = local_count;
                item.barrier(sycl::access::fence_space::local_space);
                for (size_t active = group_size; active > 1; active = (active + 1) / 2) {
                    const size_t half = (active + 1) / 2;
                    if (lane < active - half) {
                        partial[lane] += partial[lane + half];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (lane == 0) {
                    count_ptr[candidate] = partial[0];
                }
            });
        });
        return events;
    }

    /// @brief Get the voxel size of a level.
    float voxel_size(size_t level) const { return this->voxel_sizes_.at(level); }
    /// @brief Get the number of levels.
    size_t level_count() const { return this->voxel_sizes_.size(); }
    /// @brief Get the number of target points.
    size_t target_size() const { return this->target_size_; }

private:
    sycl_utils::DeviceQueue queue_;
    std::vector<float> voxel_sizes_;
    std::vector<float> voxel_size_inv_;
    std::vector<size_t> capacities_;
    std::vector<size_t> masks_;
    std::vector<shared_vector<uint64_t>> keys_;
    size_t target_size_ = 0;
};

}  // namespace sycl_points::algorithms::localization
