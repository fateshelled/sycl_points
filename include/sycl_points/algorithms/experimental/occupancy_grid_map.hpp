#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sycl/sycl.hpp>

#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {
namespace algorithms {
namespace mapping {

/// @brief Header-only occupancy grid map that accumulates log-odds values per voxel.
/// @note The heavy per-point computations are parallelized on the device using the provided SYCL queue.
class OccupancyGridMap {
public:
    using Ptr = std::shared_ptr<OccupancyGridMap>;

    /// @brief Construct the occupancy grid map.
    /// @param voxel_size Edge length of a voxel in meters.
    OccupancyGridMap(const sycl_utils::DeviceQueue& queue, const float voxel_size) : queue_(queue) {
        this->set_voxel_size(voxel_size);
        this->device_contributions_ =
            std::make_shared<DeviceContributionBuffer>(0, *this->queue_.ptr);
    }

    /// @brief Reset the map data.
    void clear() {
        this->voxels_.clear();
        this->has_rgb_data_ = false;
        this->has_intensity_data_ = false;
    }

    /// @brief Set the voxel size.
    /// @param voxel_size Edge length in meters.
    void set_voxel_size(const float voxel_size) {
        if (!(voxel_size > 0.0f)) {
            throw std::invalid_argument("voxel_size must be positive.");
        }
        this->voxel_size_ = voxel_size;
        this->inv_voxel_size_ = 1.0f / voxel_size;
    }

    /// @brief Get the voxel size.
    float voxel_size() const { return this->voxel_size_; }

    /// @brief Query the occupancy probability at the specified position.
    float voxel_probability(const Eigen::Vector3f& position) const {
        const VoxelKey key = this->compute_key(position);
        const auto it = this->voxels_.find(key);
        if (it == this->voxels_.end()) {
            return 0.5f;
        }
        return this->log_odds_to_probability(it->second.log_odds);
    }

    /// @brief Configure the log-odds increment applied on hits.
    void set_log_odds_hit(const float value) { this->log_odds_hit_ = value; }

    /// @brief Configure the log-odds decrement applied on misses.
    void set_log_odds_miss(const float value) { this->log_odds_miss_ = value; }

    /// @brief Set the visibility decay range in meters.
    void set_visibility_decay_range(const float distance) {
        if (!(distance >= 0.0f)) {
            throw std::invalid_argument("distance must be non-negative.");
        }
        this->visibility_decay_range_ = distance;
    }

    /// @brief Get the visibility decay range in meters.
    float visibility_decay_range() const { return this->visibility_decay_range_; }

    /// @brief Configure the minimum and maximum allowed log-odds.
    void set_log_odds_limits(const float minimum, const float maximum) {
        if (minimum > maximum) {
            throw std::invalid_argument("minimum must not exceed maximum.");
        }
        this->min_log_odds_ = minimum;
        this->max_log_odds_ = maximum;
    }

    /// @brief Configure the occupancy probability threshold.
    void set_occupancy_threshold(const float probability) {
        if (!(probability > 0.0f) || !(probability < 1.0f)) {
            throw std::invalid_argument("probability must be between 0 and 1.");
        }
        this->occupancy_threshold_log_odds_ = this->probability_to_log_odds(probability);
    }

    /// @brief Insert a point cloud captured at the given pose.
    /// @param cloud Point cloud in the sensor frame.
    /// @param sensor_pose Sensor pose expressed in the map frame.
    void add_point_cloud(const PointCloudShared& cloud, const Eigen::Isometry3f& sensor_pose) {
        if (!cloud.points || cloud.points->empty()) {
            return;
        }

        if (cloud.queue.ptr.get() != this->queue_.ptr.get()) {
            throw std::invalid_argument("Point cloud queue does not match occupancy grid map queue.");
        }

        const bool has_rgb = cloud.has_rgb();
        const bool has_intensity = cloud.has_intensity();
        this->has_rgb_data_ = this->has_rgb_data_ || has_rgb;
        this->has_intensity_data_ = this->has_intensity_data_ || has_intensity;

        const size_t num_points = cloud.points->size();
        if (this->device_contributions_->size() < num_points) {
            this->device_contributions_->resize(num_points);
        }

        this->queue_.set_accessed_by_device(this->device_contributions_->data(), num_points);
        this->queue_.set_accessed_by_device(cloud.points->data(), num_points);
        if (has_rgb) {
            this->queue_.set_accessed_by_device(cloud.rgb->data(), num_points);
        }
        if (has_intensity) {
            this->queue_.set_accessed_by_device(cloud.intensities->data(), num_points);
        }

        const DeviceTransform device_transform = this->make_device_transform(sensor_pose.matrix());
        const float inv_voxel_size = this->inv_voxel_size_;

        const size_t work_group_size = this->queue_.get_work_group_size();
        const size_t global_size = this->queue_.get_global_size(num_points);

        auto event = this->queue_.ptr->submit([&](sycl::handler& handler) {
            const auto contributions_ptr = this->device_contributions_->data();
            const auto points_ptr = cloud.points->data();
            const auto rgb_ptr = has_rgb ? cloud.rgb->data() : static_cast<RGBType*>(nullptr);
            const auto intensity_ptr = has_intensity ? cloud.intensities->data() : static_cast<float*>(nullptr);

            // Transform points to the world frame and compute their voxel coordinates in parallel.
            handler.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= num_points) {
                    return;
                }

                DeviceContribution contribution;
                contribution.valid = 0U;
                contribution.has_rgb = 0U;
                contribution.has_intensity = 0U;

                const PointType point = points_ptr[idx];
                const float local_x = point.x();
                const float local_y = point.y();
                const float local_z = point.z();

                if (!sycl::isfinite(local_x) || !sycl::isfinite(local_y) || !sycl::isfinite(local_z)) {
                    contributions_ptr[idx] = contribution;
                    return;
                }

                const float world_x = device_transform.m[0] * local_x + device_transform.m[1] * local_y +
                                      device_transform.m[2] * local_z + device_transform.m[3];
                const float world_y = device_transform.m[4] * local_x + device_transform.m[5] * local_y +
                                      device_transform.m[6] * local_z + device_transform.m[7];
                const float world_z = device_transform.m[8] * local_x + device_transform.m[9] * local_y +
                                      device_transform.m[10] * local_z + device_transform.m[11];

                const float scaled_x = world_x * inv_voxel_size;
                const float scaled_y = world_y * inv_voxel_size;
                const float scaled_z = world_z * inv_voxel_size;

                const int32_t voxel_x = device_fast_floor(scaled_x);
                const int32_t voxel_y = device_fast_floor(scaled_y);
                const int32_t voxel_z = device_fast_floor(scaled_z);
                contribution.voxel_x = voxel_x;
                contribution.voxel_y = voxel_y;
                contribution.voxel_z = voxel_z;
                contribution.world_x = world_x;
                contribution.world_y = world_y;
                contribution.world_z = world_z;

                if (rgb_ptr) {
                    const RGBType rgb = rgb_ptr[idx];
                    contribution.rgb_x = rgb.x();
                    contribution.rgb_y = rgb.y();
                    contribution.rgb_z = rgb.z();
                    contribution.rgb_w = rgb.w();
                    contribution.has_rgb = 1U;
                }

                if (intensity_ptr) {
                    contribution.intensity = intensity_ptr[idx];
                    contribution.has_intensity = 1U;
                }

                contribution.valid = 1U;
                contributions_ptr[idx] = contribution;
            });
        });
        event.wait();

        this->queue_.clear_accessed_by_device(this->device_contributions_->data(), num_points);
        this->queue_.clear_accessed_by_device(cloud.points->data(), num_points);
        if (has_rgb) {
            this->queue_.clear_accessed_by_device(cloud.rgb->data(), num_points);
        }
        if (has_intensity) {
            this->queue_.clear_accessed_by_device(cloud.intensities->data(), num_points);
        }

        this->queue_.set_accessed_by_host(this->device_contributions_->data(), num_points);

        std::unordered_set<VoxelKey, VoxelKeyHash> updated_voxels;
        updated_voxels.reserve(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            const DeviceContribution& contribution = (*this->device_contributions_)[i];
            if (contribution.valid == 0U) {
                continue;
            }

            const VoxelKey key{contribution.voxel_x, contribution.voxel_y, contribution.voxel_z};
            VoxelData& data = this->voxels_[key];

            updated_voxels.insert(key);

            data.hit_count += 1U;
            data.log_odds = this->clamp_log_odds(data.log_odds + this->log_odds_hit_);

            if (data.hit_count == 1U) {
                data.centroid = Eigen::Vector3f(contribution.world_x, contribution.world_y, contribution.world_z);
            } else {
                // Update centroid using incremental average to keep points stable.
                const float ratio = 1.0f / static_cast<float>(data.hit_count);
                const Eigen::Vector3f world_point(contribution.world_x, contribution.world_y, contribution.world_z);
                data.centroid = data.centroid + (world_point - data.centroid) * ratio;
            }

            if (contribution.has_rgb) {
                data.rgb_sum.x() += contribution.rgb_x;
                data.rgb_sum.y() += contribution.rgb_y;
                data.rgb_sum.z() += contribution.rgb_z;
                data.rgb_sum.w() += contribution.rgb_w;
                data.rgb_count += 1U;
            }

            if (contribution.has_intensity) {
                data.intensity_sum += contribution.intensity;
                data.intensity_count += 1U;
            }
        }

        this->queue_.clear_accessed_by_host(this->device_contributions_->data(), num_points);

        // Apply a uniform miss update to voxels within sensor range that were not hit.
        if (this->log_odds_miss_ != 0.0f) {
            this->apply_visibility_decay(sensor_pose, updated_voxels);
        }
    }

    /// @brief Extract occupied voxels that are visible from the sensor pose.
    /// @param result Output point cloud in the map frame.
    /// @param sensor_pose Sensor pose expressed in the map frame.
    /// @param max_distance Maximum visibility distance in meters.
    void downsampling(PointCloudShared& result, const Eigen::Isometry3f& sensor_pose,
                      const float max_distance = 100.0f) const {
        result.resize_points(0);
        result.resize_rgb(0);
        result.resize_intensities(0);

        if (this->voxels_.empty()) {
            return;
        }

        std::vector<const VoxelData*> visible_voxels;
        visible_voxels.reserve(this->voxels_.size());

        const Eigen::Vector3f sensor_position = sensor_pose.translation();
        const float max_distance_sq = max_distance * max_distance;

        for (const auto& entry : this->voxels_) {
            const VoxelData& data = entry.second;
            if (data.log_odds < this->occupancy_threshold_log_odds_) {
                continue;
            }

            const Eigen::Vector3f delta = data.centroid - sensor_position;
            if (delta.squaredNorm() > max_distance_sq) {
                continue;
            }

            visible_voxels.push_back(&data);
        }

        if (visible_voxels.empty()) {
            return;
        }

        result.resize_points(visible_voxels.size());
        if (this->has_rgb_data_) {
            result.resize_rgb(visible_voxels.size());
        }
        if (this->has_intensity_data_) {
            result.resize_intensities(visible_voxels.size());
        }

        for (size_t i = 0; i < visible_voxels.size(); ++i) {
            const VoxelData& data = *visible_voxels[i];
            (*result.points)[i] = PointType(data.centroid.x(), data.centroid.y(), data.centroid.z(), 1.0f);

            if (this->has_rgb_data_) {
                Eigen::Vector4f rgb = Eigen::Vector4f::Zero();
                if (data.rgb_count > 0U) {
                    rgb = data.rgb_sum / static_cast<float>(data.rgb_count);
                }
                (*result.rgb)[i] = rgb;
            }

            if (this->has_intensity_data_) {
                float intensity = 0.0f;
                if (data.intensity_count > 0U) {
                    intensity = data.intensity_sum / static_cast<float>(data.intensity_count);
                }
                (*result.intensities)[i] = intensity;
            }
        }
    }

private:
    struct VoxelKey {
        int32_t x = 0;
        int32_t y = 0;
        int32_t z = 0;

        bool operator==(const VoxelKey& other) const {
            return this->x == other.x && this->y == other.y && this->z == other.z;
        }
    };

    struct VoxelKeyHash {
        size_t operator()(const VoxelKey& key) const noexcept {
            // Combine the hash of the coordinates using large primes to reduce collisions.
            constexpr size_t kPrime1 = 73856093;
            constexpr size_t kPrime2 = 19349663;
            constexpr size_t kPrime3 = 83492791;
            return static_cast<size_t>(key.x) * kPrime1 ^ static_cast<size_t>(key.y) * kPrime2 ^
                   static_cast<size_t>(key.z) * kPrime3;
        }
    };

    struct VoxelData {
        float log_odds = 0.0f;
        uint32_t hit_count = 0U;
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        Eigen::Vector4f rgb_sum = Eigen::Vector4f::Zero();
        uint32_t rgb_count = 0U;
        float intensity_sum = 0.0f;
        uint32_t intensity_count = 0U;
    };

    using VoxelMap = std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash>;

    struct DeviceContribution {
        // Aggregated data for a single point after transforming it to the map frame.
        int32_t voxel_x = 0;
        int32_t voxel_y = 0;
        int32_t voxel_z = 0;
        float world_x = 0.0f;
        float world_y = 0.0f;
        float world_z = 0.0f;
        float rgb_x = 0.0f;
        float rgb_y = 0.0f;
        float rgb_z = 0.0f;
        float rgb_w = 0.0f;
        float intensity = 0.0f;
        uint8_t has_rgb = 0U;
        uint8_t has_intensity = 0U;
        uint8_t valid = 0U;
        uint8_t padding = 0U;
    };

    using DeviceContributionBuffer = shared_vector<DeviceContribution>;

    struct DeviceTransform {
        // Row-major 3x4 matrix extracted from the homogeneous sensor pose.
        std::array<float, 12> m = {0.0f};
    };

    static float probability_to_log_odds(const float probability) {
        return std::log(probability / (1.0f - probability));
    }

    static float log_odds_to_probability(const float log_odds) {
        return 1.0f / (1.0f + std::exp(-log_odds));
    }

    float clamp_log_odds(const float value) const {
        return std::min(std::max(value, this->min_log_odds_), this->max_log_odds_);
    }

    VoxelKey compute_key(const Eigen::Vector3f& point) const {
        const Eigen::Vector3f scaled = point * this->inv_voxel_size_;
        return VoxelKey{this->fast_floor(scaled.x()), this->fast_floor(scaled.y()), this->fast_floor(scaled.z())};
    }

    void apply_visibility_decay(const Eigen::Isometry3f& sensor_pose,
                                const std::unordered_set<VoxelKey, VoxelKeyHash>& updated_voxels) {
        const float max_distance_sq = this->visibility_decay_range_ * this->visibility_decay_range_;
        const Eigen::Vector3f sensor_position = sensor_pose.translation();

        for (auto& entry : this->voxels_) {
            const VoxelKey& key = entry.first;
            VoxelData& data = entry.second;
            if (updated_voxels.find(key) != updated_voxels.end()) {
                continue;
            }
            const Eigen::Vector3f delta = data.centroid - sensor_position;
            if (delta.squaredNorm() > max_distance_sq) {
                continue;
            }

            data.log_odds = this->clamp_log_odds(data.log_odds + this->log_odds_miss_);
        }
    }

    float voxel_size_ = 0.1f;
    float inv_voxel_size_ = 10.0f;
    float log_odds_hit_ = 0.85f;
    float log_odds_miss_ = -0.4f;
    float min_log_odds_ = -4.0f;
    float max_log_odds_ = 4.0f;
    float occupancy_threshold_log_odds_ = probability_to_log_odds(0.5f);
    float visibility_decay_range_ = 30.0f;
    bool has_rgb_data_ = false;
    bool has_intensity_data_ = false;
    VoxelMap voxels_;
    sycl_utils::DeviceQueue queue_;
    std::shared_ptr<DeviceContributionBuffer> device_contributions_ = nullptr;

    static int32_t fast_floor(const float value) {
        // Bias the input slightly to maintain stable indices near voxel boundaries.
        const float bias = (value >= 0.0f) ? 1e-6f : -1e-6f;
        return static_cast<int32_t>(std::floor(value + bias));
    }

    static int32_t device_fast_floor(const float value) {
        const float bias = (value >= 0.0f) ? 1e-6f : -1e-6f;
        return static_cast<int32_t>(sycl::floor(value + bias));
    }

    static DeviceTransform make_device_transform(const Eigen::Matrix4f& matrix) {
        DeviceTransform transform;
        transform.m = {matrix(0, 0), matrix(0, 1), matrix(0, 2), matrix(0, 3), matrix(1, 0), matrix(1, 1),
                       matrix(1, 2), matrix(1, 3), matrix(2, 0), matrix(2, 1), matrix(2, 2), matrix(2, 3)};
        return transform;
    }
};

}  // namespace mapping
}  // namespace algorithms
}  // namespace sycl_points

