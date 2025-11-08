#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {
namespace algorithms {
namespace mapping {

/// @brief Header-only occupancy grid map that accumulates log-odds values per voxel.
/// @note The implementation runs on the host and uses shared USM containers for IO.
class OccupancyGridMap {
public:
    using Ptr = std::shared_ptr<OccupancyGridMap>;

    /// @brief Construct the occupancy grid map.
    /// @param voxel_size Edge length of a voxel in meters.
    explicit OccupancyGridMap(const float voxel_size) {
        this->set_voxel_size(voxel_size);
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

        const bool has_rgb = cloud.has_rgb();
        const bool has_intensity = cloud.has_intensity();
        this->has_rgb_data_ = this->has_rgb_data_ || has_rgb;
        this->has_intensity_data_ = this->has_intensity_data_ || has_intensity;

        std::unordered_set<VoxelKey, VoxelKeyHash> updated_voxels;
        updated_voxels.reserve(cloud.points->size());

        // Update the occupied voxels using a simple inverse sensor model.
        for (size_t i = 0; i < cloud.points->size(); ++i) {
            const PointType& raw_point = (*cloud.points)[i];
            const Eigen::Vector3f local_point = raw_point.head<3>();
            const Eigen::Vector3f world_point = sensor_pose * local_point;

            const VoxelKey key = this->compute_key(world_point);
            VoxelData& data = this->voxels_[key];

            updated_voxels.insert(key);

            data.hit_count += 1U;
            data.log_odds = this->clamp_log_odds(data.log_odds + this->log_odds_hit_);

            if (data.hit_count == 1U) {
                data.centroid = world_point;
            } else {
                // Update centroid using incremental average to keep points stable.
                const float ratio = 1.0f / static_cast<float>(data.hit_count);
                data.centroid = data.centroid + (world_point - data.centroid) * ratio;
            }

            if (has_rgb) {
                const auto& rgb = (*cloud.rgb)[i];
                data.rgb_sum += rgb;
                data.rgb_count += 1U;
            }

            if (has_intensity) {
                data.intensity_sum += (*cloud.intensities)[i];
                data.intensity_count += 1U;
            }
        }

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

    static int32_t fast_floor(const float value) {
        // Bias the input slightly to maintain stable indices near voxel boundaries.
        const float bias = (value >= 0.0f) ? 1e-6f : -1e-6f;
        return static_cast<int32_t>(std::floor(value + bias));
    }
};

}  // namespace mapping
}  // namespace algorithms
}  // namespace sycl_points

