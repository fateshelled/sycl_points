#pragma once

#include <limits>
#include <optional>
#include <random>

#include "sycl_points/algorithms/common/filter_by_flags.hpp"
#include "sycl_points/algorithms/filter/preprocess_operator/angle_incidence_filter_operator.hpp"
#include "sycl_points/algorithms/filter/preprocess_operator/box_filter_operator.hpp"
#include "sycl_points/algorithms/filter/preprocess_operator/farthest_point_sampling_operator.hpp"
#include "sycl_points/algorithms/filter/preprocess_operator/random_sampling_operator.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace filter {

/// @brief Preprocessing filter for point cloud data
class PreprocessFilter {
public:
    using Ptr = std::shared_ptr<PreprocessFilter>;
    using InitializeFlagsFn = preprocess_operator::PreprocessOperatorBase::InitializeFlagsFn;
    using FilterByFlagsFn = preprocess_operator::PreprocessOperatorBase::FilterByFlagsFn;

    /// @brief Constructor
    /// @param queue SYCL queue
    PreprocessFilter(const sycl_utils::DeviceQueue& queue)
        : queue_(queue),
          filter_(std::make_shared<FilterByFlags>(this->queue_)),
          flags_(std::make_shared<shared_vector<uint8_t>>(*this->queue_.ptr)) {
        const auto initialize_flags_fn = this->make_initialize_flags_fn();
        const auto filter_by_flags_fn = this->make_filter_by_flags_fn();

        this->box_filter_op_.emplace(this->queue_, this->flags_, initialize_flags_fn, filter_by_flags_fn);
        this->random_sampling_op_.emplace(this->queue_, this->flags_, initialize_flags_fn, filter_by_flags_fn);
        this->farthest_point_sampling_op_.emplace(this->queue_, this->flags_, initialize_flags_fn, filter_by_flags_fn);
        this->angle_incidence_filter_op_.emplace(this->queue_, this->flags_, initialize_flags_fn, filter_by_flags_fn);
    }

    /// @brief Sets the seed for the random number generator
    /// @param seed Seed value to initialize the Mersenne Twister random generator
    void set_random_seed(uint_fast32_t seed) {
        this->random_sampling_op_->set_random_seed(seed);
        this->farthest_point_sampling_op_->set_random_seed(seed);
    }

    /// @brief L∞ distance (chebyshev distance) to the point cloud
    /// @param data Point cloud to be filtered (modified in-place)
    /// @param min_distance Minimum distance threshold (points closer than this are removed)
    /// @param max_distance Maximum distance threshold (points farther than this are removed)
    void box_filter(PointCloudShared& data, float min_distance = 1.0f,
                    float max_distance = std::numeric_limits<float>::max()) {
        this->box_filter_apply(data, data, min_distance, max_distance);
    }

    /// @brief L∞ distance (chebyshev distance) to the point cloud
    /// @param source Source point cloud to be filtered
    /// @param output Output point cloud
    /// @param min_distance Minimum distance threshold (points closer than this are removed)
    /// @param max_distance Maximum distance threshold (points farther than this are removed)
    void box_filter(const PointCloudShared& source, PointCloudShared& output, float min_distance = 1.0f,
                    float max_distance = std::numeric_limits<float>::max()) {
        this->box_filter_apply(source, output, min_distance, max_distance);
    }

    /// @brief Randomly samples a specified number of points from the point cloud
    /// @param data Point cloud to be sampled (modified in-place)
    /// @param sampling_num Number of points to retain after sampling
    void random_sampling(PointCloudShared& data, size_t sampling_num) {
        this->random_sampling_apply(data, data, sampling_num);
    }

    /// @brief Randomly samples a specified number of points from the point cloud
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    void random_sampling(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num) {
        this->random_sampling_apply(source, output, sampling_num);
    }

    /// @brief Farthest Point Sampling (FPS)
    /// @param source Source point cloud to be sampled (modified in-place)
    /// @param sampling_num Number of points to retain after sampling
    void farthest_point_sampling(PointCloudShared& data, size_t sampling_num) {
        this->farthest_point_sampling_apply(data, data, sampling_num);
    }

    /// @brief Farthest Point Sampling (FPS)
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    void farthest_point_sampling(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num) {
        this->farthest_point_sampling_apply(source, output, sampling_num);
    }

    /// @brief Filters points by incidence angle between point position and surface normal.
    /// @param data Point cloud to be filtered (modified in-place).
    /// @param min_angle Minimum allowable incidence angle in radians.
    /// @param max_angle Maximum allowable incidence angle in radians.
    /// @note Requires per-point normals or covariance matrices to compute normals; uses absolute incidence angle
    /// (normal direction ignored).
    void angle_incidence_filter(PointCloudShared& data, float min_angle, float max_angle) {
        this->angle_incidence_filter_apply(data, data, min_angle, max_angle);
    }

    /// @brief Filters points by incidence angle between point position and surface normal.
    /// @param source Source point cloud to be filtered.
    /// @param output Output point cloud.
    /// @param min_angle Minimum allowable incidence angle in radians.
    /// @param max_angle Maximum allowable incidence angle in radians.
    /// @note Requires per-point normals or covariance matrices to compute normals; uses absolute incidence angle
    /// (normal direction ignored).
    void angle_incidence_filter(const PointCloudShared& source, PointCloudShared& output, float min_angle,
                                float max_angle) {
        this->angle_incidence_filter_apply(source, output, min_angle, max_angle);
    }

private:
    InitializeFlagsFn make_initialize_flags_fn() {
        return
            [this](size_t data_size, uint8_t initial_flag) { return this->initialize_flags(data_size, initial_flag); };
    }

    FilterByFlagsFn make_filter_by_flags_fn() {
        return
            [this](const PointCloudShared& source, PointCloudShared& output) { this->filter_by_flags(source, output); };
    }

    sycl_utils::DeviceQueue queue_;
    FilterByFlags::Ptr filter_;
    shared_vector_ptr<uint8_t> flags_;
    std::optional<preprocess_operator::BoxFilterOperator> box_filter_op_;
    std::optional<preprocess_operator::RandomSamplingOperator> random_sampling_op_;
    std::optional<preprocess_operator::FarthestPointSamplingOperator> farthest_point_sampling_op_;
    std::optional<preprocess_operator::AngleIncidenceFilterOperator> angle_incidence_filter_op_;

    /// @brief Initializes the flags vector with a specified value
    /// @param data_size Size needed for the flags vector
    /// @param initial_flag Initial value to fill the flags with (INCLUDE_FLAG or REMOVE_FLAG)
    sycl_utils::events initialize_flags(size_t data_size, uint8_t initial_flag = INCLUDE_FLAG) {
        if (this->flags_->size() < data_size) {
            this->flags_->resize(data_size);
        }
        sycl_utils::events events;
        events += this->queue_.ptr->fill(this->flags_->data(), initial_flag, data_size);
        return events;
    }

    /// @brief Applies filtering based on the current flags
    /// @param data Point cloud to be filtered (modified in-place)
    void filter_by_flags(const PointCloudShared& source, PointCloudShared& output) {
        if (source.has_cov()) {
            this->filter_->filter_by_flags(*source.covs, *output.covs, *this->flags_);
        }
        if (source.has_normal()) {
            this->filter_->filter_by_flags(*source.normals, *output.normals, *this->flags_);
        }
        if (source.has_rgb()) {
            this->filter_->filter_by_flags(*source.rgb, *output.rgb, *this->flags_);
        }
        if (source.has_intensity()) {
            this->filter_->filter_by_flags(*source.intensities, *output.intensities, *this->flags_);
        }
        if (source.has_timestamps()) {
            this->filter_->filter_by_flags(*source.timestamp_offsets, *output.timestamp_offsets, *this->flags_);
        }
        this->filter_->filter_by_flags(*source.points, *output.points, *this->flags_);
    }

    /// @brief L∞ distance (chebyshev distance) to the point cloud
    /// @param source Source point cloud to be filtered
    /// @param output Output point cloud
    /// @param min_distance Minimum distance threshold (points closer than this are removed)
    /// @param max_distance Maximum distance threshold (points farther than this are removed)
    void box_filter_apply(const PointCloudShared& source, PointCloudShared& output, float min_distance = 1.0f,
                          float max_distance = std::numeric_limits<float>::max()) {
        this->box_filter_op_->apply(source, output, min_distance, max_distance);
    }

    /// @brief Randomly samples a specified number of points from the point cloud
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    void random_sampling_apply(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num) {
        this->random_sampling_op_->apply(source, output, sampling_num);
    }

    /// @brief Farthest Point Sampling (FPS)
    /// @param source Source point cloud to be sampled
    /// @param output Output point cloud
    /// @param sampling_num Number of points to retain after sampling
    void farthest_point_sampling_apply(const PointCloudShared& source, PointCloudShared& output, size_t sampling_num) {
        this->farthest_point_sampling_op_->apply(source, output, sampling_num);
    }

    /// @brief Removes points whose (absolute) incidence angle with the surface normal is outside [min_angle,
    /// max_angle].
    /// @param source Source point cloud to be sampled.
    /// @param output Output point cloud.
    /// @param min_angle Minimum allowable incidence angle in radians.
    /// @param max_angle Maximum allowable incidence angle in radians.
    void angle_incidence_filter_apply(const PointCloudShared& source, PointCloudShared& output, float min_angle,
                                      float max_angle) {
        this->angle_incidence_filter_op_->apply(source, output, min_angle, max_angle);
    }
};

}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
