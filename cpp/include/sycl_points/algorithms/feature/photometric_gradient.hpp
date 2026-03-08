#pragma once

#include <memory>
#include <vector>

#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {

namespace algorithms {

inline constexpr float CHOLESKY_REGULARIZATION = 1e-6f;

namespace color_gradient {

namespace kernel {

SYCL_EXTERNAL inline void compute_gradient(ColorGradient& ret, const PointType* points, const RGBType* colors,
                                           const int32_t* index_ptr, size_t k, size_t i) {
    const auto p0 = points[i];
    // The w-component of a point stores the number of points in the voxel.
    // It should not be included in the position calculation.
    const Eigen::Vector3f p0_3d = p0.head<3>();
    const auto c0 = colors[i];

    Eigen::Matrix3f A = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f B = Eigen::Matrix3f::Zero();
    for (size_t j = 0; j < k; ++j) {
        const auto idx = index_ptr[i * k + j];
        const auto pj = points[idx];
        const auto cj = colors[idx];

        const Eigen::Vector3f dp = pj.head<3>() - p0_3d;
        const Eigen::Vector3f dc(cj[0] - c0[0], cj[1] - c0[1], cj[2] - c0[2]);

        eigen_utils::add_inplace<3, 3>(A, eigen_utils::outer<3>(dp, dp));
        eigen_utils::add_inplace<3, 3>(B, eigen_utils::outer<3>(dp, dc));
    }

    // Add a small identity matrix for numerical stability
    eigen_utils::add_inplace<3, 3>(A, Eigen::Matrix3f::Identity() * CHOLESKY_REGULARIZATION);

    const Eigen::Matrix3f L = eigen_utils::cholesky_3x3(A);
    for (size_t c = 0; c < 3; ++c) {
        const Eigen::Vector3f b = B.col(c);
        const Eigen::Vector3f g = eigen_utils::solve_cholesky_3x3(L, b);
        ret.row(c) = g;
    }
}

}  // namespace kernel

inline sycl_utils::events compute_color_gradients_async(
    const PointCloudShared& cloud, const knn::KNNResult& neighbors,
    const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    const size_t N = cloud.size();
    if (!cloud.has_rgb()) {
        throw std::runtime_error("[compute_color_gradients_async] RGB field not found");
    }

    if (cloud.color_gradients->size() != N) {
        cloud.resize_color_gradients(N);
    }
    if (N == 0) return sycl_utils::events();

    const size_t work_group_size = cloud.queue.get_work_group_size();
    const size_t global_size = cloud.queue.get_global_size(N);

    const auto indices = neighbors.indices;
    const size_t k = neighbors.k;

    sycl_utils::events events;
    events += cloud.queue.ptr->submit([&, indices, k](sycl::handler& h) {
        h.depends_on(depends);
        const auto pt_ptr = cloud.points_ptr();
        const auto color_ptr = cloud.rgb_ptr();
        const auto grad_ptr = cloud.color_gradients_ptr();
        const auto index_ptr = indices->data();
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t idx = item.get_global_id(0);
            if (idx >= N) return;

            kernel::compute_gradient(grad_ptr[idx], pt_ptr, color_ptr, index_ptr, k, idx);
        });
    });
    return events;
}

inline sycl_utils::events compute_color_gradients_async(
    const PointCloudShared& cloud, const knn::KNNBase& knn, size_t k_correspondences,
    const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    auto neighbors = std::make_shared<knn::KNNResult>();
    auto knn_events = knn.knn_search_async(cloud, k_correspondences, *neighbors, depends);
    auto gradient_events = compute_color_gradients_async(cloud, *neighbors, knn_events.evs);
    gradient_events += knn_events;
    gradient_events.add_resource(neighbors);
    return gradient_events;
}

}  // namespace color_gradient

namespace intensity_gradient {

namespace kernel {

SYCL_EXTERNAL inline void compute_gradient(IntensityGradient& ret, const PointType* points, const float* intensities,
                                           const int32_t* index_ptr, size_t k, size_t i) {
    const auto p0 = points[i];
    const Eigen::Vector3f p0_3d = p0.head<3>();
    const float intensity0 = intensities[i];

    Eigen::Matrix3f A = Eigen::Matrix3f::Zero();
    Eigen::Vector3f b = Eigen::Vector3f::Zero();

    for (size_t j = 0; j < k; ++j) {
        const auto idx = index_ptr[i * k + j];
        const auto pj = points[idx];
        const float intensity_j = intensities[idx];

        const Eigen::Vector3f dp = pj.head<3>() - p0_3d;
        const float di = intensity_j - intensity0;

        eigen_utils::add_inplace<3, 3>(A, eigen_utils::outer<3>(dp, dp));
        b += dp * di;
    }

    eigen_utils::add_inplace<3, 3>(A, Eigen::Matrix3f::Identity() * CHOLESKY_REGULARIZATION);

    const Eigen::Matrix3f L = eigen_utils::cholesky_3x3(A);
    const Eigen::Vector3f gradient = eigen_utils::solve_cholesky_3x3(L, b);
    ret = gradient;
}

}  // namespace kernel

inline sycl_utils::events compute_intensity_gradients_async(
    const PointCloudShared& cloud, const knn::KNNResult& neighbors,
    const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    const size_t N = cloud.size();
    if (!cloud.has_intensity()) {
        throw std::runtime_error("[compute_intensity_gradients_async] Intensity field not found");
    }

    if (cloud.intensity_gradients->size() != N) {
        cloud.resize_intensity_gradients(N);
    }
    if (N == 0) return sycl_utils::events();

    const size_t work_group_size = cloud.queue.get_work_group_size();
    const size_t global_size = cloud.queue.get_global_size(N);

    const auto indices = neighbors.indices;
    const size_t k = neighbors.k;

    sycl_utils::events events;
    events += cloud.queue.ptr->submit([&, indices, k](sycl::handler& h) {
        h.depends_on(depends);
        const auto pt_ptr = cloud.points_ptr();
        const auto intensity_ptr = cloud.intensities_ptr();
        const auto grad_ptr = cloud.intensity_gradients_ptr();
        const auto index_ptr = indices->data();
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t idx = item.get_global_id(0);
            if (idx >= N) return;

            kernel::compute_gradient(grad_ptr[idx], pt_ptr, intensity_ptr, index_ptr, k, idx);
        });
    });
    return events;
}

inline sycl_utils::events compute_intensity_gradients_async(
    const PointCloudShared& cloud, const knn::KNNBase& knn, size_t k_correspondences,
    const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    auto neighbors = std::make_shared<knn::KNNResult>();
    auto knn_events = knn.knn_search_async(cloud, k_correspondences, *neighbors, depends);
    auto gradient_events = compute_intensity_gradients_async(cloud, *neighbors, knn_events.evs);
    gradient_events += knn_events;
    gradient_events.add_resource(neighbors);
    return gradient_events;
}

}  // namespace intensity_gradient

namespace intensity_z_score {

namespace kernel {

SYCL_EXTERNAL inline float compute_z_score(const float* intensities, const int32_t* index_ptr, size_t k, size_t i) {
    const float intensity = intensities[i];

    float neighbor_sum = 0.0f;
    float neighbor_sum_sq = 0.0f;
    size_t valid_neighbors = 0;
    for (size_t j = 0; j < k; ++j) {
        const int32_t idx = index_ptr[i * k + j];
        if (idx < 0) continue;
        const float neighbor_intensity = intensities[idx];
        neighbor_sum += neighbor_intensity;
        neighbor_sum_sq += neighbor_intensity * neighbor_intensity;
        ++valid_neighbors;
    }

    if (valid_neighbors == 0) {
        return 0.0f;
    }

    const float inv_count = 1.0f / static_cast<float>(valid_neighbors);
    const float neighbor_mean = neighbor_sum * inv_count;
    const float neighbor_variance = sycl::max(0.0f, neighbor_sum_sq * inv_count - neighbor_mean * neighbor_mean);
    const float neighbor_stddev = sycl::sqrt(neighbor_variance);

    return (intensity - neighbor_mean) / (neighbor_stddev + 1e-6f);
}

}  // namespace kernel

/// @brief Computes per-point intensity z-score from neighborhood intensity statistics.
/// The output value is defined as (point_intensity - mean(neighbor_intensity)) / stddev(neighbor_intensity).
class IntensityZScoreCalculator {
public:
    using Ptr = std::shared_ptr<IntensityZScoreCalculator>;

    explicit IntensityZScoreCalculator(const sycl_utils::DeviceQueue& queue)
        : queue_(queue), intensity_z_scores_(std::make_shared<shared_vector<float>>(*queue.ptr)) {}

    sycl_utils::events compute_async(const PointCloudShared& cloud, const knn::KNNResult& neighbors,
                                     const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
        this->validate_inputs(cloud, neighbors);

        const size_t N = cloud.size();
        if (this->intensity_z_scores_->size() != N) {
            this->intensity_z_scores_->resize(N);
        }
        if (N == 0) {
            return sycl_utils::events();
        }

        const size_t work_group_size = this->queue_.get_work_group_size();
        const size_t global_size = this->queue_.get_global_size(N);
        const size_t k = neighbors.k;
        const auto indices = neighbors.indices;

        sycl_utils::events events;
        events += this->queue_.ptr->submit([&, k, indices](sycl::handler& h) {
            h.depends_on(depends);

            const auto intensity_ptr = cloud.intensities_ptr();
            const auto output_ptr = this->intensity_z_scores_->data();
            const auto index_ptr = indices->data();

            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= N) return;

                output_ptr[idx] = kernel::compute_z_score(intensity_ptr, index_ptr, k, idx);
            });
        });
        return events;
    }

    sycl_utils::events compute_async(const PointCloudShared& cloud, const knn::KNNBase& knn, size_t k_correspondences,
                                     const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
        auto neighbors = std::make_shared<knn::KNNResult>();
        auto knn_events = knn.knn_search_async(cloud, k_correspondences, *neighbors, depends);
        auto diff_events = this->compute_async(cloud, *neighbors, knn_events.evs);
        diff_events += knn_events;
        diff_events.add_resource(neighbors);
        return diff_events;
    }

    shared_vector_ptr<float> intensity_z_scores() const { return this->intensity_z_scores_; }

private:
    sycl_utils::DeviceQueue queue_;
    shared_vector_ptr<float> intensity_z_scores_;

    static void validate_inputs(const PointCloudShared& cloud, const knn::KNNResult& neighbors) {
        if (!cloud.has_intensity()) {
            throw std::runtime_error("[IntensityZScoreCalculator] Intensity field not found");
        }
        if (!neighbors.indices) {
            throw std::runtime_error("[IntensityZScoreCalculator] Neighbor indices are not allocated");
        }
        if (neighbors.query_size != cloud.size()) {
            throw std::runtime_error("[IntensityZScoreCalculator] Neighbor query size must match cloud size");
        }
    }
};

}  // namespace intensity_z_score

}  // namespace algorithms

}  // namespace sycl_points
