#pragma once

#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/algorithms/robust/robust.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace covariance {

namespace kernel {

SYCL_EXTERNAL inline void compute_covariance(Covariance& ret, const PointType* point_ptr,
                                             const size_t k_correspondences, const int32_t* index_ptr, const size_t i,
                                             size_t min_num_correspondences = 4) {
    ret.setZero();
    Eigen::Vector3f sum_points = Eigen::Vector3f::Zero();
    Eigen::Matrix3f sum_outer = Eigen::Matrix3f::Zero();

    size_t correspondences = 0;
    for (size_t j = 0; j < k_correspondences; ++j) {
        const int32_t idx = index_ptr[i * k_correspondences + j];
        if (idx < 0) continue;

        const auto pt = point_ptr[idx].head<3>();
        eigen_utils::add_inplace<3, 1>(sum_points, pt);

        const auto outer = eigen_utils::outer<3>(pt, pt);
        eigen_utils::add_inplace<3, 3>(sum_outer, outer);
        ++correspondences;
    }

    if (correspondences < min_num_correspondences) {
        ret(0, 0) = 1.0f;
        ret(1, 1) = 1.0f;
        ret(2, 2) = 1.0f;
        return;
    }

    const Eigen::Vector3f mean = eigen_utils::multiply<3>(sum_points, 1.0f / correspondences);
    ret.block<3, 3>(0, 0) = eigen_utils::ensure_symmetric<3>(eigen_utils::subtract<3, 3>(
        eigen_utils::multiply<3, 3>(sum_outer, 1.0f / correspondences), eigen_utils::outer<3>(mean, mean)));
}

SYCL_EXTERNAL inline void compute_normal_from_covariance(const PointType& point, const Covariance& cov,
                                                         Normal& normal) {
    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;
    eigen_utils::symmetric_eigen_decomposition_3x3(cov.block<3, 3>(0, 0), eigenvalues, eigenvectors);
    Eigen::Vector3f normal3 = eigenvectors.col(0);
    if (eigen_utils::dot<3>(normal3, {point[0], point[1], point[2]}) <= 1.0) {
        normal.x() = normal3.x();
        normal.y() = normal3.y();
        normal.z() = normal3.z();
        normal.w() = 0.0f;
    } else {
        normal.x() = -normal3.x();
        normal.y() = -normal3.y();
        normal.z() = -normal3.z();
        normal.w() = 0.0f;
    }
}

SYCL_EXTERNAL inline void update_covariance_plane(Covariance& cov) {
    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;
    eigen_utils::symmetric_eigen_decomposition_3x3(cov.block<3, 3>(0, 0), eigenvalues, eigenvectors);
    const auto diag = eigen_utils::as_diagonal<3>({1e-3f, 1.0f, 1.0f});
    cov.block<3, 3>(0, 0) = eigen_utils::multiply<3, 3, 3>(eigen_utils::multiply<3, 3, 3>(eigenvectors, diag),
                                                           eigen_utils::transpose<3, 3>(eigenvectors));
}

SYCL_EXTERNAL inline void normalize_covariance(Covariance& cov) {
    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;

    // Multiplied by 1e3f for numerical stability
    eigen_utils::symmetric_eigen_decomposition_3x3(eigen_utils::multiply<3, 3>(cov.block<3, 3>(0, 0), 1e3f),
                                                   eigenvalues, eigenvectors);
    const float max_eigenvalue = eigenvalues(2);
    if (max_eigenvalue < std::numeric_limits<float>::min()) {
        cov.block<3, 3>(0, 0).setIdentity();
        return;
    }
    eigenvalues(0) = std::clamp(eigenvalues(0) / max_eigenvalue, 1e-3f, 1.0f);
    eigenvalues(1) = std::clamp(eigenvalues(1) / max_eigenvalue, 1e-3f, 1.0f);
    eigenvalues(2) = 1.0f;

    const auto diag = eigen_utils::as_diagonal<3>(eigenvalues);
    cov.block<3, 3>(0, 0) = eigen_utils::multiply<3, 3, 3>(eigen_utils::multiply<3, 3, 3>(eigenvectors, diag),
                                                           eigen_utils::transpose<3, 3>(eigenvectors));
}

SYCL_EXTERNAL inline bool compute_covariance_weighted(Covariance& ret, PointType& mean, const PointType* point_ptr,
                                                      const size_t k_correspondences, const int32_t* index_ptr,
                                                      const float* weights, const size_t i,
                                                      size_t min_num_correspondences = 4) {
    ret.setZero();
    Eigen::Vector3f sum_points = Eigen::Vector3f::Zero();
    Eigen::Matrix3f sum_outer = Eigen::Matrix3f::Zero();

    size_t correspondences = 0;
    float total_weight = 0.0f;
    for (size_t j = 0; j < k_correspondences; ++j) {
        const int32_t idx = index_ptr[i * k_correspondences + j];
        if (idx < 0) continue;

        const auto pt = point_ptr[idx].head<3>();
        eigen_utils::add_inplace<3, 1>(sum_points, eigen_utils::multiply<3>(pt, weights[j]));

        const auto outer = eigen_utils::outer<3>(pt, pt);
        eigen_utils::add_inplace<3, 3>(sum_outer, eigen_utils::multiply<3, 3>(outer, weights[j]));
        ++correspondences;
        total_weight += weights[j];
    }

    if (correspondences < min_num_correspondences || total_weight < std::numeric_limits<float>::epsilon()) {
        ret(0, 0) = 1.0f;
        ret(1, 1) = 1.0f;
        ret(2, 2) = 1.0f;
        return false;
    }

    const Eigen::Vector3f mean_3d = eigen_utils::multiply<3>(sum_points, 1.0f / total_weight);
    mean.setZero();
    mean.head<3>() = mean_3d;
    ret.block<3, 3>(0, 0) = eigen_utils::ensure_symmetric<3>(eigen_utils::subtract<3, 3>(
        eigen_utils::multiply<3, 3>(sum_outer, 1.0f / total_weight), eigen_utils::outer<3>(mean_3d, mean_3d)));
    return true;
}

SYCL_EXTERNAL inline Covariance compute_covariance_inverse(const Covariance& cov) {
    Covariance ret = Covariance::Zero();
    const Eigen::Matrix3f cov_inv = eigen_utils::inverse(cov.block<3, 3>(0, 0));
    ret.block<3, 3>(0, 0) = cov_inv;
    return ret;
}

template <typename T>
SYCL_EXTERNAL float compute_median(const T* data, T* buffer, size_t data_num) {
    if (data_num == 0) return 0.0f;

    // 1. Copy data to temporary buffer for in-place sorting
    if (data != buffer) {
        for (size_t i = 0; i < data_num; ++i) {
            buffer[i] = data[i];
        }
    }

    // 2. Perform Insertion Sort (Efficient for small N and low memory overhead)
    for (size_t i = 1; i < data_num; ++i) {
        const T key = buffer[i];
        size_t j = i;

        while (j > 0 && buffer[j - 1] > key) {
            buffer[j] = buffer[j - 1];
            --j;
        }
        buffer[j] = key;
    }

    // 3. Extract Median
    const size_t mid = data_num / 2;
    if (data_num % 2 == 0) {
        return static_cast<float>(buffer[mid - 1] + buffer[mid]) * 0.5f;
    } else {
        return static_cast<float>(buffer[mid]);
    }
}

SYCL_EXTERNAL inline float compute_mahalanobis_distance_squared(const Covariance& cov_inv, const PointType& mean,
                                                                const PointType& pt) {
    const PointType diff(pt.x() - mean.x(), pt.y() - mean.y(), pt.z() - mean.z(), 0.0f);
    const float squared_norm = eigen_utils::dot<4>(diff, eigen_utils::multiply<4, 4>(cov_inv, diff));
    return squared_norm;
}

template <size_t MAX_K = 32, robust::RobustLossType robust_type = robust::RobustLossType::CAUCHY>
SYCL_EXTERNAL Covariance compute_covariances_with_m_estimation(const PointType* point_ptr, size_t k_correspondences,
                                                               const int32_t* index_ptr, size_t i, float mad_scale,
                                                               float min_robust_scale, size_t robust_max_iter) {
    float weights[MAX_K];
    float dist_squared[MAX_K];
    // std::fill(weights, weights + MAX_K, 1.0f);
    // std::fill(dist_squared, dist_squared + MAX_K, 0.0f);
    for (size_t j = 0; j < MAX_K; ++j) {
        weights[j] = 1.0f;
        dist_squared[j] = 0.0f;
    }

    Covariance cov;
    PointType mean;

    // compute initial cov and mean
    bool success = kernel::compute_covariance_weighted(cov, mean, point_ptr, k_correspondences, index_ptr, weights, i);
    if (success) {
        // robust estimation
        for (size_t iter = 0; iter < robust_max_iter; ++iter) {
            const Covariance cov_inv = kernel::compute_covariance_inverse(cov);
            for (size_t j = 0; j < k_correspondences; ++j) {
                const int32_t idx = index_ptr[i * k_correspondences + j];
                if (idx < 0) continue;
                dist_squared[j] = kernel::compute_mahalanobis_distance_squared(cov_inv, mean, point_ptr[idx]);
            }
            // compute median (weights use as buffer)
            const float median = kernel::compute_median(dist_squared, weights, k_correspondences);

            // compute robust
            float robust_scale = mad_scale * median;
            // Add a lower bound to prevent the scale from becoming too small
            if (robust_scale < min_robust_scale) {
                robust_scale = min_robust_scale;
            }

            for (size_t j = 0; j < k_correspondences; ++j) {
                weights[j] = robust::kernel::compute_robust_weight<robust_type>(dist_squared[j], robust_scale);
            }

            success =
                kernel::compute_covariance_weighted(cov, mean, point_ptr, k_correspondences, index_ptr, weights, i);
            if (!success) {
                break;
            }
        }
    }
    return cov;
}

template <size_t MAX_K = 32>
SYCL_EXTERNAL Covariance compute_covariances_with_m_estimation(const PointType* point_ptr, size_t k_correspondences,
                                                               const int32_t* index_ptr, size_t i, float mad_scale,
                                                               float min_robust_scale, size_t robust_max_iter,
                                                               robust::RobustLossType robust_type) {
    switch (robust_type) {
        case robust::RobustLossType::HUBER:
            return compute_covariances_with_m_estimation<MAX_K, robust::RobustLossType::HUBER>(
                point_ptr, k_correspondences, index_ptr, i, mad_scale, min_robust_scale, robust_max_iter);
            break;
        case robust::RobustLossType::TUKEY:
            return compute_covariances_with_m_estimation<MAX_K, robust::RobustLossType::TUKEY>(
                point_ptr, k_correspondences, index_ptr, i, mad_scale, min_robust_scale, robust_max_iter);
            break;
        case robust::RobustLossType::CAUCHY:
            return compute_covariances_with_m_estimation<MAX_K, robust::RobustLossType::CAUCHY>(
                point_ptr, k_correspondences, index_ptr, i, mad_scale, min_robust_scale, robust_max_iter);
            break;
        case robust::RobustLossType::GEMAN_MCCLURE:
            return compute_covariances_with_m_estimation<MAX_K, robust::RobustLossType::GEMAN_MCCLURE>(
                point_ptr, k_correspondences, index_ptr, i, mad_scale, min_robust_scale, robust_max_iter);
            break;
        default:
            break;
    }
}
}  // namespace kernel

/// @brief Async compute covariance using SYCL
/// @param queue SYCL queue
/// @param neightbors KNN search result
/// @param points Point Container
/// @param covs Covariance Container
/// @return eventscd
inline sycl_utils::events compute_covariances_async(
    const sycl_utils::DeviceQueue& queue, const knn::KNNResult& neightbors, const PointContainerShared& points,
    CovarianceContainerShared& covs, const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    const size_t N = points.size();
    if (covs.size() != N) {
        covs.resize(N);
    }
    if (N == 0) return sycl_utils::events();

    const size_t work_group_size = queue.get_work_group_size();
    const size_t global_size = queue.get_global_size(N);

    sycl_utils::events events;
    events += queue.ptr->submit([&](sycl::handler& h) {
        const auto point_ptr = points.data();
        const auto cov_ptr = covs.data();
        const auto index_ptr = neightbors.indices->data();
        const auto k_correspondences = neightbors.k;

        h.depends_on(depends);
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;
            kernel::compute_covariance(cov_ptr[i], point_ptr, k_correspondences, index_ptr, i);
        });
    });
    return events;
}

/// @brief Compute covariance using SYCL
/// @param neightbors KNN search result
/// @param points Point Cloud
/// @return events
inline sycl_utils::events compute_covariances_async(
    const knn::KNNResult& neightbors, const PointCloudShared& points,
    const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    return compute_covariances_async(points.queue, neightbors, *points.points, *points.covs, depends);
}

/// @brief Async compute covariance using SYCL
/// @param knn KNN search
/// @param points Point Cloud
/// @param k_correspondences Number of neighbor points
/// @return events
inline sycl_utils::events compute_covariances_async(
    const knn::KNNBase& knn, const PointCloudShared& points, const size_t k_correspondences,
    const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    knn::KNNResult neightbors;
    auto knn_events = knn.knn_search_async(points, k_correspondences, neightbors, depends);
    return compute_covariances_async(neightbors, points, knn_events.evs);
}

/// @brief Async compute covariance with M estimation using SYCL
/// @param queue SYCL queue
/// @param neightbors KNN search result
/// @param points Point Container
/// @param covs Covariance Container
/// @return eventscd
inline sycl_utils::events compute_covariances_with_m_estimation_async(
    const sycl_utils::DeviceQueue& queue, const knn::KNNResult& neightbors, const PointContainerShared& points,
    CovarianceContainerShared& covs, robust::RobustLossType robust_type = robust::RobustLossType::CAUCHY,
    float mad_scale = 1.0f, float min_robust_scale = 1.0f, size_t robust_max_iterations = 1,
    const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    constexpr size_t MAX_K = 64;
    if (MAX_K < neightbors.k) {
        throw std::runtime_error("[compute_covariances_with_m_estimation_async] neightbor K is too large. MAX_K is 64");
    }

    const size_t N = points.size();
    if (covs.size() != N) {
        covs.resize(N);
    }
    if (N == 0) return sycl_utils::events();

    const size_t work_group_size = queue.get_work_group_size();
    const size_t global_size = queue.get_global_size(N);

    sycl_utils::events events;
    events += queue.ptr->submit([&](sycl::handler& h) {
        const auto point_ptr = points.data();
        const auto cov_ptr = covs.data();
        const auto index_ptr = neightbors.indices->data();
        const auto k_correspondences = neightbors.k;
        const auto robust_max_iter = robust_max_iterations;

        h.depends_on(depends);
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;

            Covariance cov;
            if (k_correspondences <= 16) {
                cov = kernel::compute_covariances_with_m_estimation<16>(point_ptr, k_correspondences, index_ptr, i,
                                                                        mad_scale, min_robust_scale, robust_max_iter,
                                                                        robust_type);
            } else if (k_correspondences <= 32) {
                cov = kernel::compute_covariances_with_m_estimation<32>(point_ptr, k_correspondences, index_ptr, i,
                                                                        mad_scale, min_robust_scale, robust_max_iter,
                                                                        robust_type);
            } else if (k_correspondences <= 64) {
                cov = kernel::compute_covariances_with_m_estimation<64>(point_ptr, k_correspondences, index_ptr, i,
                                                                        mad_scale, min_robust_scale, robust_max_iter,
                                                                        robust_type);
            } else {
                cov.setIdentity();
            }

            // write to global memory
            eigen_utils::copy<4, 4>(cov, cov_ptr[i]);
        });
    });
    return events;
}

inline sycl_utils::events compute_covariances_with_m_estimation_async(
    const knn::KNNResult& neightbors, const PointCloudShared& points,
    robust::RobustLossType robust_type = robust::RobustLossType::CAUCHY, float mad_scale = 1.0f,
    float min_robust_scale = 1.0f, size_t robust_max_iterations = 1,
    const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    return compute_covariances_with_m_estimation_async(points.queue, neightbors, *points.points, *points.covs,
                                                       robust_type, mad_scale, min_robust_scale, robust_max_iterations,
                                                       depends);
}

inline sycl_utils::events compute_covariances_with_m_estimation_async(
    const knn::KNNBase& knn, const PointCloudShared& points, const size_t k_correspondences,
    robust::RobustLossType robust_type = robust::RobustLossType::CAUCHY, float mad_scale = 1.0f,
    float min_robust_scale = 1.0f, size_t robust_max_iterations = 1,
    const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    knn::KNNResult neightbors;
    auto knn_events = knn.knn_search_async(points, k_correspondences, neightbors, depends);
    return compute_covariances_with_m_estimation_async(neightbors, points, robust_type, mad_scale, min_robust_scale,
                                                       robust_max_iterations, knn_events.evs);
}

/// @brief Compute normal vector using SYCL
/// @param neightbors KNN search result
/// @param points Point Cloud
/// @return events
inline sycl_utils::events compute_normals_async(const knn::KNNResult& neightbors, const PointCloudShared& points,
                                                const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    const size_t N = points.size();
    if (points.normals->size() != N) {
        points.normals->resize(N);
    }
    if (N == 0) return sycl_utils::events();

    const size_t work_group_size = points.queue.get_work_group_size();
    const size_t global_size = points.queue.get_global_size(N);

    sycl_utils::events events;
    events += points.queue.ptr->submit([&](sycl::handler& h) {
        const auto point_ptr = points.points_ptr();
        const auto normal_ptr = points.normals_ptr();
        const auto index_ptr = neightbors.indices->data();
        const auto k_correspondences = neightbors.k;

        h.depends_on(depends);
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;
            Covariance cov;
            kernel::compute_covariance(cov, point_ptr, k_correspondences, index_ptr, i);
            kernel::compute_normal_from_covariance(point_ptr[i], cov, normal_ptr[i]);
        });
    });
    return events;
}

/// @brief Async compute normal vector using SYCL
/// @param knn KNN search
/// @param points Point Cloud
/// @param k_correspondences Number of neighbor points
/// @return events
inline sycl_utils::events compute_normals_async(const knn::KNNBase& knn, const PointCloudShared& points,
                                                const size_t k_correspondences,
                                                const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    knn::KNNResult neightbors;
    auto knn_events = knn.knn_search_async(points, k_correspondences, neightbors, depends);
    return compute_normals_async(neightbors, points, knn_events.evs);
}

/// @brief Async compute normal vector from covariance using SYCL
/// @param points Point Cloud with covatiance
/// @return events
inline sycl_utils::events compute_normals_from_covariances_async(
    const PointCloudShared& points, const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    const size_t N = points.size();
    if (!points.has_cov()) {
        throw std::runtime_error("[compute_normals_from_covariances_async] not computed covariances");
    }

    const size_t work_group_size = points.queue.get_work_group_size();
    const size_t global_size = points.queue.get_global_size(N);

    if (points.normals->size() != N) {
        points.resize_normals(N);
    }

    sycl_utils::events events;
    events += points.queue.ptr->submit([&](sycl::handler& h) {
        const auto point_ptr = points.points_ptr();
        const auto cov_ptr = points.covs_ptr();
        const auto normal_ptr = points.normals_ptr();

        h.depends_on(depends);
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;

            kernel::compute_normal_from_covariance(point_ptr[i], cov_ptr[i], normal_ptr[i]);
        });
    });

    return events;
}

/// @brief Compute normal vector from covariance
/// @param points Point Cloud with covatiance
inline void compute_normals_from_covariances(const PointCloudShared& points,
                                             const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    const size_t N = points.size();
    compute_normals_from_covariances_async(points, depends).wait_and_throw();
}
}  // namespace covariance
}  // namespace algorithms
}  // namespace sycl_points
