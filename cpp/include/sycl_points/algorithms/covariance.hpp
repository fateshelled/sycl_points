#pragma once

#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {

namespace algorithms {

namespace covariance {

namespace kernel {

SYCL_EXTERNAL inline void compute_covariance(Covariance& ret, const PointType* point_ptr,
                                             const size_t k_correspondences, const int32_t* index_ptr, const size_t i) {
    ret.setZero();
    PointType sum_points = PointType::Zero();
    Eigen::Matrix3f sum_outer = Eigen::Matrix3f::Zero();

    for (size_t j = 0; j < k_correspondences; ++j) {
        const auto pt = point_ptr[index_ptr[i * k_correspondences + j]];
        eigen_utils::add_inplace<4, 1>(sum_points, pt);

        const auto outer = eigen_utils::outer<4>(pt, pt).block<3, 3>(0, 0);
        eigen_utils::add_inplace<3, 3>(sum_outer, outer);
    }

    const PointType mean = eigen_utils::multiply<4>(sum_points, 1.0f / k_correspondences);

    ret.block<3, 3>(0, 0) = eigen_utils::ensure_symmetric<3>(
        eigen_utils::subtract<3, 3>(sum_outer, eigen_utils::outer<4>(mean, sum_points).block<3, 3>(0, 0)));
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
    constexpr float min_eigenvalue = 1e-3f;
    constexpr float min_condition_number = 1.0e-2f;

    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;
    eigen_utils::symmetric_eigen_decomposition_3x3(cov.block<3, 3>(0, 0), eigenvalues, eigenvectors);

    // Clamp eigenvalues to ensure minimum value and condition number
    eigenvalues(2) = std::max(eigenvalues(2), min_eigenvalue);
    const float min_allowed = std::max(min_condition_number * eigenvalues(2), min_eigenvalue);
    eigenvalues(0) = std::max(eigenvalues(0), min_allowed);
    eigenvalues(1) = std::max(eigenvalues(1), min_allowed);

    const auto diag = eigen_utils::as_diagonal<3>(eigenvalues);
    cov.block<3, 3>(0, 0) = eigen_utils::multiply<3, 3, 3>(eigen_utils::multiply<3, 3, 3>(eigenvectors, diag),
                                                           eigen_utils::transpose<3, 3>(eigenvectors));
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

/// @brief Async update covariance matrix to a plane
/// @param points Point Cloud with covatiance
/// @return events
inline sycl_utils::events covariance_update_plane_async(
    const PointCloudShared& points, const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    if (!points.has_cov()) {
        throw std::runtime_error("[covariance_update_plane_async] not computed covariances");
    }

    const size_t N = points.size();
    const size_t work_group_size = points.queue.get_work_group_size();
    const size_t global_size = points.queue.get_global_size(N);

    sycl_utils::events events;
    events += points.queue.ptr->submit([&](sycl::handler& h) {
        const auto cov_ptr = points.covs->data();

        h.depends_on(depends);
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;

            kernel::update_covariance_plane(cov_ptr[i]);
        });
    });

    return events;
}

/// @brief Update covariance matrix to a plane
/// @param points Point Cloud with covatiance
inline void covariance_update_plane(const PointCloudShared& points,
                                    const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    covariance_update_plane_async(points, depends).wait_and_throw();
}

/// @brief Async normalize covariance matrix
/// @param points Point Cloud with covatiance
/// @return events
inline sycl_utils::events covariance_normalize_async(
    const PointCloudShared& points, const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    if (!points.has_cov()) {
        throw std::runtime_error("[covariance_normalize_async] not computed covariances");
    }

    const size_t N = points.size();
    const size_t work_group_size = points.queue.get_work_group_size();
    const size_t global_size = points.queue.get_global_size(N);

    sycl_utils::events events;
    events += points.queue.ptr->submit([&](sycl::handler& h) {
        const auto cov_ptr = points.covs->data();

        h.depends_on(depends);
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;

            kernel::normalize_covariance(cov_ptr[i]);
        });
    });

    return events;
}

/// @brief Normalize covariance matrix
/// @param points Point Cloud with covatiance
inline void covariance_normalize(const PointCloudShared& points,
                                 const std::vector<sycl::event>& depends = std::vector<sycl::event>()) {
    covariance_normalize_async(points, depends).wait_and_throw();
}

}  // namespace covariance
}  // namespace algorithms
}  // namespace sycl_points
