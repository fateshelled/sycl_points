#pragma once

#include "knn_search.hpp"
#include "point_cloud.hpp"

namespace sycl_points {

namespace kernel {

SYCL_EXTERNAL inline void compute_covariance(Covariance& ret, const PointType* point_ptr,
                                             const size_t k_correspondences, const int* index_ptr, const size_t i) {
    PointType sum_points = PointType::Zero();
    Eigen::Matrix3f sum_outer = Eigen::Matrix3f::Zero();

    for (size_t j = 0; j < k_correspondences; ++j) {
        const auto pt = point_ptr[index_ptr[i * k_correspondences + j]];
        eigen_utils::add_zerocopy<4, 1>(sum_points, pt);

        const auto outer = eigen_utils::block3x3(eigen_utils::outer(pt, pt));
        eigen_utils::add_zerocopy<3, 3>(sum_outer, outer);
    }

    const PointType mean = eigen_utils::multiply<4>(sum_points, 1.0f / k_correspondences);

    const auto cov3x3 = eigen_utils::ensure_symmetric<3>(
        eigen_utils::subtract<3, 3>(sum_outer, eigen_utils::block3x3(eigen_utils::outer(mean, sum_points))));

    ret(0, 0) = cov3x3(0, 0);
    ret(0, 1) = cov3x3(0, 1);
    ret(0, 2) = cov3x3(0, 2);

    ret(1, 0) = cov3x3(1, 0);
    ret(1, 1) = cov3x3(1, 1);
    ret(1, 2) = cov3x3(1, 2);

    ret(2, 0) = cov3x3(2, 0);
    ret(2, 1) = cov3x3(2, 1);
    ret(2, 2) = cov3x3(2, 2);
}

SYCL_EXTERNAL inline void update_covariance_plane(Covariance& cov) {
    Eigen::Vector3f eigenvalues;
    Eigen::Matrix3f eigenvectors;
    eigen_utils::symmetric_eigen_decomposition_3x3(eigen_utils::block3x3(cov), eigenvalues, eigenvectors);
    const auto diag = eigen_utils::as_diagonal<3>({1e-3f, 1.0f, 1.0f});
    const auto new_cov = eigen_utils::multiply<3, 3, 3>(eigen_utils::multiply<3, 3, 3>(eigenvectors, diag),
                                                        eigen_utils::transpose<3, 3>(eigenvectors));
    cov(0, 0) = new_cov(0, 0);
    cov(0, 1) = new_cov(0, 1);
    cov(0, 2) = new_cov(0, 2);
    cov(1, 0) = new_cov(1, 0);
    cov(1, 1) = new_cov(1, 1);
    cov(1, 2) = new_cov(1, 2);
    cov(2, 0) = new_cov(2, 0);
    cov(2, 1) = new_cov(2, 1);
    cov(2, 2) = new_cov(2, 2);
}

}  // namespace kernel

/// @brief Async compute covariance using SYCL
/// @param queue SYCL queue
/// @param neightbors KNN search result
/// @param points Point Container
/// @param covs Covariance Container
/// @return events
inline sycl_utils::events compute_covariances_sycl_async(const std::shared_ptr<sycl::queue>& queue_ptr, const KNNResultSYCL& neightbors,
                                                         const PointContainerShared& points,
                                                         CovarianceContainerShared& covs) {
    const size_t N = points.size();
    covs.resize(N, Covariance::Zero());

    const auto point_ptr = points.data();
    const auto cov_ptr = covs.data();
    const auto index_ptr = neightbors.indices->data();
    const auto k_correspondences = neightbors.k;

    // Optimize work group size
    const size_t work_group_size = sycl_utils::get_work_group_size(*queue_ptr);
    const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;

    auto event = queue_ptr->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(work_group_size)),
                       [=](sycl::nd_item<1> item) {
                           const size_t i = item.get_global_id(0);
                           if (i >= N) return;
                           kernel::compute_covariance(cov_ptr[i], point_ptr, k_correspondences, index_ptr, i);
                       });
    });
    sycl_utils::events events;
    events.push_back(event);
    return events;
}

/// @brief Async compute covariance using SYCL
/// @param kdtree KDTree
/// @param points Point Container
/// @param k_correspondences Number of neighbor points
/// @param covs Covariance Container
/// @return events
inline sycl_utils::events compute_covariances_sycl_async(const KDTreeSYCL& kdtree, const PointContainerShared& points,
                                                         const size_t k_correspondences,
                                                         CovarianceContainerShared& covs) {
    const auto neightbors = kdtree.knn_search(points, k_correspondences);
    return compute_covariances_sycl_async(kdtree.queue_, neightbors, points, covs);
}

/// @brief Async compute covariance using SYCL
/// @param kdtree KDTree
/// @param points Point Cloud
/// @param k_correspondences Number of neighbor points
/// @return events
inline sycl_utils::events compute_covariances_sycl_async(const KDTreeSYCL& kdtree, const PointCloudShared& points,
                                                         const size_t k_correspondences) {
    return compute_covariances_sycl_async(kdtree, *points.points, k_correspondences, *points.covs);
}

/// @brief Compute covariance using SYCL
/// @param queue SYCL queue
/// @param neightbors KNN search result
/// @param points Point Container
inline void compute_covariances_sycl(const std::shared_ptr<sycl::queue>& queue_ptr, const KNNResultSYCL& neightbors,
                                     const PointCloudShared& points) {
    const size_t N = points.size();
    compute_covariances_sycl_async(queue_ptr, neightbors, *points.points, *points.covs).wait();
}

/// @brief Compute covariance using SYCL
/// @param queue SYCL queue
/// @param neightbors KNN search result
/// @param points Point Container
/// @return Covariances
inline CovarianceContainerShared compute_covariances_sycl(const std::shared_ptr<sycl::queue>& queue_ptr, const KNNResultSYCL& neightbors,
                                                          const PointContainerShared& points) {
    const size_t N = points.size();
    CovarianceContainerShared covs(N, Covariance::Zero(), CovarianceAllocatorShared(*queue_ptr));
    compute_covariances_sycl_async(queue_ptr, neightbors, points, covs).wait();
    return covs;
}

/// @brief Compute covariance using SYCL
/// @param kdtree KDTree
/// @param points Point Container
/// @param k_correspondences Number of neighbor points
/// @return Covariances
inline CovarianceContainerShared compute_covariances_sycl(const KDTreeSYCL& kdtree, const PointContainerShared& points,
                                                          const size_t k_correspondences) {
    const KNNResultSYCL neightbors = kdtree.knn_search(points, k_correspondences);
    return compute_covariances_sycl(kdtree.queue_, neightbors, points);
}

/// @brief Compute covariance using SYCL
/// @param kdtree KDTree
/// @param points Point Cloud
/// @param k_correspondences Number of neighbor points
inline void compute_covariances_sycl(const KDTreeSYCL& kdtree, const PointCloudShared& points,
                                     const size_t k_correspondences) {
    *points.covs = compute_covariances_sycl(kdtree, *points.points, k_correspondences);
}

/// @brief Update covariance matrix to a plane
/// @param points Point Cloud with covatiance
/// @return Covariances
inline void covariance_update_plane(const PointCloudShared& points) {
    const Eigen::Vector3f values = {1e-3f, 1.0f, 1.0f};

    for (auto& cov : *points.covs) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
        eig.computeDirect(cov.block<3, 3>(0, 0));
        cov.block<3, 3>(0, 0) = eig.eigenvectors() * values.asDiagonal() * eig.eigenvectors().transpose();
    }
}

inline sycl_utils::events covariance_update_plane_sycl_async(const std::shared_ptr<sycl::queue>& queue_ptr, const PointCloudShared& points) {
    const size_t N = points.size();
    const size_t work_group_size = sycl_utils::get_work_group_size(*queue_ptr);
    const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;

    auto event = queue_ptr->submit([&](sycl::handler& h) {
        const auto cov_ptr = points.covs->data();
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(work_group_size)),
                       [=](sycl::nd_item<1> item) {
                           const size_t i = item.get_global_id(0);
                           if (i >= N) return;

                           kernel::update_covariance_plane(cov_ptr[i]);
                       });
    });

    sycl_utils::events events;
    events.push_back(event);
    return events;
}

inline void covariance_update_plane_sycl(const std::shared_ptr<sycl::queue>& queue_ptr, const PointCloudShared& points) {
    covariance_update_plane_sycl_async(queue_ptr, points).wait();
}
}  // namespace sycl_points
