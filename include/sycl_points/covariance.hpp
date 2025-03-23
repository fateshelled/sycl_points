#pragma once

#include "point_cloud.hpp"
#include "knn_search.hpp"

namespace sycl_points {

// inline CovarianceContainerCPU compute_covariances(
//   KNNSearch& kdtree,             // KDTree
//   const PointContainerCPU& points,  // Point Cloud
//   const size_t k_correspondences,   // Number of neighbor points
//   const size_t num_threads = 1) {

//   // todo
//   const auto neightbors = kdtree.searchBruteForce(points, k_correspondences);

//   const size_t N = points.size();
//   CovarianceContainerCPU covs(N);

// #pragma omp parallel for num_threads(num_threads)
//   for (size_t i = 0; i < N; ++i) {
//     PointType sum_points = PointType::Zero();
//     Covariance sum_cross = Covariance::Zero();

//     const auto& indices = neightbors.indices[i];
//     for (size_t j = 0; j < k_correspondences; ++j) {
//       const auto& pt = points[indices[j]];
//       sum_points += pt;
//       sum_cross += pt * pt.transpose();
//     }
//     const PointType mean = sum_points / k_correspondences;
//     covs[i] = (sum_cross - mean * sum_points.transpose()) / k_correspondences;
//   }

//   return covs;
// }

// inline CovarianceContainerCPU compute_covariances(
//   const PointContainerCPU& points,  // Point Cloud
//   const size_t k_correspondences,   // Number of neighbor points
//   const size_t num_threads = 1) {
//   auto kdtree = KNNSearch::buildKDTree(points);
//   return compute_covariances(kdtree, points, k_correspondences, num_threads);
// }

// inline void compute_covariances(
//   KNNSearch& kdtree,            // KDTree
//   PointCloudCPU& points,     // Point Cloud
//   const size_t k_correspondences,  // Number of neighbor points
//   const size_t num_threads = 1) {
//   points.covs = compute_covariances(kdtree, points.points, k_correspondences, num_threads);
// }

// inline void compute_covariances(
//   PointCloudCPU& points,     // Point Cloud
//   const size_t k_correspondences,  // Number of neighbor points
//   const size_t num_threads = 1) {
//   points.covs = compute_covariances(queue, points.points, k_correspondences, num_threads);
// }

/// @brief Compute covariance using SYCL
/// @param kdtree KDTree
/// @param points Point Container
/// @return Covariances
inline CovarianceContainerShared compute_covariances_sycl(
  sycl::queue& queue,
  const KNNResultSYCL& neightbors,     // KNN search result
  const PointContainerShared& points   // Point Cloud
) {
  const size_t N = points.size();
  CovarianceContainerShared covs(N, Covariance::Zero(), shared_allocator<Covariance>(queue));

  const auto point_ptr = points.data();
  const auto cov_ptr = covs.data();
  const auto index_ptr = neightbors.indices->data();
  const auto k_correspondences = neightbors.k;

  auto event = queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
      PointType sum_points = PointType::Zero();
      Eigen::Matrix3f sum_outer = Eigen::Matrix3f::Zero();

      for (size_t j = 0; j < k_correspondences; ++j) {
        const auto& pt = point_ptr[index_ptr[i * k_correspondences + j]];
        sum_points.x() += pt.x();
        sum_points.y() += pt.y();
        sum_points.z() += pt.z();
        sum_points.w() += pt.w();

        const auto outer = eigen_utils::block3x3(eigen_utils::outer(pt, pt));
        eigen_utils::add_zerocopy<3, 3>(sum_outer, outer);
      }
      const PointType mean = eigen_utils::multiply<4>(sum_points, 1.0f / k_correspondences);

      const auto cov3x3 = eigen_utils::subtract<3, 3>(sum_outer, eigen_utils::block3x3(eigen_utils::outer(mean, sum_points)));
      cov_ptr[i](0, 0) = cov3x3(0, 0);
      cov_ptr[i](0, 1) = cov3x3(0, 1);
      cov_ptr[i](0, 2) = cov3x3(0, 2);

      cov_ptr[i](1, 0) = cov3x3(1, 0);
      cov_ptr[i](1, 1) = cov3x3(1, 1);
      cov_ptr[i](1, 2) = cov3x3(1, 2);

      cov_ptr[i](2, 0) = cov3x3(2, 0);
      cov_ptr[i](2, 1) = cov3x3(2, 1);
      cov_ptr[i](2, 2) = cov3x3(2, 2);
    });
  });
  event.wait();

  return covs;
}

/// @brief Compute covariance using SYCL
/// @param kdtree KDTree
/// @param points Point Container
/// @param k_correspondences Number of neighbor points
/// @return Covariances
inline CovarianceContainerShared compute_covariances_sycl(
  const KNNSearchSYCL& kdtree,
  const PointContainerShared& points,
  const size_t k_correspondences) {
  const auto neightbors = kdtree.searchKDTree_sycl(points, k_correspondences);
  return compute_covariances_sycl(*kdtree.queue_, neightbors, points);
}

/// @brief Compute covariance using SYCL
/// @param kdtree KDTree
/// @param points Point Cloud
/// @param k_correspondences Number of neighbor points
/// @return Covariances
inline void compute_covariances_sycl(
  const KNNSearchSYCL& kdtree,
  const PointCloudShared& points,
  const size_t k_correspondences
) {
  const auto neightbors = kdtree.searchKDTree_sycl(points, k_correspondences);
  *points.covs = compute_covariances_sycl(*kdtree.queue_, neightbors, *points.points);
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

}  // namespace sycl_points
