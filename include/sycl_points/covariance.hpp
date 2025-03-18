#pragma once

#include "point_cloud.hpp"
#include "knn_search.hpp"

namespace sycl_points {

template <typename T = float>
inline CovarianceContainerCPU<T> computeCovariances(
  KNNSearch<T>& kdtree,             // KDTree
  const PointContainerCPU<T>& points,  // Point Cloud
  const size_t k_correspondences,   // Number of neighbor points
  const size_t num_threads = 1) {

  const auto neightbors = kdtree.searchKDTree_sycl(points, k_correspondences);

  const size_t N = points.size();
  CovarianceContainerCPU covs(N);

#pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < N; ++i) {
    PointType<T> sum_points = PointType<T>::Zero();
    Covariance<T> sum_cross = Covariance<T>::Zero();

    const auto& indices = neightbors.indices[i];
    for (size_t j = 0; j < k_correspondences; ++j) {
      const auto& pt = points[indices[j]];
      sum_points += pt;
      sum_cross += pt * pt.transpose();
    }
    const PointType<T> mean = sum_points / k_correspondences;
    covs[i] = (sum_cross - mean * sum_points.transpose()) / k_correspondences;
  }

  return covs;
}

template <typename T = float>
inline CovarianceContainerCPU<T> computeCovariances(
  sycl::queue& queue,               // SYCL execution queue
  const PointContainerCPU<T>& points,  // Point Cloud
  const size_t k_correspondences,   // Number of neighbor points
  const size_t num_threads = 1) {
  auto kdtree = KNNSearch<T>::buildKDTree(queue, points);
  return computeCovariances<T>(kdtree, points, k_correspondences, num_threads);
}

template <typename T = float>
inline void computeCovariances(
  KNNSearch<T>& kdtree,            // KDTree
  PointCloudCPU<T>& points,     // Point Cloud
  const size_t k_correspondences,  // Number of neighbor points
  const size_t num_threads = 1) {
  points.covs = computeCovariances<T>(kdtree, points.points, k_correspondences, num_threads);
}

template <typename T = float>
inline void computeCovariances(
  sycl::queue& queue,              // SYCL execution queue
  PointCloudCPU<T>& points,     // Point Cloud
  const size_t k_correspondences,  // Number of neighbor points
  const size_t num_threads = 1) {
  points.covs = computeCovariances(queue, points.points, k_correspondences, num_threads);
}

}  // namespace sycl_points
