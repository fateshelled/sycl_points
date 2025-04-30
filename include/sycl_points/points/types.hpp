#pragma once

#include <Eigen/Dense>
#include <sycl_points/utils/sycl_utils.hpp>
#include <vector>

namespace sycl_points {

using PointType = Eigen::Vector4f;
using Covariance = Eigen::Matrix4f;
using TransformMatrix = Eigen::Matrix4f;

constexpr size_t PointAlignment = 16;
constexpr size_t CovarianceAlignment = 64;

using PointAllocatorShared = shared_allocator<PointType, PointAlignment>;
using CovarianceAllocatorShared = shared_allocator<PointType, CovarianceAlignment>;

using PointContainerCPU = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
using PointContainerShared = shared_vector<PointType, PointAlignment>;

using CovarianceContainerCPU = std::vector<Covariance, Eigen::aligned_allocator<Covariance>>;
using CovarianceContainerShared = shared_vector<Covariance, CovarianceAlignment>;

}  // namespace sycl_points
