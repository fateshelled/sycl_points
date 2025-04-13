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

using PointAllocatorHost = host_allocator<PointType, PointAlignment>;
using PointAllocatorShared = shared_allocator<PointType, PointAlignment>;
// using PointAllocatorDevice = device_allocator<PointType, PointAlignment>;

using CovarianceAllocatorHost = host_allocator<PointType, CovarianceAlignment>;
using CovarianceAllocatorShared = shared_allocator<PointType, CovarianceAlignment>;
// using CovarianceAllocatorDevice = device_allocator<PointType, CovarianceAlignment>;

using PointContainerCPU = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
using PointContainerHost = host_vector<PointType, PointAlignment>;
using PointContainerShared = shared_vector<PointType, PointAlignment>;
// using PointContainerDevice = device_vector<PointType, PointAlignment>;
using PointContainerDevice = ContainerDevice<PointType, PointAlignment>;

using CovarianceContainerCPU = std::vector<Covariance, Eigen::aligned_allocator<Covariance>>;
using CovarianceContainerHost = host_vector<Covariance, CovarianceAlignment>;
using CovarianceContainerShared = shared_vector<Covariance, CovarianceAlignment>;
// using CovarianceContainerDevice = device_vector<Covariance, CovarianceAlignment>;
using CovarianceContainerDevice = ContainerDevice<Covariance, CovarianceAlignment>;

}  // namespace sycl_points
