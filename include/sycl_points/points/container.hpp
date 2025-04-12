#pragma once

#include <Eigen/Dense>
#include <vector>

#include <sycl_points/utils/sycl_utils.hpp>


namespace sycl_points {

    using PointType = Eigen::Vector4f;
    using Covariance = Eigen::Matrix4f;
    using TransformMatrix = Eigen::Matrix4f;

    constexpr size_t PointAlignment = 16;
    constexpr size_t CovarianceAlignment = 64;

    using PointAllocatorHost = host_allocator<PointType, PointAlignment>;
    using CovarianceAllocatorHost = host_allocator<PointType, CovarianceAlignment>;
    using PointAllocatorShared = shared_allocator<PointType, PointAlignment>;
    using CovarianceAllocatorShared = shared_allocator<PointType, CovarianceAlignment>;

    using PointContainerCPU = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    using PointContainerHost = host_vector<PointType, PointAlignment>;
    using PointContainerShared = shared_vector<PointType, PointAlignment>;
    using PointContainerDevice = sycl_utils::ContainerDevice<PointType, PointAlignment>;

    using CovarianceContainerCPU = std::vector<Covariance, Eigen::aligned_allocator<Covariance>>;
    using CovarianceContainerHost = host_vector<Covariance, CovarianceAlignment>;
    using CovarianceContainerShared = shared_vector<Covariance, CovarianceAlignment>;
    using CovarianceContainerDevice = sycl_utils::ContainerDevice<Covariance, CovarianceAlignment>;

} // namespace sycl_points
