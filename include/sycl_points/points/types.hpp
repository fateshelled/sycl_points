#pragma once

#include <Eigen/Dense>
#include <sycl_points/utils/sycl_utils.hpp>
#include <vector>

namespace sycl_points {

using PointType = Eigen::Vector4f;
using Covariance = Eigen::Matrix4f;
using Normal = Eigen::Vector4f;
/// @brief RGB color with 4 channels (RGBA).
/// @note data is in range [0.0, 1.0].
/// @note x: R, y: G, z: B, w: A
using RGBType = Eigen::Vector4f;
using TransformMatrix = Eigen::Matrix4f;

constexpr size_t PointAlignment = 16;
constexpr size_t CovarianceAlignment = 64;
constexpr size_t NormalAlignment = 16;
constexpr size_t RGBAlignment = 16;

// Vector of point on CPU. Accessible from CPU process only.
using PointContainerCPU = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
// Vector of point on shared memory. Accessible directly from the CPU and via pointer from the device.
using PointContainerShared = shared_vector<PointType, PointAlignment>;

// Vector of covariance on CPU
using CovarianceContainerCPU = std::vector<Covariance, Eigen::aligned_allocator<Covariance>>;
// Vector of covariance on shared memory
using CovarianceContainerShared = shared_vector<Covariance, CovarianceAlignment>;

// Vector of normal on CPU
using NormalContainerCPU = std::vector<Normal, Eigen::aligned_allocator<Normal>>;
// Vector of normal on shared memory
using NormalContainerShared = shared_vector<Normal, NormalAlignment>;

// Vector of RGB on CPU
using RGBContainerCPU = std::vector<RGBType, Eigen::aligned_allocator<RGBType>>;
// Vector of RGB on shared memory
using RGBContainerShared = shared_vector<RGBType, RGBAlignment>;

}  // namespace sycl_points
