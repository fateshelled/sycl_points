#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstdint>
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
/// @brief Color gradient type (Row(0): r, Row(1): g, Row(2): b; Col(0-2): gradient, Col(3): unused)
using ColorGradient = Eigen::Matrix3f;
/// @brief Intensity gradient type (gradient along x, y, z)
using IntensityGradient = Eigen::Vector3f;
using TransformMatrix = Eigen::Matrix4f;

/// @brief Timestamp offset representation relative to the first measurement.
using TimestampOffset = std::uint32_t;

/// @brief IMU measurement packet bundled for deskewing.
struct IMUData {
  /// @brief Timestamp seconds following ROS2 format.
  int32_t timestamp_sec{0};
  /// @brief Timestamp nanoseconds following ROS2 format.
  uint32_t timestamp_nanosec{0};
  Eigen::Vector3f angular_velocity{Eigen::Vector3f::Zero()};
  Eigen::Vector3f linear_acceleration{Eigen::Vector3f::Zero()};

  /// @brief Default constructor for containers.
  IMUData() = default;

  /// @brief Construct an IMU sample with explicit values.
  /// @param sec Timestamp seconds component.
  /// @param nanosec Timestamp nanoseconds component.
  /// @param angular_vel Angular velocity (rad/s).
  /// @param linear_acc Linear acceleration (m/s^2).
  IMUData(int32_t sec, uint32_t nanosec, const Eigen::Vector3f &angular_vel,
          const Eigen::Vector3f &linear_acc)
      : timestamp_sec(sec),
        timestamp_nanosec(nanosec),
        angular_velocity(angular_vel),
        linear_acceleration(linear_acc) {}

  /// @brief Convert ROS2 timestamp representation to seconds.
  /// @return Timestamp in seconds as double precision floating point.
  double timestamp_seconds() const {
    constexpr double kNanoToSec = 1e-9;
    return static_cast<double>(timestamp_sec) +
           static_cast<double>(timestamp_nanosec) * kNanoToSec;
  }
};

constexpr size_t PointAlignment = 16;
constexpr size_t CovarianceAlignment = 64;
constexpr size_t NormalAlignment = 16;
constexpr size_t RGBAlignment = 16;
constexpr size_t IntensityAlignment = 4;
constexpr size_t ColorGradientAlignment = 0; // 36 is bad alignment
constexpr size_t IntensityGradientAlignment = 16;
constexpr size_t TimestampAlignment = alignof(TimestampOffset);

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

// Vector of Color gradient on CPU
using ColorGradientContainerCPU = std::vector<ColorGradient, Eigen::aligned_allocator<ColorGradient>>;
// Vector of Color gradient on shared memory
using ColorGradientContainerShared = shared_vector<ColorGradient, ColorGradientAlignment>;

// Vector of intensity gradient on CPU
using IntensityGradientContainerCPU =
    std::vector<IntensityGradient, Eigen::aligned_allocator<IntensityGradient>>;
// Vector of intensity gradient on shared memory
using IntensityGradientContainerShared = shared_vector<IntensityGradient, IntensityGradientAlignment>;

// Vector of Intensity on CPU
using IntensityContainerCPU = std::vector<float, Eigen::aligned_allocator<float>>;
// Vector of RGB on shared memory
using IntensityContainerShared = shared_vector<float, IntensityAlignment>;

// Vector of timestamp offsets on CPU
using TimestampContainerCPU = std::vector<TimestampOffset>;
// Vector of timestamp offsets on shared memory
using TimestampContainerShared = shared_vector<TimestampOffset, TimestampAlignment>;

// Vector of IMU data on CPU.
using IMUDataContainerCPU = std::vector<IMUData>;
// Vector of IMU data on shared memory.
using IMUDataContainerShared = shared_vector<IMUData, alignof(IMUData)>;

}  // namespace sycl_points
