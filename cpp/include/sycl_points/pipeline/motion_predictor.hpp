#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "sycl_points/pipeline/adaptive_motion_predictor.hpp"

namespace sycl_points {
namespace pipeline {
namespace lidar_odometry {

enum class MotionPredictionMode {
    LIDAR_CV = 0,
    GYRO_LIDAR_CV,
    IMU_SE3,
};

inline MotionPredictionMode MotionPredictionMode_from_string(const std::string& str) {
    std::string upper = str;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });
    if (upper == "LIDAR_CV") return MotionPredictionMode::LIDAR_CV;
    if (upper == "GYRO_LIDAR_CV") return MotionPredictionMode::GYRO_LIDAR_CV;
    if (upper == "IMU_SE3") return MotionPredictionMode::IMU_SE3;
    throw std::runtime_error("[MotionPredictionMode_from_string] Invalid motion prediction mode '" + str + "'");
}

inline std::string MotionPredictionMode_to_string(const MotionPredictionMode mode) {
    switch (mode) {
        case MotionPredictionMode::LIDAR_CV:
            return "LIDAR_CV";
        case MotionPredictionMode::GYRO_LIDAR_CV:
            return "GYRO_LIDAR_CV";
        case MotionPredictionMode::IMU_SE3:
            return "IMU_SE3";
    }
    throw std::runtime_error("[MotionPredictionMode_to_string] Invalid motion prediction mode");
}

struct MotionPredictionCandidates {
    std::optional<Eigen::Matrix3f> gyro_delta_rotation_lidar;
    std::optional<Eigen::Isometry3f> imu_se3_pose;
};

/// Selects and combines available initial-pose prediction sources.
class MotionPredictor {
public:
    using Ptr = std::shared_ptr<MotionPredictor>;

    struct Params : AdaptiveMotionPredictor::Params {
        MotionPredictionMode mode = MotionPredictionMode::GYRO_LIDAR_CV;
    };

    explicit MotionPredictor(const Params& params) : params_(params), lidar_cv_predictor_(params) {}

    Eigen::Isometry3f predict(const Eigen::Vector3f& linear_velocity, const Eigen::AngleAxisf& angular_velocity,
                              const Eigen::Isometry3f& odom, float dt,
                              const algorithms::registration::RegistrationResult::Ptr& reg_result, bool registrated,
                              const MotionPredictionCandidates& candidates = {}) {
        if (params_.mode == MotionPredictionMode::IMU_SE3 && candidates.imu_se3_pose) {
            return *candidates.imu_se3_pose;
        }

        Eigen::Isometry3f prediction =
            lidar_cv_predictor_.predict(linear_velocity, angular_velocity, odom, dt, reg_result, registrated);
        if (params_.mode == MotionPredictionMode::GYRO_LIDAR_CV && candidates.gyro_delta_rotation_lidar) {
            Eigen::Isometry3f relative = odom.inverse() * prediction;
            relative.linear() = *candidates.gyro_delta_rotation_lidar;
            prediction = odom * relative;
        }
        return prediction;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    Params params_;
    AdaptiveMotionPredictor lidar_cv_predictor_;
};

}  // namespace lidar_odometry
}  // namespace pipeline
}  // namespace sycl_points
