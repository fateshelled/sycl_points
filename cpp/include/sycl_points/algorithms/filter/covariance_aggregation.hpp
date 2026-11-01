#pragma once

#include <cmath>
#include <stdexcept>

#include <Eigen/Eigenvalues>

#include "sycl_points/points/types.hpp"

namespace sycl_points::algorithms::filter {

// Only the spatial 3x3 block is stored in map files. Keep the homogeneous row zero.
inline Covariance regularize_map_covariance(const Covariance& input, float floor, size_t* clamped = nullptr) {
    if (!std::isfinite(floor) || floor <= 0.0f || !input.topLeftCorner<3, 3>().allFinite()) {
        throw std::invalid_argument("Invalid map covariance or eigenvalue floor");
    }
    const Eigen::Matrix3f symmetric =
        0.5f * (input.topLeftCorner<3, 3>() + input.topLeftCorner<3, 3>().transpose()).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(symmetric);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
        throw std::invalid_argument("Cannot decompose map covariance");
    }
    Eigen::Vector3f values = solver.eigenvalues();
    for (int i = 0; i < 3; ++i) {
        if (values[i] < floor) {
            values[i] = floor;
            if (clamped) ++*clamped;
        }
    }
    Covariance output = Covariance::Zero();
    output.topLeftCorner<3, 3>() = solver.eigenvectors() * values.asDiagonal() * solver.eigenvectors().transpose();
    return output;
}

inline Eigen::Matrix3f map_covariance_log(const Covariance& covariance, float floor, size_t* clamped = nullptr) {
    const auto normalized = regularize_map_covariance(covariance, floor, clamped);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(normalized.topLeftCorner<3, 3>());
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
        throw std::invalid_argument("Cannot compute map covariance logarithm");
    }
    return solver.eigenvectors() * solver.eigenvalues().array().max(floor).log().matrix().asDiagonal() *
           solver.eigenvectors().transpose();
}

inline Covariance map_covariance_exp(const Eigen::Matrix3f& mean_log) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(0.5f * (mean_log + mean_log.transpose()).eval());
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
        throw std::invalid_argument("Cannot compute map covariance exponential");
    }
    Covariance output = Covariance::Zero();
    output.topLeftCorner<3, 3>() =
        solver.eigenvectors() * solver.eigenvalues().array().exp().matrix().asDiagonal() * solver.eigenvectors().transpose();
    if (!output.allFinite()) throw std::invalid_argument("Map covariance exponential overflow");
    return output;
}

}  // namespace sycl_points::algorithms::filter
