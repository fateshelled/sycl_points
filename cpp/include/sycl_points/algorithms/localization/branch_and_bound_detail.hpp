#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "sycl_points/algorithms/localization/branch_and_bound_params.hpp"
#include "sycl_points/points/types.hpp"

namespace sycl_points::algorithms::localization::detail {

struct AngularInfo {
    Eigen::Vector3i num_division = Eigen::Vector3i::Ones();
    Eigen::Vector3f rpy_res = Eigen::Vector3f::Zero();
    Eigen::Vector3f min_rpy = Eigen::Vector3f::Zero();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline int checked_int(double value) {
    if (!std::isfinite(value) || value < static_cast<double>(std::numeric_limits<int>::min()) ||
        value > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("[BranchAndBoundLocalizer] grid index exceeds int range");
    }
    return static_cast<int>(value);
}

inline std::vector<float> level_resolutions(const BranchAndBoundParams& params) {
    std::vector<float> resolutions(static_cast<size_t>(params.max_level) + 1);
    resolutions[0] = params.min_resolution;
    for (int i = 1; i <= params.max_level; ++i) {
        resolutions[i] = resolutions[i - 1] * static_cast<float>(params.branch_factor);
        if (!std::isfinite(resolutions[i])) {
            throw std::overflow_error("[BranchAndBoundLocalizer] resolution overflow");
        }
    }
    return resolutions;
}

/// @brief Angular divisions from the BBS3D schedule.
inline std::vector<AngularInfo> angular_info(const BranchAndBoundParams& params, const std::vector<float>& resolutions,
                                             float max_norm) {
    std::vector<AngularInfo> info(resolutions.size());
    for (int level = params.max_level; level >= 0; --level) {
        const double ratio = max_norm > 0.0f ? static_cast<double>(resolutions[level]) / max_norm : 1e30;
        const float cosine = static_cast<float>(std::clamp(1.0 - 0.5 * ratio * ratio, -1.0, 1.0));
        const float ori_res = std::floor(std::acos(cosine) * 10000.0f) / 10000.0f;
        for (int axis = 0; axis < 3; ++axis) {
            const float range = params.max_rpy[axis] - params.min_rpy[axis];
            // A level starts a new range when it is the top level or its parent is undivided.
            const bool range_start = level == params.max_level || info[level + 1].rpy_res[axis] == 0.0f;
            const float piece = range_start ? range : info[level + 1].rpy_res[axis];
            const float proposed = ori_res > 0.0f && ori_res <= range ? ori_res : 0.0f;
            int divisions = proposed == 0.0f ? 1 : checked_int(std::ceil(static_cast<double>(piece) / proposed));
            // Keep zero on the angular grid. A new symmetric range needs an even division count so that
            // zero is the center of a cell. Lower levels keep the alignment of their parent.
            if (range_start &&                          //
                divisions > 1 && divisions % 2 != 0 &&  //
                params.min_rpy[axis] < 0.0f && params.max_rpy[axis] > 0.0f &&
                divisions < std::numeric_limits<int>::max()) {
                ++divisions;
            }
            info[level].num_division[axis] = divisions;
            // 3D-BBS scores an undivided level at the zero angle, not at the cell center.
            info[level].rpy_res[axis] = divisions == 1 ? 0.0f : piece / divisions;
            info[level].min_rpy[axis] = divisions == 1 ? 0.0f : params.min_rpy[axis];
        }
    }
    return info;
}

struct DiscreteTransformation {
    uint32_t score = 0;
    int level = 0;
    int x = 0, y = 0, z = 0;
    int roll = 0, pitch = 0, yaw = 0;

    bool operator<(const DiscreteTransformation& other) const { return this->score < other.score; }
    bool is_leaf() const { return this->level == 0; }

    TransformMatrix create_matrix(float res, const Eigen::Vector3f& rpy_res, const Eigen::Vector3f& min_rpy) const {
        return (Eigen::Translation3f(this->x * res, this->y * res, this->z * res) *
                Eigen::AngleAxisf(this->yaw * rpy_res.z() + min_rpy.z(), Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(this->pitch * rpy_res.y() + min_rpy.y(), Eigen::Vector3f::UnitY()) *
                Eigen::AngleAxisf(this->roll * rpy_res.x() + min_rpy.x(), Eigen::Vector3f::UnitX()))
            .matrix();
    }

    void branch(std::vector<DiscreteTransformation>& output, int child_level, int factor,
                const Eigen::Vector3i& divisions) const {
        for (int i = 0; i < factor; ++i) {
            for (int j = 0; j < factor; ++j) {
                for (int k = 0; k < factor; ++k) {
                    for (int r = 0; r < divisions.x(); ++r) {
                        for (int p = 0; p < divisions.y(); ++p) {
                            for (int yv = 0; yv < divisions.z(); ++yv) {
                                output.push_back({0, child_level,
                                                  checked_int(static_cast<int64_t>(this->x) * factor + i),
                                                  checked_int(static_cast<int64_t>(this->y) * factor + j),
                                                  checked_int(static_cast<int64_t>(this->z) * factor + k),
                                                  checked_int(static_cast<int64_t>(this->roll) * divisions.x() + r),
                                                  checked_int(static_cast<int64_t>(this->pitch) * divisions.y() + p),
                                                  checked_int(static_cast<int64_t>(this->yaw) * divisions.z() + yv)});
                            }
                        }
                    }
                }
            }
        }
    }
};

}  // namespace sycl_points::algorithms::localization::detail
