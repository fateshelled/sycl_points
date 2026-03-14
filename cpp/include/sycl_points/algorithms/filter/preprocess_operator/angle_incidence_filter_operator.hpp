#pragma once

#include <cmath>
#include <stdexcept>

#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/filter/preprocess_operator/common.hpp"
#include "sycl_points/algorithms/filter/preprocess_operator/preprocess_operator_base.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace filter {
namespace preprocess_operator {

class AngleIncidenceFilterOperator : public PreprocessOperatorBase {
public:
    AngleIncidenceFilterOperator(const sycl_utils::DeviceQueue& queue, shared_vector_ptr<uint8_t> flags,
                                 InitializeFlagsFn initialize_flags, FilterByFlagsFn filter_by_flags)
        : PreprocessOperatorBase(queue, std::move(flags), std::move(initialize_flags), std::move(filter_by_flags)) {}

    void apply(const PointCloudShared& source, PointCloudShared& output, float min_angle, float max_angle) const {
        const size_t N = source.size();
        if (N == 0) return;

        if (!source.has_normal() && !source.has_cov()) {
            throw std::runtime_error(
                "[PreprocessFilter::angle_incidence_filter] Normal vector or covariance matrices must be "
                "pre-computed.");
        }
        if (min_angle < 0.0f || max_angle > M_PIf * 0.5f || min_angle >= max_angle) {
            throw std::invalid_argument("[PreprocessFilter::angle_incidence_filter] Invalid angle range");
        }

        this->initialize_flags_(N, INCLUDE_FLAG).wait_and_throw();

        {
            this->queue_.set_accessed_by_device(this->flags_->data(), N);
            this->queue_.set_accessed_by_device(source.points_ptr(), N);
            if (source.has_normal()) {
                this->queue_.set_accessed_by_device(source.normals_ptr(), N);
            } else {
                this->queue_.set_accessed_by_device(source.covs_ptr(), N);
            }
        }

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t work_group_size = this->queue_.get_work_group_size();
            const size_t global_size = this->queue_.get_global_size(N);
            const auto point_ptr = source.points_ptr();
            const auto cov_ptr = source.covs_ptr();
            const auto normal_ptr = source.normals_ptr();
            const auto flag_ptr = this->flags_->data();
            const auto max_cos = std::cos(min_angle);
            const auto min_cos = std::cos(max_angle);

            auto compute_flag = [=](const PointType& pt, const Normal& normal, uint8_t& flag) {
                const float dot = eigen_utils::dot<3>(pt.head<3>(), normal.head<3>());
                const float denom =
                    eigen_utils::frobenius_norm<3>(pt.head<3>()) * eigen_utils::frobenius_norm<3>(normal.head<3>());

                if (denom <= 1e-6f) {
                    flag = REMOVE_FLAG;
                    return;
                }
                const float abs_cos = sycl::fabs(dot / denom);
                if (abs_cos < min_cos || abs_cos > max_cos) {
                    flag = REMOVE_FLAG;
                }
            };

            if (source.has_normal()) {
                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_global_id(0);
                    if (i >= N) return;
                    if (!kernel::is_finite(point_ptr[i])) {
                        flag_ptr[i] = REMOVE_FLAG;
                        return;
                    }
                    compute_flag(point_ptr[i], normal_ptr[i], flag_ptr[i]);
                });
            } else {
                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_global_id(0);
                    if (i >= N) return;
                    if (!kernel::is_finite(point_ptr[i])) {
                        flag_ptr[i] = REMOVE_FLAG;
                        return;
                    }
                    Normal normal;
                    algorithms::covariance::kernel::compute_normal_from_covariance(point_ptr[i], cov_ptr[i], normal);
                    compute_flag(point_ptr[i], normal, flag_ptr[i]);
                });
            }
        });
        event.wait_and_throw();

        {
            this->queue_.clear_accessed_by_device(this->flags_->data(), N);
            this->queue_.clear_accessed_by_device(source.points_ptr(), N);
            if (source.has_normal()) {
                this->queue_.clear_accessed_by_device(source.normals_ptr(), N);
            } else {
                this->queue_.clear_accessed_by_device(source.covs_ptr(), N);
            }
        }

        this->filter_by_flags_(source, output);
    }
};

}  // namespace preprocess_operator
}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
