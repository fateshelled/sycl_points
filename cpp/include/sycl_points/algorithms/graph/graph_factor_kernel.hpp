#pragma once

#include <algorithm>
#include <memory>
#include <tuple>

#include "sycl_points/algorithms/common/transform.hpp"
#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/algorithms/registration/factor.hpp"
#include "sycl_points/algorithms/registration/registration_params.hpp"
#include "sycl_points/algorithms/robust/robust.hpp"
#include "sycl_points/algorithms/graph/pose_node.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace graph {

/// @brief Per-point binary GICP linearization result (both nodes variable).
struct BinaryLinearizedKernelResult {
    Eigen::Matrix<float, 6, 6> H00 = Eigen::Matrix<float, 6, 6>::Zero();  // source-source
    Eigen::Matrix<float, 6, 6> H01 = Eigen::Matrix<float, 6, 6>::Zero();  // source-target
    Eigen::Matrix<float, 6, 6> H11 = Eigen::Matrix<float, 6, 6>::Zero();  // target-target
    Eigen::Matrix<float, 6, 1> b0 = Eigen::Matrix<float, 6, 1>::Zero();   // source
    Eigen::Matrix<float, 6, 1> b1 = Eigen::Matrix<float, 6, 1>::Zero();   // target
    float squared_error = 0.0f;
    uint32_t inlier = 0;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Per-point binary GICP linearization.
///
/// Residual:  r = T_tgt * tgt - T_src * src   (world frame, 4D with w=0)
/// Jacobian:  J0 = d(T_src*src)/dδ_src,  J1 = d(T_tgt*tgt)/dδ_tgt
/// Because ∂r/∂δ_src = -J0 and ∂r/∂δ_tgt = +J1, the binary blocks accumulate as:
///   H00 = J0ᵀ Ω J0,  H11 = J1ᵀ Ω J1,  H01 = -J0ᵀ Ω J1
///   b0  = +J0ᵀ Ω r,  b1  = -J1ᵀ Ω r      (so the solver solves H δ = -b)
/// where Ω is the GICP Mahalanobis (src_cov_w + tgt_cov_w)⁻¹.
SYCL_EXTERNAL inline BinaryLinearizedKernelResult linearize_gicp_binary(
    const std::array<sycl::float4, 4>& T_src, const std::array<sycl::float4, 4>& T_tgt,
    const PointType& src_pt, const Covariance& src_cov, const PointType& tgt_pt, const Covariance& tgt_cov) {
    PointType src_world;
    transform::kernel::transform_point(src_pt, src_world, T_src);
    PointType tgt_world;
    transform::kernel::transform_point(tgt_pt, tgt_world, T_tgt);

    const PointType residual(tgt_world.x() - src_world.x(), tgt_world.y() - src_world.y(),
                             tgt_world.z() - src_world.z(), 0.0f);

    Covariance src_cov_w;
    transform::kernel::transform_covs(src_cov, src_cov_w, T_src);
    Covariance tgt_cov_w;
    transform::kernel::transform_covs(tgt_cov, tgt_cov_w, T_tgt);

    Covariance mahalanobis = Covariance::Zero();
    mahalanobis.block<3, 3>(0, 0) =
        eigen_utils::add<3, 3>(src_cov_w.block<3, 3>(0, 0), tgt_cov_w.block<3, 3>(0, 0));
    const Covariance mah_inv = covariance::kernel::inverse(mahalanobis);

    const Eigen::Matrix<float, 4, 6> J0 = registration::kernel::compute_se3_jacobian(T_src, src_pt);
    const Eigen::Matrix<float, 4, 6> J1 = registration::kernel::compute_se3_jacobian(T_tgt, tgt_pt);

    const Eigen::Matrix<float, 4, 6> J0w = registration::kernel::apply_weight_to_jacobian(J0, mah_inv);
    const Eigen::Matrix<float, 4, 6> J1w = registration::kernel::apply_weight_to_jacobian(J1, mah_inv);

    BinaryLinearizedKernelResult ret;
    // H00 = J0ᵀ Ω J0
    ret.H00 = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(eigen_utils::transpose<4, 6>(J0w), J0));
    // H11 = J1ᵀ Ω J1
    ret.H11 = eigen_utils::ensure_symmetric<6>(eigen_utils::multiply<6, 4, 6>(eigen_utils::transpose<4, 6>(J1w), J1));
    // H01 = -J0ᵀ Ω J1  (avoid native unary minus on fixed-size matrices in device code)
    ret.H01 = eigen_utils::multiply<6, 6>(
        eigen_utils::multiply<6, 4, 6>(eigen_utils::transpose<4, 6>(J0w), J1), -1.0f);
    // b0 = +J0ᵀ Ω r
    ret.b0 = eigen_utils::multiply<6, 4>(eigen_utils::transpose<4, 6>(J0w), residual);
    // b1 = -J1ᵀ Ω r  (avoid native unary minus on fixed-size matrices in device code)
    ret.b1 = eigen_utils::multiply<6>(
        eigen_utils::multiply<6, 4>(eigen_utils::transpose<4, 6>(J1w), residual), -1.0f);

    const float squared_norm = eigen_utils::dot<4>(residual, eigen_utils::multiply<4, 4>(mah_inv, residual));
    ret.squared_error = squared_norm;
    ret.inlier = 1;
    return ret;
}

/// @brief Device accumulation buffers for the binary linearization reduction.
namespace {
struct BinaryLinearizedDevice {
    sycl::float16* H00_0 = nullptr;
    sycl::float16* H00_1 = nullptr;
    sycl::float4* H00_2 = nullptr;
    sycl::float16* H01_0 = nullptr;
    sycl::float16* H01_1 = nullptr;
    sycl::float4* H01_2 = nullptr;
    sycl::float16* H11_0 = nullptr;
    sycl::float16* H11_1 = nullptr;
    sycl::float4* H11_2 = nullptr;
    sycl::float3* b0_0 = nullptr;
    sycl::float3* b0_1 = nullptr;
    sycl::float3* b1_0 = nullptr;
    sycl::float3* b1_1 = nullptr;
    float* error = nullptr;
    uint32_t* inlier = nullptr;
    size_t size;
    sycl_utils::DeviceQueue queue;

    BinaryLinearizedDevice(const sycl_utils::DeviceQueue& q, size_t N = 1) : size(N), queue(q) {
        auto alloc_f16 = [&](sycl::float16*& p) { p = sycl::malloc_shared<sycl::float16>(size, *queue.ptr); };
        auto alloc_f4 = [&](sycl::float4*& p) { p = sycl::malloc_shared<sycl::float4>(size, *queue.ptr); };
        auto alloc_f3 = [&](sycl::float3*& p) { p = sycl::malloc_shared<sycl::float3>(size, *queue.ptr); };
        alloc_f16(H00_0); alloc_f16(H00_1); alloc_f4(H00_2);
        alloc_f16(H01_0); alloc_f16(H01_1); alloc_f4(H01_2);
        alloc_f16(H11_0); alloc_f16(H11_1); alloc_f4(H11_2);
        alloc_f3(b0_0); alloc_f3(b0_1); alloc_f3(b1_0); alloc_f3(b1_1);
        error = sycl::malloc_shared<float>(size, *queue.ptr);
        inlier = sycl::malloc_shared<uint32_t>(size, *queue.ptr);
    }
    ~BinaryLinearizedDevice() {
        auto free_f16 = [&](sycl::float16* p) { sycl_utils::free(p, *queue.ptr); };
        auto free_f4 = [&](sycl::float4* p) { sycl_utils::free(p, *queue.ptr); };
        auto free_f3 = [&](sycl::float3* p) { sycl_utils::free(p, *queue.ptr); };
        free_f16(H00_0); free_f16(H00_1); free_f4(H00_2);
        free_f16(H01_0); free_f16(H01_1); free_f4(H01_2);
        free_f16(H11_0); free_f16(H11_1); free_f4(H11_2);
        free_f3(b0_0); free_f3(b0_1); free_f3(b1_0); free_f3(b1_1);
        sycl_utils::free(error, *queue.ptr);
        sycl_utils::free(inlier, *queue.ptr);
    }
    void setZero() {
        for (size_t n = 0; n < size; ++n) {
            H00_0[n] = H00_1[n] = H01_0[n] = H01_1[n] = H11_0[n] = H11_1[n] = sycl::float16();
            H00_2[n] = H01_2[n] = H11_2[n] = sycl::float4();
            b0_0[n] = b0_1[n] = b1_0[n] = b1_1[n] = sycl::float3();
            error[n] = 0.0f;
            inlier[n] = 0;
        }
    }
    FactorLinearization toCPU(size_t i = 0) const {
        FactorLinearization ret;
        ret.H00 = eigen_utils::from_sycl_vec({H00_0[i], H00_1[i], H00_2[i]});
        ret.H01 = eigen_utils::from_sycl_vec({H01_0[i], H01_1[i], H01_2[i]});
        ret.H11 = eigen_utils::from_sycl_vec({H11_0[i], H11_1[i], H11_2[i]});
        ret.b0 = eigen_utils::from_sycl_vec({b0_0[i], b0_1[i]});
        ret.b1 = eigen_utils::from_sycl_vec({b1_0[i], b1_1[i]});
        ret.error = error[i];
        ret.inlier = inlier[i];
        return ret;
    }
};
}  // namespace

/// @brief Computes the binary GICP factor linearization (both endpoints variable)
///        via a SYCL parallel reduction, mirroring Registration's reduction.
class BinaryGicpLinearizer {
public:
    BinaryGicpLinearizer(const sycl_utils::DeviceQueue& queue,
                         const registration::RegistrationParams& params = registration::RegistrationParams())
        : queue_(queue), params_(params) {
        this->neighbors_ = std::make_shared<shared_vector<knn::KNNResult>>(1, knn::KNNResult(), *this->queue_.ptr);
        this->neighbors_->at(0).allocate(this->queue_, 1, 1);
        this->device_ = std::make_shared<BinaryLinearizedDevice>(this->queue_);
    }

    FactorLinearization linearize(const PointCloudShared& source, const knn::KNNBase& target_knn,
                                  const Eigen::Matrix4f& T_src, const PointCloudShared& target,
                                  const Eigen::Matrix4f& T_tgt) const {
        const float robust_scale = this->params_.robust.default_scale;
        const auto T_search_mat = Eigen::Isometry3f(Eigen::Isometry3f(Eigen::Matrix4f(T_tgt).inverse()) *
                                                     Eigen::Matrix4f(T_src))
                                      .matrix();
        const auto T_search = eigen_utils::to_sycl_vec(T_search_mat);

        auto knn_event = target_knn.nearest_neighbor_search_async(source, (*this->neighbors_)[0], {}, T_search_mat);

        auto events = this->dispatch([&]<registration::RegType reg, robust::RobustLossType loss>() {
            return this->linearize_async<reg, loss>(source, target, T_src, T_tgt, robust_scale, knn_event.evs);
        });
        events.wait_and_throw();
        return this->device_->toCPU(0);
    }

private:
    template <registration::RegType reg, robust::RobustLossType loss>
    sycl_utils::events linearize_async(const PointCloudShared& source, const PointCloudShared& target,
                                       const Eigen::Matrix4f& T_src, const Eigen::Matrix4f& T_tgt,
                                       float robust_scale, const std::vector<sycl::event>& depends) const {
        sycl_utils::events events;
        events += this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t N = source.size();
            // SYCL parallel_reduction requires a work-group size <= 64 on the
            // CPU OpenCL backend; clamp for portability (64 is valid everywhere).
            const size_t work_group_size =
                std::min(this->queue_.get_work_group_size_for_parallel_reduction(), size_t{64});
            const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;

            const auto T_src_v = eigen_utils::to_sycl_vec(T_src);
            const auto T_tgt_v = eigen_utils::to_sycl_vec(T_tgt);

            const auto source_ptr = source.points_ptr();
            const auto source_cov_ptr = source.has_cov() ? source.covs_ptr() : nullptr;
            const auto target_ptr = target.points_ptr();
            const auto target_cov_ptr = target.has_cov() ? target.covs_ptr() : nullptr;

            const auto neighbors_index_ptr = (*this->neighbors_)[0].indices->data();
            const auto neighbors_distances_ptr = (*this->neighbors_)[0].distances->data();

            const float max_corr_dist_squared =
                this->params_.max_correspondence_distance * this->params_.max_correspondence_distance;

            this->device_->setZero();
            auto sum_H00_0 = sycl::reduction(this->device_->H00_0, sycl::plus<sycl::float16>());
            auto sum_H00_1 = sycl::reduction(this->device_->H00_1, sycl::plus<sycl::float16>());
            auto sum_H00_2 = sycl::reduction(this->device_->H00_2, sycl::plus<sycl::float4>());
            auto sum_H01_0 = sycl::reduction(this->device_->H01_0, sycl::plus<sycl::float16>());
            auto sum_H01_1 = sycl::reduction(this->device_->H01_1, sycl::plus<sycl::float16>());
            auto sum_H01_2 = sycl::reduction(this->device_->H01_2, sycl::plus<sycl::float4>());
            auto sum_H11_0 = sycl::reduction(this->device_->H11_0, sycl::plus<sycl::float16>());
            auto sum_H11_1 = sycl::reduction(this->device_->H11_1, sycl::plus<sycl::float16>());
            auto sum_H11_2 = sycl::reduction(this->device_->H11_2, sycl::plus<sycl::float4>());
            auto sum_b0_0 = sycl::reduction(this->device_->b0_0, sycl::plus<sycl::float3>());
            auto sum_b0_1 = sycl::reduction(this->device_->b0_1, sycl::plus<sycl::float3>());
            auto sum_b1_0 = sycl::reduction(this->device_->b1_0, sycl::plus<sycl::float3>());
            auto sum_b1_1 = sycl::reduction(this->device_->b1_1, sycl::plus<sycl::float3>());
            auto sum_error = sycl::reduction(this->device_->error, sycl::plus<float>());
            auto sum_inlier = sycl::reduction(this->device_->inlier, sycl::plus<uint32_t>());

            h.depends_on(depends);
            h.parallel_for(  //
                sycl::nd_range<1>(global_size, work_group_size),
                sum_H00_0, sum_H00_1, sum_H00_2, sum_H01_0, sum_H01_1, sum_H01_2, sum_H11_0, sum_H11_1, sum_H11_2,
                sum_b0_0, sum_b0_1, sum_b1_0, sum_b1_1, sum_error, sum_inlier,
                [=](sycl::nd_item<1> item, auto& aH00_0, auto& aH00_1, auto& aH00_2, auto& aH01_0, auto& aH01_1,
                    auto& aH01_2, auto& aH11_0, auto& aH11_1, auto& aH11_2, auto& ab0_0, auto& ab0_1, auto& ab1_0,
                    auto& ab1_1, auto& aerror, auto& ainlier) {
                    const size_t index = item.get_global_id(0);
                    if (index >= N) return;
                    if (neighbors_distances_ptr[index] > max_corr_dist_squared) return;

                    const auto tgt_idx = neighbors_index_ptr[index];
                    const auto src_cov = source_cov_ptr ? source_cov_ptr[index] : Covariance::Identity();
                    const auto tgt_cov = target_cov_ptr ? target_cov_ptr[tgt_idx] : Covariance::Identity();

                    float residual_norm = 0.0f;
                    auto lin = linearize_gicp_binary(T_src_v, T_tgt_v, source_ptr[index], src_cov,
                                                    target_ptr[tgt_idx], tgt_cov);
                    residual_norm = sycl::sqrt(lin.squared_error);

                    const float weight = robust::kernel::compute_weight<loss>(residual_norm, robust_scale);

                    const auto [lH00_0, lH00_1, lH00_2] = eigen_utils::to_sycl_vec(lin.H00);
                    const auto [lH01_0, lH01_1, lH01_2] = eigen_utils::to_sycl_vec(lin.H01);
                    const auto [lH11_0, lH11_1, lH11_2] = eigen_utils::to_sycl_vec(lin.H11);
                    const auto [lb0_0, lb0_1] = eigen_utils::to_sycl_vec(lin.b0);
                    const auto [lb1_0, lb1_1] = eigen_utils::to_sycl_vec(lin.b1);

                    aH00_0 += weight * lH00_0;
                    aH00_1 += weight * lH00_1;
                    aH00_2 += weight * lH00_2;
                    aH01_0 += weight * lH01_0;
                    aH01_1 += weight * lH01_1;
                    aH01_2 += weight * lH01_2;
                    aH11_0 += weight * lH11_0;
                    aH11_1 += weight * lH11_1;
                    aH11_2 += weight * lH11_2;
                    ab0_0 += weight * lb0_0;
                    ab0_1 += weight * lb0_1;
                    ab1_0 += weight * lb1_0;
                    ab1_1 += weight * lb1_1;
                    aerror += robust::kernel::compute_error<loss>(residual_norm, robust_scale);
                    ++ainlier;
                });
        });
        return events;
    }

    template <typename Func>
    sycl_utils::events dispatch(Func&& exec) const {
        sycl_utils::events events;
        auto dispatch_inner = [&]<typename RobustLossTypeTags, size_t... Js>(robust::RobustLossType loss,
                                                                            std::index_sequence<Js...>) {
            return (((loss == std::tuple_element_t<Js, RobustLossTypeTags>::value)
                         ? (events += exec.template operator()<registration::RegType::GICP,
                                                             std::tuple_element_t<Js, RobustLossTypeTags>::value>(),
                            true)
                         : false) ||
                    ...);
        };
        auto found = dispatch_inner.template operator()<robust::RobustLossTypeTags>(
            this->params_.robust.type, std::make_index_sequence<std::tuple_size_v<robust::RobustLossTypeTags>>());
        if (!found) {
            throw std::runtime_error("[BinaryGicpLinearizer::dispatch] robust type not found");
        }
        return events;
    }

    sycl_utils::DeviceQueue queue_;
    registration::RegistrationParams params_;
    shared_vector_ptr<knn::KNNResult> neighbors_ = nullptr;
    std::shared_ptr<BinaryLinearizedDevice> device_ = nullptr;
};

}  // namespace graph
}  // namespace algorithms
}  // namespace sycl_points
