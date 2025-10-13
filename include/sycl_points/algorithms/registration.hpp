#pragma once

#include <random>
#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/algorithms/knn_search.hpp>
#include <sycl_points/algorithms/registration_factor.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {

namespace algorithms {

namespace registration {

struct RegistrationParams {
    struct Criteria {
        float translation = 1e-3f;  // translation tolerance
        float rotation = 1e-3f;     // rotation tolerance [rad]
    };
    struct Robust {
        RobustLossType type = RobustLossType::NONE;  // robust loss function type
        float scale = 1.0f;                          // scale for robust loss function
    };
    struct PhotometricTerm {
        bool enable = false;              // If true, use photometric term.
        float photometric_weight = 0.2f;  // weight for photometric term (0.0f ~ 1.0f)
    };
    struct LevenbergMarquardt {
        bool enable = false;               // If true, use Levenberg-Marquardt method, else use Gauss-Newton method.
        size_t max_inner_iterations = 10;  // (for LM method)
        float lambda_factor = 10.0f;       // lambda increase factor (for LM method)
    };

    size_t max_iterations = 20;                // max iteration
    float lambda = 1e-6f;                      // damping factor
    float max_correspondence_distance = 2.0f;  // max correspondence distance

    Criteria crireria;
    Robust robust;
    PhotometricTerm photometric;
    LevenbergMarquardt lm;

    bool verbose = false;  // If true, print debug messages
};

namespace {

/// @brief Device copyable linealized result
struct LinearlizedDevice {
    // H is 6x6 -> 16 + 16 + 4
    sycl::float16* H0 = nullptr;  // H(0, 0) ~ H(2, 3)
    sycl::float16* H1 = nullptr;  // H(2, 4) ~ H(5, 1)
    sycl::float4* H2 = nullptr;   // H(5, 2) ~ H(5, 5)
    // b is 6x1 -> 3 + 3
    sycl::float3* b0 = nullptr;  // b(0) ~ b(2)
    sycl::float3* b1 = nullptr;  // b(3) ~ b(5)
    // error is scalar
    float* error = nullptr;
    // inlier point num
    uint32_t* inlier = nullptr;
    size_t size;

    sycl_utils::DeviceQueue queue;

    LinearlizedDevice(const sycl_utils::DeviceQueue& q, size_t N = 1) : queue(q), size(N) {
        H0 = sycl::malloc_shared<sycl::float16>(size, *this->queue.ptr);
        H1 = sycl::malloc_shared<sycl::float16>(size, *this->queue.ptr);
        H2 = sycl::malloc_shared<sycl::float4>(size, *this->queue.ptr);
        b0 = sycl::malloc_shared<sycl::float3>(size, *this->queue.ptr);
        b1 = sycl::malloc_shared<sycl::float3>(size, *this->queue.ptr);
        error = sycl::malloc_shared<float>(size, *this->queue.ptr);
        inlier = sycl::malloc_shared<uint32_t>(size, *this->queue.ptr);
    }
    ~LinearlizedDevice() {
        sycl_utils::free(H0, *this->queue.ptr);
        sycl_utils::free(H1, *this->queue.ptr);
        sycl_utils::free(H2, *this->queue.ptr);
        sycl_utils::free(b0, *this->queue.ptr);
        sycl_utils::free(b1, *this->queue.ptr);
        sycl_utils::free(error, *this->queue.ptr);
        sycl_utils::free(inlier, *this->queue.ptr);
    }
    void setZero() {
        for (size_t n = 0; n < size; ++n) {
            for (size_t i = 0; i < 16; ++i) {
                H0[n][i] = 0.0f;
                H1[n][i] = 0.0f;
            }
            for (size_t i = 0; i < 4; ++i) {
                H2[n][i] = 0.0f;
            }
            for (size_t i = 0; i < 3; ++i) {
                b0[n][i] = 0.0f;
                b1[n][i] = 0.0f;
            }
            error[n] = 0.0f;
            inlier[n] = 0;
        }
    }
    LinearlizedResult toCPU(size_t i = 0) {
        const LinearlizedResult result = {.H = eigen_utils::from_sycl_vec({this->H0[i], this->H1[i], this->H2[i]}),
                                          .b = eigen_utils::from_sycl_vec({this->b0[i], this->b1[i]}),
                                          .error = this->error[i],
                                          .inlier = this->inlier[i]};
        return result;
    }
};
}  // namespace

/// @brief Point cloud registration
/// @tparam icp icp type
template <ICPType icp = ICPType::GICP>
class Registration {
public:
    using Ptr = std::shared_ptr<Registration<icp>>;

    /// @brief Constructor
    /// @param queue SYCL queue
    /// @param params Registration parameters
    Registration(const sycl_utils::DeviceQueue& queue, const RegistrationParams& params = RegistrationParams())
        : params_(params), queue_(queue) {
        this->neighbors_ = std::make_shared<shared_vector<knn_search::KNNResult>>(
            1, knn_search::KNNResult(), shared_allocator<knn_search::KNNResult>(*this->queue_.ptr));
        this->neighbors_->at(0).allocate(this->queue_, 1, 1);

        this->aligned_ = std::make_shared<PointCloudShared>(this->queue_);
        this->linearlized_on_device_ = std::make_shared<LinearlizedDevice>(this->queue_);
        this->linearlized_on_host_ = std::make_shared<shared_vector<LinearlizedResult>>(
            1, LinearlizedResult(), shared_allocator<LinearlizedResult>(*this->queue_.ptr));
        this->error_on_host_ = std::make_shared<shared_vector<float>>(*this->queue_.ptr);
        this->inlier_on_host_ = std::make_shared<shared_vector<uint32_t>>(*this->queue_.ptr);
    }

    /// @brief Get aligned point cloud
    /// @return aligned point cloud pointer
    PointCloudShared::Ptr get_aligned_point_cloud() const { return this->aligned_; }

    /// @brief do registration
    /// @param source Source point cloud
    /// @param target Target point cloud
    /// @param target_tree Target KDTree
    /// @param init_T Initial transformation matrix
    /// @return Registration result
    RegistrationResult align(const PointCloudShared& source, const PointCloudShared& target,
                             const knn_search::KDTree& target_tree,
                             const TransformMatrix& init_T = TransformMatrix::Identity()) {
        const size_t N = source.size();
        const size_t TARGET_SIZE = target.size();
        RegistrationResult result;
        result.T.matrix() = init_T;

        if (N == 0) return result;

        if constexpr (icp == ICPType::POINT_TO_PLANE) {
            if (!target.has_normal()) {
                if (!target.has_cov()) {
                    throw std::runtime_error(
                        "Normal vector or covariance matrices must be pre-computed before performing Point-to-Plane "
                        "ICP matching.");
                }
                std::cout << "[Caution] Normal vectors for Point-to-Plane ICP are not provided. " << std::endl;
                std::cout << "          Attempting to derive them from pre-computed covariance matrices." << std::endl;
                target.reserve_covs(target.size());
                covariance::compute_normals_from_covariances(target);
            }
        }
        if constexpr (icp == ICPType::GICP) {
            if (!source.has_cov() || !target.has_cov()) {
                throw std::runtime_error("Covariance matrices must be pre-computed before performing GICP matching.");
            }
        }
        if (this->params_.photometric.enable) {
            if (!source.has_rgb() || !target.has_rgb()) {
                throw std::runtime_error("RGB fields is required for photometric matching.");
            }
            if (!target.has_color_gradient() || !target.has_normal()) {
                throw std::runtime_error(
                    "Target color gradient and target normal vector must be pre-computed before performing "
                    "photometric matching.");
            }
            if (this->params_.photometric.photometric_weight == 0.0f) {
                std::cout << "[Caution] `photometric_weight` is set to zero. Disable photometric matching."
                          << std::endl;
                this->params_.photometric.enable = false;
            }
            if (this->params_.photometric.photometric_weight < 0.0f ||
                this->params_.photometric.photometric_weight > 1.0f) {
                std::cout << "[Caution] `photometric_weight` must be in range [0.0f, 1.0f]. Disable photometric "
                             "matching."
                          << std::endl;
                this->params_.photometric.enable = false;
            }
        }

        Eigen::Isometry3f prev_T = Eigen::Isometry3f::Identity();
        // copy
        this->aligned_ = std::make_shared<PointCloudShared>(source);

        // mem_advise to device
        {
            this->queue_.set_accessed_by_device(this->aligned_->points->data(), N);
            this->queue_.set_accessed_by_device(source.points->data(), N);
            this->queue_.set_accessed_by_device(target.points->data(), TARGET_SIZE);
        }
        // transform
        transform::transform(*this->aligned_, init_T);

        const float max_dist = this->params_.max_correspondence_distance;
        const auto verbose = this->params_.verbose;

        sycl_utils::events transform_events;
        float lambda = this->params_.lambda;
        for (size_t iter = 0; iter < this->params_.max_iterations; ++iter) {
            prev_T = result.T;

            // Nearest neighbor search on device
            auto knn_event = target_tree.nearest_neighbor_search_async(*this->aligned_, (*this->neighbors_)[0],
                                                                       transform_events.evs);

            // Linearlize on device
            const float max_dist_2 = max_dist * max_dist;
            const LinearlizedResult linearlized_result =
                this->linearlize(source, target, result.T.matrix(), max_dist_2, knn_event.evs);

            // Optimize on Host
            if (this->params_.lm.enable) {
                this->optimize_levenberg_marquardt(source, target, max_dist_2, result, linearlized_result, lambda,
                                                   iter);

            } else {
                this->optimize_gauss_newton(result, linearlized_result, lambda, iter);
            }

            // Async transform source points on device
            transform_events = transform::transform_async(*this->aligned_,
                                                          result.T.matrix() * prev_T.matrix().inverse());  // zero copy

            if (result.converged) {
                break;
            }
        }
        transform_events.wait();
        // mem_advise clear
        {
            this->queue_.clear_accessed_by_device(this->aligned_->points->data(), N);
            this->queue_.clear_accessed_by_device(source.points->data(), N);
            this->queue_.clear_accessed_by_device(target.points->data(), TARGET_SIZE);
        }

        return result;
    }

private:
    RegistrationParams params_;
    sycl_utils::DeviceQueue queue_;
    PointCloudShared::Ptr aligned_ = nullptr;
    shared_vector_ptr<knn_search::KNNResult> neighbors_ = nullptr;
    std::shared_ptr<LinearlizedDevice> linearlized_on_device_ = nullptr;
    shared_vector_ptr<LinearlizedResult> linearlized_on_host_ = nullptr;
    shared_vector_ptr<float> error_on_host_ = nullptr;
    shared_vector_ptr<uint32_t> inlier_on_host_ = nullptr;

    bool is_converged(const Eigen::Vector<float, 6>& delta) const {
        return delta.template head<3>().norm() < this->params_.crireria.rotation &&
               delta.template tail<3>().norm() < this->params_.crireria.translation;
    }

    template <RobustLossType loss = RobustLossType::NONE>
    sycl_utils::events linearlize_parallel_reduction_async(const PointCloudShared& source,
                                                           const PointCloudShared& target, const Eigen::Matrix4f transT,
                                                           float max_correspondence_distance_2,
                                                           const std::vector<sycl::event>& depends) {
        sycl_utils::events events;
        events += this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t N = source.size();

            const size_t work_group_size = this->queue_.get_work_group_size_for_parallel_reduction();
            const size_t global_size = this->queue_.get_global_size_for_parallel_reduction(N);

            // convert to sycl::float4
            const auto cur_T = eigen_utils::to_sycl_vec(transT);

            const auto robust_scale = this->params_.robust.scale;

            // get pointers
            // input
            const auto source_ptr = source.points_ptr();
            const auto source_cov_ptr = source.has_cov() ? source.covs_ptr() : nullptr;
            const auto target_ptr = target.points_ptr();
            const auto target_cov_ptr = target.has_cov() ? target.covs_ptr() : nullptr;
            const auto target_normal_ptr = target.has_normal() ? target.normals_ptr() : nullptr;
            const auto source_rgb_ptr = source.has_rgb() ? source.rgb_ptr() : nullptr;
            const auto target_rgb_ptr = target.has_rgb() ? target.rgb_ptr() : nullptr;
            const auto target_grad_ptr = target.has_color_gradient() ? target.color_gradients_ptr() : nullptr;
            const float photometric_weight =
                this->params_.photometric.enable ? this->params_.photometric.photometric_weight : 0.0f;
            const auto neighbors_index_ptr = (*this->neighbors_)[0].indices->data();
            const auto neighbors_distances_ptr = (*this->neighbors_)[0].distances->data();

            // reduction
            this->linearlized_on_device_->setZero();
            auto sum_H0 = sycl::reduction(this->linearlized_on_device_->H0, sycl::plus<sycl::float16>());
            auto sum_H1 = sycl::reduction(this->linearlized_on_device_->H1, sycl::plus<sycl::float16>());
            auto sum_H2 = sycl::reduction(this->linearlized_on_device_->H2, sycl::plus<sycl::float4>());
            auto sum_b0 = sycl::reduction(this->linearlized_on_device_->b0, sycl::plus<sycl::float3>());
            auto sum_b1 = sycl::reduction(this->linearlized_on_device_->b1, sycl::plus<sycl::float3>());
            auto sum_error = sycl::reduction(this->linearlized_on_device_->error, sycl::plus<float>());
            auto sum_inlier = sycl::reduction(this->linearlized_on_device_->inlier, sycl::plus<uint32_t>());

            // wait for knn search
            h.depends_on(depends);

            h.parallel_for(                                                     //
                sycl::nd_range<1>(global_size, work_group_size),                // range
                sum_H0, sum_H1, sum_H2, sum_b0, sum_b1, sum_error, sum_inlier,  // reduction
                [=](sycl::nd_item<1> item, auto& sum_H0_arg, auto& sum_H1_arg, auto& sum_H2_arg, auto& sum_b0_arg,
                    auto& sum_b1_arg, auto& sum_error_arg, auto& sum_inlier_arg) {
                    const size_t index = item.get_global_id(0);
                    if (index >= N) return;

                    if (neighbors_distances_ptr[index] > max_correspondence_distance_2) {
                        return;
                    }
                    const auto target_idx = neighbors_index_ptr[index];
                    const auto source_cov = source_cov_ptr ? source_cov_ptr[index] : Covariance::Identity();
                    const auto target_cov = target_cov_ptr ? target_cov_ptr[target_idx] : Covariance::Identity();
                    const auto target_normal = target_normal_ptr ? target_normal_ptr[target_idx] : Normal::Zero();
                    const auto source_rgb = source_rgb_ptr ? source_rgb_ptr[index] : RGBType::Zero();
                    const auto target_rgb = target_rgb_ptr ? target_rgb_ptr[target_idx] : RGBType::Zero();
                    const auto target_grad = target_grad_ptr ? target_grad_ptr[target_idx] : ColorGradient::Zero();

                    const LinearlizedResult linearlized =
                        kernel::linearlize<icp>(cur_T, source_ptr[index], source_cov,               //
                                                target_ptr[target_idx], target_cov, target_normal,  //
                                                source_rgb, target_rgb, target_grad, photometric_weight);
                    if (linearlized.inlier == 1U) {
                        const float robust_weight =
                            kernel::compute_robust_weight<loss>(linearlized.error, robust_scale);

                        // reduction on device
                        const auto& [H0, H1, H2] = eigen_utils::to_sycl_vec(linearlized.H);
                        const auto& [b0, b1] = eigen_utils::to_sycl_vec(linearlized.b);
                        sum_H0_arg += H0 * robust_weight;
                        sum_H1_arg += H1 * robust_weight;
                        sum_H2_arg += H2 * robust_weight;
                        sum_b0_arg += b0 * robust_weight;
                        sum_b1_arg += b1 * robust_weight;
                        sum_error_arg += kernel::compute_robust_error<loss>(linearlized.error, robust_weight);
                        ++sum_inlier_arg;
                    }
                });
        });
        return events;
    }

    LinearlizedResult linearlize(const PointCloudShared& source, const PointCloudShared& target,
                                 const Eigen::Matrix4f transT, float max_correspondence_distance_2,
                                 const std::vector<sycl::event>& depends) {
        sycl_utils::events events;
        if (this->params_.robust.type == RobustLossType::NONE) {
            events += this->linearlize_parallel_reduction_async<RobustLossType::NONE>(
                source, target, transT, max_correspondence_distance_2, depends);
        } else if (this->params_.robust.type == RobustLossType::HUBER) {
            events += this->linearlize_parallel_reduction_async<RobustLossType::HUBER>(
                source, target, transT, max_correspondence_distance_2, depends);
        } else if (this->params_.robust.type == RobustLossType::TUKEY) {
            events += this->linearlize_parallel_reduction_async<RobustLossType::TUKEY>(
                source, target, transT, max_correspondence_distance_2, depends);
        } else if (this->params_.robust.type == RobustLossType::CAUCHY) {
            events += this->linearlize_parallel_reduction_async<RobustLossType::CAUCHY>(
                source, target, transT, max_correspondence_distance_2, depends);
        } else if (this->params_.robust.type == RobustLossType::GEMAN_MCCLURE) {
            events += this->linearlize_parallel_reduction_async<RobustLossType::GEMAN_MCCLURE>(
                source, target, transT, max_correspondence_distance_2, depends);
        } else {
            throw std::runtime_error("Unknown robust loss type.");
        }
        events.wait();
        return this->linearlized_on_device_->toCPU(0);
    }

    template <RobustLossType loss = RobustLossType::NONE>
    std::tuple<float, uint32_t> compute_error_parallel_reduction(const PointCloudShared& source,
                                                                 const PointCloudShared& target,
                                                                 const knn_search::KNNResult& knn_results,
                                                                 const Eigen::Matrix4f transT,
                                                                 float max_correspondence_distance_2) {
        shared_vector<float> error(1, 0.0f, shared_allocator<float>(*this->queue_.ptr));
        shared_vector<uint32_t> inlier(1, 0, shared_allocator<uint32_t>(*this->queue_.ptr));

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t N = source.size();

            const size_t work_group_size = this->queue_.get_work_group_size_for_parallel_reduction();
            const size_t global_size = this->queue_.get_global_size_for_parallel_reduction(N);

            // convert to sycl::float4
            const auto cur_T = eigen_utils::to_sycl_vec(transT);

            const auto robust_scale = this->params_.robust.scale;

            // get pointers
            // input
            const auto source_ptr = source.points_ptr();
            const auto source_cov_ptr = source.has_cov() ? source.covs_ptr() : nullptr;
            const auto source_normal_ptr = source.has_normal() ? source.normals_ptr() : nullptr;
            const auto target_ptr = target.points_ptr();
            const auto target_cov_ptr = target.has_cov() ? target.covs_ptr() : nullptr;
            const auto target_normal_ptr = target.has_normal() ? target.normals_ptr() : nullptr;
            const auto source_rgb_ptr = source.has_rgb() ? source.rgb_ptr() : nullptr;
            const auto target_rgb_ptr = target.has_rgb() ? target.rgb_ptr() : nullptr;
            const auto target_grad_ptr = target.has_color_gradient() ? target.color_gradients_ptr() : nullptr;
            const float photometric_weight =
                this->params_.photometric.enable ? this->params_.photometric.photometric_weight : 0.0f;
            const auto neighbors_index_ptr = knn_results.indices->data();
            const auto neighbors_distances_ptr = knn_results.distances->data();

            // output
            auto sum_error = sycl::reduction(error.data(), sycl::plus<float>());
            auto sum_inlier = sycl::reduction(inlier.data(), sycl::plus<uint32_t>());

            h.parallel_for(                                       //
                sycl::nd_range<1>(global_size, work_group_size),  // range
                sum_error, sum_inlier,                            // reduction
                [=](sycl::nd_item<1> item, auto& sum_error_arg, auto& sum_inlier_arg) {
                    const size_t index = item.get_global_id(0);
                    if (index >= N) return;

                    if (neighbors_distances_ptr[index] > max_correspondence_distance_2) {
                        return;
                    }
                    const auto target_idx = neighbors_index_ptr[index];
                    const auto source_cov = source_cov_ptr ? source_cov_ptr[index] : Covariance::Identity();
                    const auto target_cov = target_cov_ptr ? target_cov_ptr[target_idx] : Covariance::Identity();
                    const auto target_normal = target_normal_ptr ? target_normal_ptr[target_idx] : Normal::Zero();
                    const auto source_rgb = source_rgb_ptr ? source_rgb_ptr[index] : RGBType::Zero();
                    const auto target_rgb = target_rgb_ptr ? target_rgb_ptr[target_idx] : RGBType::Zero();
                    const auto target_grad = target_grad_ptr ? target_grad_ptr[target_idx] : ColorGradient::Zero();

                    const float err =
                        kernel::calculate_error<icp>(cur_T,                                              //
                                                     source_ptr[index], source_cov,                      // source
                                                     target_ptr[target_idx], target_cov, target_normal,  // target
                                                     source_rgb, target_rgb, target_grad, photometric_weight);

                    sum_error_arg += kernel::compute_robust_error<loss>(err, robust_scale);
                    ;
                    ++sum_inlier_arg;
                });
        });
        event.wait();
        return {error[0], inlier[0]};
    }

    std::tuple<float, uint32_t> compute_error(const PointCloudShared& source, const PointCloudShared& target,
                                              const knn_search::KNNResult& knn_results, const Eigen::Matrix4f transT,
                                              float max_correspondence_distance_2) {
        if (this->params_.robust.type == RobustLossType::NONE) {
            return this->compute_error_parallel_reduction<RobustLossType::NONE>(  //
                source, target, knn_results, transT, max_correspondence_distance_2);
        } else if (this->params_.robust.type == RobustLossType::HUBER) {
            return this->compute_error_parallel_reduction<RobustLossType::HUBER>(source, target, knn_results, transT,
                                                                                 max_correspondence_distance_2);
        } else if (this->params_.robust.type == RobustLossType::TUKEY) {
            return this->compute_error_parallel_reduction<RobustLossType::TUKEY>(source, target, knn_results, transT,
                                                                                 max_correspondence_distance_2);
        } else if (this->params_.robust.type == RobustLossType::CAUCHY) {
            return this->compute_error_parallel_reduction<RobustLossType::CAUCHY>(source, target, knn_results, transT,
                                                                                  max_correspondence_distance_2);
        } else if (this->params_.robust.type == RobustLossType::GEMAN_MCCLURE) {
            return this->compute_error_parallel_reduction<RobustLossType::GEMAN_MCCLURE>(
                source, target, knn_results, transT, max_correspondence_distance_2);
        }
        throw std::runtime_error("Unknown robust loss type.");
    }

    void optimize_gauss_newton(RegistrationResult& result, const LinearlizedResult& linearlized_result, float lambda,
                               size_t iter) {
        const Eigen::Vector<float, 6> delta = (linearlized_result.H + lambda * Eigen::Matrix<float, 6, 6>::Identity())
                                                  .ldlt()
                                                  .solve(-linearlized_result.b);
        result.converged = this->is_converged(delta);
        result.T = result.T * eigen_utils::lie::se3_exp(delta);
        result.iterations = iter;
        result.H = linearlized_result.H;
        result.b = linearlized_result.b;
        result.error = linearlized_result.error;
        result.inlier = linearlized_result.inlier;

        if (this->params_.verbose) {
            std::cout << "iter [" << iter << "] ";
            std::cout << "error: " << result.error << ", ";
            std::cout << "inlier: " << result.inlier << ", ";
            std::cout << "dt: " << delta.tail<3>().norm() << ", ";
            std::cout << "dr: " << delta.head<3>().norm() << std::endl;
        }
    }

    bool optimize_levenberg_marquardt(const PointCloudShared& source, const PointCloudShared& target,
                                      float max_correspondence_distance_2, RegistrationResult& result,
                                      const LinearlizedResult& linearlized_result, float& lambda, size_t iter) {
        bool updated = false;
        float last_error = std::numeric_limits<float>::max();

        for (size_t i = 0; i < this->params_.lm.max_inner_iterations; ++i) {
            const Eigen::Vector<float, 6> delta =
                (linearlized_result.H + lambda * Eigen::Matrix<float, 6, 6>::Identity())
                    .ldlt()
                    .solve(-linearlized_result.b);
            const Eigen::Isometry3f new_T = result.T * eigen_utils::lie::se3_exp(delta);

            const auto [new_error, inlier] =
                compute_error(source, target, this->neighbors_->at(0), new_T.matrix(), max_correspondence_distance_2);

            if (this->params_.verbose) {
                std::cout << "iter [" << iter << "] ";
                std::cout << "inner: " << i << ", ";
                std::cout << "lambda: " << lambda << ", ";
                std::cout << "error: " << new_error << ", ";
                std::cout << "inlier: " << inlier << ", ";
                std::cout << "dt: " << delta.tail<3>().norm() << ", ";
                std::cout << "dr: " << delta.head<3>().norm() << std::endl;
            }
            if (new_error <= linearlized_result.error) {
                result.converged = this->is_converged(delta);
                result.T = new_T;
                result.error = new_error;
                result.inlier = inlier;
                updated = true;

                lambda /= this->params_.lm.lambda_factor;

                break;
            } else if (std::fabs(new_error - last_error) <= 1e-6f) {
                result.converged = this->is_converged(delta);
                result.T = new_T;
                result.error = new_error;
                result.inlier = inlier;
                updated = false;

                break;
            } else {
                lambda *= this->params_.lm.lambda_factor;
            }
            last_error = new_error;
        }

        result.iterations = iter;
        result.H = linearlized_result.H;
        result.b = linearlized_result.b;
        return updated;
    }
};

using RegistrationPointToPoint = Registration<ICPType::POINT_TO_POINT>;
using RegistrationPointToPlane = Registration<ICPType::POINT_TO_PLANE>;
using RegistrationGICP = Registration<ICPType::GICP>;

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
