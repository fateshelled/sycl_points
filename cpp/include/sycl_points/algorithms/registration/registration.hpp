#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/algorithms/deskew/relative_pose_deskew.hpp>
#include <sycl_points/algorithms/knn/knn.hpp>
#include <sycl_points/algorithms/registration/degenerate_regularization.hpp>
#include <sycl_points/algorithms/registration/factor.hpp>
#include <sycl_points/algorithms/registration/linearized_result.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {

namespace algorithms {

namespace registration {

enum class OptimizationMethod {
    GAUSS_NEWTON = 0,
    LEVENBERG_MARQUARDT,
    POWELL_DOGLEG,
};

OptimizationMethod OptimizationMethod_from_string(const std::string& str) {
    std::string upper = str;
    std::transform(str.begin(), str.end(), upper.begin(), [](u_char c) { return std::toupper(c); });

    if (upper.compare("GN") == 0 || upper.compare("GAUSS_NEWTON") == 0) {
        return OptimizationMethod::GAUSS_NEWTON;
    } else if (upper.compare("LM") == 0 || upper.compare("LEVENBERG_MARQUARDT") == 0) {
        return OptimizationMethod::LEVENBERG_MARQUARDT;
    } else if (upper.compare("DOGLEG") == 0 || upper.compare("POWELL_DOGLEG") == 0) {
        return OptimizationMethod::POWELL_DOGLEG;
    }
    std::string error_str = "[OptimizationMethod_from_string] Invalid OptimizationMethod str [";
    error_str += str;
    error_str += "]";
    throw std::runtime_error(error_str);
}

struct RegistrationParams {
    struct Criteria {
        float translation = 1e-3f;  // translation tolerance
        float rotation = 1e-3f;     // rotation tolerance [rad]
    };
    struct Robust {
        RobustLossType type = RobustLossType::NONE;  // robust loss function type
        bool auto_scale = false;                     // enable auto robust scale
        float init_scale = 10.0f;                    // scale for robust loss function
        float min_scale = 0.5f;                      // minimum scale
        size_t scaling_iter = 4;                     // scaling iteration
    };
    struct PhotometricTerm {
        bool enable = false;              // If true, use photometric term.
        float photometric_weight = 0.2f;  // weight for photometric term (0.0f ~ 1.0f)
    };
    struct GenZ {
        float planarity_threshold = 0.2f;
    };
    struct LevenbergMarquardt {
        size_t max_inner_iterations = 10;  // (for LM method)
        float lambda_factor = 2.0f;        // lambda increase factor (for LM method)
        float max_lambda = 1e3f;           // max lambda (for LM method)
        float min_lambda = 1e-6f;          // min lambda (for LM method)
    };
    struct Dogleg {
        float initial_trust_region_radius = 1.0f;  // Initial trust region radius (for Powell's dogleg method)
        float min_trust_region_radius = 1e-4f;     // Minimum trust region radius (for Powell's dogleg method)
        float max_trust_region_radius = 10.0f;     // Maximum trust region radius (for Powell's dogleg method)
        float eta1 = 0.25f;                        // Lower acceptance threshold for ratio (for Powell's dogleg method)
        float eta2 = 0.75f;                        // Upper acceptance threshold for ratio (for Powell's dogleg method)
        float gamma_decrease = 0.25f;              // Shrink factor for trust region (for Powell's dogleg method)
        float gamma_increase = 2.0f;               // Expand factor for trust region (for Powell's dogleg method)
    };

    RegType reg_type = RegType::GICP;          // Registration Type
    size_t max_iterations = 20;                // max iteration
    float lambda = 1e-6f;                      // damping factor
    float max_correspondence_distance = 2.0f;  // max correspondence distance

    Criteria criteria;
    Robust robust;
    PhotometricTerm photometric;
    GenZ genz;
    LevenbergMarquardt lm;
    Dogleg dogleg;
    OptimizationMethod optimization_method = OptimizationMethod::GAUSS_NEWTON;  // Optimization method selector

    DegenerateRegularizationParams degenerate_reg;  // Degenerate Regularization

    bool verbose = false;  // If true, print debug messages
};

namespace {

/// @brief Device copyable linearized result
struct LinearizedDevice {
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

    LinearizedDevice(const sycl_utils::DeviceQueue& q, size_t N = 1) : queue(q), size(N) {
        H0 = sycl::malloc_shared<sycl::float16>(size, *this->queue.ptr);
        H1 = sycl::malloc_shared<sycl::float16>(size, *this->queue.ptr);
        H2 = sycl::malloc_shared<sycl::float4>(size, *this->queue.ptr);
        b0 = sycl::malloc_shared<sycl::float3>(size, *this->queue.ptr);
        b1 = sycl::malloc_shared<sycl::float3>(size, *this->queue.ptr);
        error = sycl::malloc_shared<float>(size, *this->queue.ptr);
        inlier = sycl::malloc_shared<uint32_t>(size, *this->queue.ptr);
    }
    ~LinearizedDevice() {
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
    LinearizedResult toCPU(size_t i = 0) {
        const LinearizedResult result = {.H = eigen_utils::from_sycl_vec({this->H0[i], this->H1[i], this->H2[i]}),
                                         .b = eigen_utils::from_sycl_vec({this->b0[i], this->b1[i]}),
                                         .error = this->error[i],
                                         .inlier = this->inlier[i]};
        return result;
    }
};
}  // namespace

/// @brief Point cloud registration
class Registration {
public:
    using Ptr = std::shared_ptr<Registration>;

    /// @brief Constructor
    /// @param queue SYCL queue
    /// @param params Registration parameters
    Registration(const sycl_utils::DeviceQueue& queue, const RegistrationParams& params = RegistrationParams())
        : params_(params), queue_(queue) {
        this->neighbors_ = std::make_shared<shared_vector<knn::KNNResult>>(1, knn::KNNResult(), *this->queue_.ptr);
        this->neighbors_->at(0).allocate(this->queue_, 1, 1);

        this->linearized_on_device_ = std::make_shared<LinearizedDevice>(this->queue_);
        this->linearized_on_host_ =
            std::make_shared<shared_vector<LinearizedResult>>(1, LinearizedResult(), *this->queue_.ptr);
        this->error_on_host_ = std::make_shared<shared_vector<float>>(*this->queue_.ptr);
        this->inlier_on_host_ = std::make_shared<shared_vector<uint32_t>>(*this->queue_.ptr);

        this->degenerate_reg_.set_params(this->params_.degenerate_reg);
    }

    /// @brief validate parameters
    void validate_params(const PointCloudShared& source, const PointCloudShared& target) {
        if (this->params_.reg_type == RegType::POINT_TO_PLANE) {
            if (!target.has_normal()) {
                if (!target.has_cov()) {
                    throw std::runtime_error(
                        "[Registration::validate_params] "
                        "Normal vector or covariance matrices of target must be pre-computed before performing "
                        "Point-to-Plane ICP matching.");
                }
                std::cout << "[Caution] Normal vectors for Point-to-Plane ICP are not provided. " << std::endl;
                std::cout << "          Attempting to derive them from pre-computed covariance matrices." << std::endl;
                covariance::compute_normals_from_covariances(target);
            }
        }
        if (this->params_.reg_type == RegType::GICP) {
            if (!source.has_cov() || !target.has_cov()) {
                throw std::runtime_error(
                    "[Registration::validate_params] "
                    "Covariance matrices of source and target must be pre-computed before performing GICP matching.");
            }
        }
        if (this->params_.reg_type == RegType::GENZ) {
            if (!target.has_cov()) {
                throw std::runtime_error(
                    "[Registration::validate_params] "
                    "Covariance matrices of target must be pre-computed before performing GenZ-ICP matching.");
            }
            if (!target.has_normal()) {
                if (target.has_cov()) {
                    std::cout << "[Caution] Normal vectors for GenZ-ICP are not provided. " << std::endl;
                    std::cout << "          Attempting to derive them from pre-computed covariance matrices."
                              << std::endl;
                    covariance::compute_normals_from_covariances(target);
                }
            }
        }
        if (this->params_.reg_type == RegType::POINT_TO_DISTRIBUTION) {
            if (!target.has_cov()) {
                throw std::runtime_error(
                    "[Registration::validate_params] "
                    "Covariance matrices of target must be pre-computed before performing Point-to-Distribution ICP "
                    "matching.");
            }
        }
        if (this->params_.photometric.enable) {
            if (!target.has_normal()) {
                throw std::runtime_error(
                    "[Registration::validate_params] "
                    "Target normal vector must be pre-computed before performing photometric matching.");
            }

            const bool color_ready = source.has_rgb() && target.has_rgb() && target.has_color_gradient();
            const bool intensity_ready =
                source.has_intensity() && target.has_intensity() && target.has_intensity_gradient();

            if (!color_ready && !intensity_ready) {
                throw std::runtime_error(
                    "[Registration::validate_params] "
                    "RGB fields with gradients or intensity fields with gradients are required for photometric "
                    "matching.");
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
        if (this->params_.robust.type != RobustLossType::NONE) {
            if (this->params_.robust.init_scale <= 0.0f) {
                std::cout << "[Caution] `robust.init_scale` must be greater than zero. Disable robust loss."
                          << std::endl;
                this->params_.robust.type = RobustLossType::NONE;
            }
            if (this->params_.robust.auto_scale) {
                if (this->params_.robust.min_scale <= 0.0f ||
                    this->params_.robust.min_scale >= this->params_.robust.init_scale) {
                    std::cout
                        << "[Caution] `robust.min_scale` must be greater than zero and less than robust.init_scale."
                        << std::endl;
                    this->params_.robust.auto_scale = false;
                }
                if (this->params_.robust.scaling_iter <= 0) {
                    std::cout << "[Caution] `robust.scaling_iter` must be greater than zero. Disable auto scaling."
                              << std::endl;
                    this->params_.robust.auto_scale = false;
                }
            }
        }
    }

    /// @brief do registration
    /// @param source Source point cloud
    /// @param target Target point cloud
    /// @param target_knn Target KNN search
    /// @param initial_guess Initial transformation matrix
    /// @return Registration result
    RegistrationResult align(const PointCloudShared& source, const PointCloudShared& target,
                             const knn::KNNBase& target_knn,
                             const TransformMatrix& initial_guess = TransformMatrix::Identity()) {
        const size_t N = source.size();
        const size_t TARGET_SIZE = target.size();
        RegistrationResult result;
        result.T.matrix() = initial_guess;

        if (N == 0) return result;

        this->validate_params(source, target);

        {
            const float max_corr_dist = this->params_.max_correspondence_distance;
            float lambda = this->params_.lambda;
            float trust_region_radius = this->params_.dogleg.initial_trust_region_radius;
            float robust_scale = this->params_.robust.init_scale;
            const bool enable_robust_auto_scaling =
                this->params_.robust.type != RobustLossType::NONE && this->params_.robust.auto_scale;
            const size_t robust_levels =
                enable_robust_auto_scaling ? std::max<size_t>(1, this->params_.robust.scaling_iter) : 1;
            const float robust_scaling_factor =
                robust_levels > 1 ? std::pow(this->params_.robust.min_scale / this->params_.robust.init_scale,
                                             1.0f / static_cast<float>(robust_levels - 1))
                                  : 1.0f;

            // Iterate over each configured robust loss scale and perform the standard ICP update cycle.
            for (size_t robust_level = 0; robust_level < robust_levels; ++robust_level) {
                if (enable_robust_auto_scaling && this->params_.verbose) {
                    std::cout << "Robust scale: " << robust_scale << std::endl;
                }
                for (size_t iter = 0; iter < this->params_.max_iterations; ++iter) {
                    // Nearest neighbor search on device
                    auto knn_event =
                        target_knn.nearest_neighbor_search_async(source, (*this->neighbors_)[0], {}, result.T.matrix());

                    // Linearize on device for the current robust scale level
                    LinearizedResult linearized_result =
                        this->linearize(source, target, result.T.matrix(), max_corr_dist, robust_scale, knn_event.evs);

                    // Regularization
                    this->degenerate_reg_.regularize(linearized_result, result.T, Eigen::Isometry3f(initial_guess));

                    // Optimize on Host
                    switch (this->params_.optimization_method) {
                        case OptimizationMethod::LEVENBERG_MARQUARDT:
                            this->optimize_levenberg_marquardt(source, target, max_corr_dist, result, linearized_result,
                                                               lambda, iter, robust_scale);
                            break;
                        case OptimizationMethod::POWELL_DOGLEG:
                            this->optimize_powell_dogleg(source, target, max_corr_dist, result, linearized_result,
                                                         trust_region_radius, iter, robust_scale);
                            break;
                        case OptimizationMethod::GAUSS_NEWTON:
                            this->optimize_gauss_newton(result, linearized_result, lambda, iter);
                            break;
                    }
                    if (result.converged) {
                        break;
                    }
                }

                robust_scale *= robust_scaling_factor;
            }
        }

        return result;
    }

    /// @brief do registration with point cloud deskew using Velocity updating ICP (VICP)
    /// @param source Source point cloud
    /// @param target Target point cloud
    /// @param target_knn Target KNN search
    /// @param initial_guess Initial transformation matrix
    /// @param dt Initial transformation matrix
    /// @param prev_pose Initial transformation matrix
    /// @param initial_linear_vel Initial linear velocity
    /// @param initial_angular_vel Initial angular velocity
    /// @return Registration result
    RegistrationResult align_velocity_update(const PointCloudShared& source, const PointCloudShared& target,
                                             const knn::KNNBase& target_knn,
                                             const TransformMatrix& initial_guess = TransformMatrix::Identity(),
                                             float dt = 0.1f, size_t velocity_update_iter = 1,
                                             const TransformMatrix& prev_pose = TransformMatrix::Identity()) {
        const size_t N = source.size();
        const size_t TARGET_SIZE = target.size();
        RegistrationResult result;
        result.T.matrix() = initial_guess;

        if (N == 0) return result;

        this->validate_params(source, target);

        // copy
        auto deskewed = source;

        {
            const float max_corr_dist = this->params_.max_correspondence_distance;
            float lambda = this->params_.lambda;
            float trust_region_radius = this->params_.dogleg.initial_trust_region_radius;
            float robust_scale = this->params_.robust.init_scale;
            const bool enable_robust_auto_scaling =
                this->params_.robust.type != RobustLossType::NONE && this->params_.robust.auto_scale;
            const size_t robust_levels =
                enable_robust_auto_scaling ? std::max<size_t>(1, this->params_.robust.scaling_iter) : 1;
            const float robust_scaling_factor =
                robust_levels > 1 ? std::pow(this->params_.robust.min_scale / this->params_.robust.init_scale,
                                             1.0f / static_cast<float>(robust_levels - 1))
                                  : 1.0f;
            const bool has_timestamp = source.has_timestamps();
            const size_t deskew_levels = std::max<size_t>(1, velocity_update_iter);

            // Iterate over each configured robust loss scale and perform the standard ICP update cycle.
            for (size_t robust_level = 0; robust_level < robust_levels; ++robust_level) {
                if (enable_robust_auto_scaling && this->params_.verbose) {
                    std::cout << "Robust scale: " << robust_scale << std::endl;
                }
                for (size_t deskew_iter = 0; deskew_iter < deskew_levels; ++deskew_iter) {
                    if (this->params_.verbose) {
                        std::cout << "deskewed: " << deskew_iter << std::endl;
                    }
                    const Eigen::Isometry3f delta_pose = Eigen::Isometry3f(prev_pose).inverse() * result.T;
                    const Eigen::Vector<float, 6> delta_twist = eigen_utils::lie::se3_log(delta_pose);
                    const float delta_angle = delta_twist.head<3>().norm();
                    const float delta_dist = delta_twist.tail<3>().norm();
                    if (this->params_.verbose) {
                        std::cout << "deskewed[" << deskew_iter << "]: angle=" << delta_angle << ", dist=" << delta_dist
                                  << std::endl;
                    }
                    deskew::deskew_point_cloud_constant_velocity(source, deskewed, Eigen::Isometry3f(prev_pose),
                                                                 result.T, dt);

                    for (size_t iter = 0; iter < this->params_.max_iterations; ++iter) {
                        // Nearest neighbor search on device
                        auto knn_event = target_knn.nearest_neighbor_search_async(deskewed, (*this->neighbors_)[0], {},
                                                                                  result.T.matrix());

                        // Linearize on device for the current robust scale level
                        LinearizedResult linearized_result = this->linearize(
                            deskewed, target, result.T.matrix(), max_corr_dist, robust_scale, knn_event.evs);

                        // Regularization
                        this->degenerate_reg_.regularize(linearized_result, result.T, Eigen::Isometry3f(initial_guess));

                        // Optimize on Host
                        switch (this->params_.optimization_method) {
                            case OptimizationMethod::LEVENBERG_MARQUARDT:
                                this->optimize_levenberg_marquardt(deskewed, target, max_corr_dist, result,
                                                                   linearized_result, lambda, iter, robust_scale);
                                break;
                            case OptimizationMethod::POWELL_DOGLEG:
                                this->optimize_powell_dogleg(deskewed, target, max_corr_dist, result, linearized_result,
                                                             trust_region_radius, iter, robust_scale);
                                break;
                            case OptimizationMethod::GAUSS_NEWTON:
                                this->optimize_gauss_newton(result, linearized_result, lambda, iter);
                                break;
                        }
                        if (result.converged) {
                            break;
                        }
                    }
                }

                robust_scale *= robust_scaling_factor;
            }
        }

        return result;
    }

private:
    RegistrationParams params_;
    sycl_utils::DeviceQueue queue_;

    shared_vector_ptr<knn::KNNResult> neighbors_ = nullptr;
    std::shared_ptr<LinearizedDevice> linearized_on_device_ = nullptr;
    shared_vector_ptr<LinearizedResult> linearized_on_host_ = nullptr;
    shared_vector_ptr<float> error_on_host_ = nullptr;
    shared_vector_ptr<uint32_t> inlier_on_host_ = nullptr;

    DegenerateRegularization degenerate_reg_;
    float genz_alpha_ = 1.0f;

    template <typename Func>
    sycl_utils::events dispatch(Func&& exec) {
        sycl_utils::events events;
        auto dispatch_inner = [&]<RegType reg, typename RobustLossTypeTags, size_t... Js>(RobustLossType loss,
                                                                                          std::index_sequence<Js...>) {
            // Search for RobustLossType candidates
            return (
                ((loss == std::tuple_element_t<Js, RobustLossTypeTags>::value)
                     ? (events += exec.template operator()<reg, std::tuple_element_t<Js, RobustLossTypeTags>::value>(),
                        true)
                     : false) ||
                ...);
        };
        auto dispatch_outer = [&]<typename RegTypeTags, typename RobustLossTypeTags, size_t... Is>(
                                  RegType reg, RobustLossType loss, std::index_sequence<Is...>) {
            // Search for RegType candidates
            return (((reg == std::tuple_element_t<Is, RegTypeTags>::value)
                         ? dispatch_inner
                               .template operator()<std::tuple_element_t<Is, RegTypeTags>::value, RobustLossTypeTags>(
                                   loss, std::make_index_sequence<std::tuple_size_v<RobustLossTypeTags>>())
                         : false) ||
                    ...);
        };

        // Start search
        bool found = dispatch_outer.template operator()<RegTypeTags, RobustLossTypeTags>(
            this->params_.reg_type, this->params_.robust.type,
            std::make_index_sequence<std::tuple_size_v<RegTypeTags>>());

        if (!found) {
            throw std::runtime_error("[Registration::dispatch] Combination not found in tags!");
        }
        return events;
    }

    bool is_converged(const Eigen::Vector<float, 6>& delta) const {
        return delta.template head<3>().norm() < this->params_.criteria.rotation &&
               delta.template tail<3>().norm() < this->params_.criteria.translation;
    }

    float compute_genz_alpha(const PointCloudShared& source,  //
                             const PointCloudShared& target,  //
                             float max_correspondence_distance, const std::vector<sycl::event>& depends = {}) {
        sycl_points::shared_vector<uint32_t> counter_inlier(1, 0, *this->queue_.ptr);
        sycl_points::shared_vector<uint32_t> counter_plane(1, 0, *this->queue_.ptr);

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t N = source.size();

            const size_t work_group_size = this->queue_.get_work_group_size_for_parallel_reduction();
            const size_t global_size = this->queue_.get_global_size_for_parallel_reduction(N);

            const auto target_cov_ptr = target.covs_ptr();
            const auto neighbors_index_ptr = (*this->neighbors_)[0].indices->data();
            const auto neighbors_distances_ptr = (*this->neighbors_)[0].distances->data();

            auto sum_inlier = sycl::reduction(counter_inlier.data(), sycl::plus<uint32_t>());
            auto sum_plane = sycl::reduction(counter_plane.data(), sycl::plus<uint32_t>());

            const float planarity_threshold = this->params_.genz.planarity_threshold;
            const float max_corr_dist2 = max_correspondence_distance * max_correspondence_distance;

            h.depends_on(depends);
            h.parallel_for(                                       //
                sycl::nd_range<1>(global_size, work_group_size),  // range
                sum_inlier, sum_plane,                            // reduction
                [=](sycl::nd_item<1> item, auto& sum_inlier_arg, auto& sum_plane_arg) {
                    const size_t index = item.get_global_id(0);
                    if (index >= N) return;

                    if (neighbors_distances_ptr[index] > max_corr_dist2) {
                        return;
                    }

                    const auto target_idx = neighbors_index_ptr[index];
                    const auto cov = target_cov_ptr[target_idx];

                    Eigen::Vector3f eigenvalues;
                    Eigen::Matrix3f eigenvectors;
                    eigen_utils::symmetric_eigen_decomposition_3x3(cov.block<3, 3>(0, 0), eigenvalues, eigenvectors);
                    const float sum_eigenvalues = eigenvalues(0) + eigenvalues(1) + eigenvalues(2);
                    const float surface_variation = (sum_eigenvalues > std::numeric_limits<float>::epsilon())
                                                        ? eigenvalues(0) / sum_eigenvalues
                                                        : 1.0f;
                    if (surface_variation < planarity_threshold) {
                        ++sum_plane_arg;
                    }
                    ++sum_inlier_arg;
                });
        });
        event.wait_and_throw();
        const auto inlier_count = counter_inlier[0];
        if (inlier_count == 0U) {
            return 1.0f;
        }
        return static_cast<float>(counter_plane[0]) / static_cast<float>(inlier_count);
    }

    template <RegType reg, RobustLossType loss>
    sycl_utils::events linearize_parallel_reduction_async(const PointCloudShared& source,
                                                          const PointCloudShared& target, const Eigen::Matrix4f transT,
                                                          float max_correspondence_distance, float robust_scale,
                                                          const std::vector<sycl::event>& depends) {
        if constexpr (reg == RegType::GENZ) {
            this->genz_alpha_ = compute_genz_alpha(source, target, max_correspondence_distance, depends);
        }

        // The robust_scale argument controls the influence of the robust loss inside the reduction kernel.
        sycl_utils::events events;
        events += this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t N = source.size();

            const size_t work_group_size = this->queue_.get_work_group_size_for_parallel_reduction();
            const size_t global_size = this->queue_.get_global_size_for_parallel_reduction(N);

            // convert to sycl::float4
            const auto cur_T = eigen_utils::to_sycl_vec(transT);

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

            const auto source_intensity_ptr = source.has_intensity() ? source.intensities_ptr() : nullptr;
            const auto target_intensity_ptr = target.has_intensity() ? target.intensities_ptr() : nullptr;

            const auto target_intensity_grad_ptr =
                target.has_intensity_gradient() ? target.intensity_gradients_ptr() : nullptr;
            const float photometric_weight =
                this->params_.photometric.enable ? this->params_.photometric.photometric_weight : 0.0f;

            const auto neighbors_index_ptr = (*this->neighbors_)[0].indices->data();
            const auto neighbors_distances_ptr = (*this->neighbors_)[0].distances->data();

            const float max_corr_dist2 = max_correspondence_distance * max_correspondence_distance;
            const float genz_alpha = this->genz_alpha_;

            // reduction
            this->linearized_on_device_->setZero();
            auto sum_H0 = sycl::reduction(this->linearized_on_device_->H0, sycl::plus<sycl::float16>());
            auto sum_H1 = sycl::reduction(this->linearized_on_device_->H1, sycl::plus<sycl::float16>());
            auto sum_H2 = sycl::reduction(this->linearized_on_device_->H2, sycl::plus<sycl::float4>());
            auto sum_b0 = sycl::reduction(this->linearized_on_device_->b0, sycl::plus<sycl::float3>());
            auto sum_b1 = sycl::reduction(this->linearized_on_device_->b1, sycl::plus<sycl::float3>());
            auto sum_error = sycl::reduction(this->linearized_on_device_->error, sycl::plus<float>());
            auto sum_inlier = sycl::reduction(this->linearized_on_device_->inlier, sycl::plus<uint32_t>());

            // wait for knn search
            h.depends_on(depends);

            h.parallel_for(                                                     //
                sycl::nd_range<1>(global_size, work_group_size),                // range
                sum_H0, sum_H1, sum_H2, sum_b0, sum_b1, sum_error, sum_inlier,  // reduction
                [=](sycl::nd_item<1> item, auto& sum_H0_arg, auto& sum_H1_arg, auto& sum_H2_arg, auto& sum_b0_arg,
                    auto& sum_b1_arg, auto& sum_error_arg, auto& sum_inlier_arg) {
                    const size_t index = item.get_global_id(0);
                    if (index >= N) return;

                    if (neighbors_distances_ptr[index] > max_corr_dist2) {
                        return;
                    }
                    const auto target_idx = neighbors_index_ptr[index];

                    const auto source_cov = source_cov_ptr ? source_cov_ptr[index] : Covariance::Identity();
                    const auto target_cov = target_cov_ptr ? target_cov_ptr[target_idx] : Covariance::Identity();

                    const auto target_normal = target_normal_ptr ? target_normal_ptr[target_idx] : Normal::Zero();

                    const auto source_rgb = source_rgb_ptr ? source_rgb_ptr[index] : RGBType::Zero();
                    const auto target_rgb = target_rgb_ptr ? target_rgb_ptr[target_idx] : RGBType::Zero();
                    const auto target_grad = target_grad_ptr ? target_grad_ptr[target_idx] : ColorGradient::Zero();

                    const float source_intensity = source_intensity_ptr ? source_intensity_ptr[index] : 0.0f;
                    const float target_intensity = target_intensity_ptr ? target_intensity_ptr[target_idx] : 0.0f;
                    const auto target_intensity_grad =
                        target_intensity_grad_ptr ? target_intensity_grad_ptr[target_idx] : IntensityGradient::Zero();

                    const bool use_color = source_rgb_ptr && target_rgb_ptr && target_grad_ptr;
                    const bool use_intensity =
                        source_intensity_ptr && target_intensity_ptr && target_intensity_grad_ptr;

                    const LinearizedResult linearized =
                        kernel::linearize<reg>(cur_T, source_ptr[index], source_cov,               //
                                               target_ptr[target_idx], target_cov, target_normal,  //
                                               source_rgb, target_rgb, target_grad, use_color,     //
                                               source_intensity, target_intensity,                 //
                                               target_intensity_grad, use_intensity, photometric_weight, genz_alpha);
                    const float robust_weight = kernel::compute_robust_weight<loss>(linearized.error, robust_scale);

                    // reduction on device
                    const auto& [H0, H1, H2] = eigen_utils::to_sycl_vec(linearized.H);
                    const auto& [b0, b1] = eigen_utils::to_sycl_vec(linearized.b);
                    sum_H0_arg += H0 * robust_weight;
                    sum_H1_arg += H1 * robust_weight;
                    sum_H2_arg += H2 * robust_weight;
                    sum_b0_arg += b0 * robust_weight;
                    sum_b1_arg += b1 * robust_weight;
                    sum_error_arg += kernel::compute_robust_error<loss>(linearized.error, robust_scale);
                    ++sum_inlier_arg;
                });
        });
        return events;
    }

    LinearizedResult linearize(const PointCloudShared& source, const PointCloudShared& target,
                               const Eigen::Matrix4f transT, float max_correspondence_distance, float robust_scale,
                               const std::vector<sycl::event>& depends) {
        auto events = this->dispatch([&]<RegType reg, RobustLossType loss>() {
            return this->linearize_parallel_reduction_async<reg, loss>(
                source, target, transT, max_correspondence_distance, robust_scale, depends);
        });

        events.wait_and_throw();
        return this->linearized_on_device_->toCPU(0);
    }

    template <RegType reg, RobustLossType loss>
    std::tuple<float, uint32_t> compute_error_parallel_reduction(
        const PointCloudShared& source, const PointCloudShared& target, const knn::KNNResult& knn_results,
        const Eigen::Matrix4f transT, float max_correspondence_distance, float robust_scale) {
        // The robust_scale argument ensures error reduction uses the caller-provided loss scale.
        shared_vector<float> error(1, 0.0f, *this->queue_.ptr);
        shared_vector<uint32_t> inlier(1, 0, *this->queue_.ptr);

        auto event = this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t N = source.size();

            const size_t work_group_size = this->queue_.get_work_group_size_for_parallel_reduction();
            const size_t global_size = this->queue_.get_global_size_for_parallel_reduction(N);

            // convert to sycl::float4
            const auto cur_T = eigen_utils::to_sycl_vec(transT);

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
            const auto source_intensity_ptr = source.has_intensity() ? source.intensities_ptr() : nullptr;
            const auto target_intensity_ptr = target.has_intensity() ? target.intensities_ptr() : nullptr;
            const auto target_intensity_grad_ptr =
                target.has_intensity_gradient() ? target.intensity_gradients_ptr() : nullptr;
            const float photometric_weight =
                this->params_.photometric.enable ? this->params_.photometric.photometric_weight : 0.0f;
            const auto neighbors_index_ptr = knn_results.indices->data();
            const auto neighbors_distances_ptr = knn_results.distances->data();

            const float max_corr_dist2 = max_correspondence_distance * max_correspondence_distance;
            const float genz_alpha = this->genz_alpha_;

            // output
            auto sum_error = sycl::reduction(error.data(), sycl::plus<float>());
            auto sum_inlier = sycl::reduction(inlier.data(), sycl::plus<uint32_t>());

            h.parallel_for(                                       //
                sycl::nd_range<1>(global_size, work_group_size),  // range
                sum_error, sum_inlier,                            // reduction
                [=](sycl::nd_item<1> item, auto& sum_error_arg, auto& sum_inlier_arg) {
                    const size_t index = item.get_global_id(0);
                    if (index >= N) return;

                    if (neighbors_distances_ptr[index] > max_corr_dist2) {
                        return;
                    }
                    const auto target_idx = neighbors_index_ptr[index];
                    const auto source_cov = source_cov_ptr ? source_cov_ptr[index] : Covariance::Identity();
                    const auto target_cov = target_cov_ptr ? target_cov_ptr[target_idx] : Covariance::Identity();
                    const auto target_normal = target_normal_ptr ? target_normal_ptr[target_idx] : Normal::Zero();
                    const auto source_rgb = source_rgb_ptr ? source_rgb_ptr[index] : RGBType::Zero();
                    const auto target_rgb = target_rgb_ptr ? target_rgb_ptr[target_idx] : RGBType::Zero();
                    const auto target_grad = target_grad_ptr ? target_grad_ptr[target_idx] : ColorGradient::Zero();
                    const float source_intensity = source_intensity_ptr ? source_intensity_ptr[index] : 0.0f;
                    const float target_intensity = target_intensity_ptr ? target_intensity_ptr[target_idx] : 0.0f;
                    const auto target_intensity_grad =
                        target_intensity_grad_ptr ? target_intensity_grad_ptr[target_idx] : IntensityGradient::Zero();
                    const bool use_color = source_rgb_ptr && target_rgb_ptr && target_grad_ptr;
                    const bool use_intensity =
                        source_intensity_ptr && target_intensity_ptr && target_intensity_grad_ptr;

                    const float err =
                        kernel::calculate_error<reg>(cur_T,                                              //
                                                     source_ptr[index], source_cov,                      // source
                                                     target_ptr[target_idx], target_cov, target_normal,  // target
                                                     source_rgb, target_rgb, target_grad, use_color,     //
                                                     source_intensity, target_intensity, target_intensity_grad,
                                                     use_intensity, photometric_weight, genz_alpha);

                    sum_error_arg += kernel::compute_robust_error<loss>(err, robust_scale);
                    ++sum_inlier_arg;
                });
        });
        event.wait_and_throw();
        return {error[0], inlier[0]};
    }

    std::tuple<float, uint32_t> compute_error(const PointCloudShared& source, const PointCloudShared& target,
                                              const knn::KNNResult& knn_results, const Eigen::Matrix4f transT,
                                              float max_correspondence_distance, float robust_scale) {
        std::tuple<float, uint32_t> result = {0.0f, 0};
        this->dispatch([&]<RegType reg, RobustLossType loss>() {
            result = this->compute_error_parallel_reduction<reg, loss>(source, target, knn_results, transT,
                                                                       max_correspondence_distance, robust_scale);
            return sycl_utils::events{};
        });
        return result;
    }

    bool solve_linear_system(const Eigen::Matrix<float, 6, 6>& H, const Eigen::Vector<float, 6>& b,
                             Eigen::Vector<float, 6>& solution) {
        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt;
        ldlt.compute(H);
        if (ldlt.info() == Eigen::Success) {
            solution = ldlt.solve(-b);
            return true;
        }
        solution.setZero();
        return false;
    }

    void optimize_gauss_newton(RegistrationResult& result, const LinearizedResult& linearized_result, float lambda,
                               size_t iter) {
        Eigen::Vector<float, 6> delta;
        const bool success = this->solve_linear_system(
            linearized_result.H + lambda * Eigen::Matrix<float, 6, 6>::Identity(), linearized_result.b, delta);
        if (success) {
            result.converged = this->is_converged(delta);
        } else {
            result.converged = false;
        }
        result.T = result.T * Eigen::Isometry3f(eigen_utils::lie::se3_exp(delta));
        result.iterations = iter;
        result.H = linearized_result.H;
        result.b = linearized_result.b;
        result.error = linearized_result.error;
        result.inlier = linearized_result.inlier;

        if (this->params_.verbose) {
            std::cout << "iter [" << iter << "] ";
            std::cout << "error: " << result.error << ", ";
            std::cout << "inlier: " << result.inlier << ", ";
            std::cout << "dt: " << delta.tail<3>().norm() << ", ";
            std::cout << "dr: " << delta.head<3>().norm() << std::endl;
        }
    }

    bool optimize_levenberg_marquardt(const PointCloudShared& source, const PointCloudShared& target,
                                      float max_correspondence_distance, RegistrationResult& result,
                                      const LinearizedResult& linearized_result, float& lambda, size_t iter,
                                      float robust_scale) {
        const auto& H = linearized_result.H;
        const auto& g = linearized_result.b;
        const float current_error = linearized_result.error;

        bool updated = false;
        float last_error = std::numeric_limits<float>::max();

        Eigen::Vector<float, 6> delta;
        for (size_t i = 0; i < this->params_.lm.max_inner_iterations; ++i) {
            const bool success =
                this->solve_linear_system(H + lambda * Eigen::Matrix<float, 6, 6>::Identity(), g, delta);
            if (success) {
                result.converged = this->is_converged(delta);
            } else {
                result.converged = false;
            }
            const Eigen::Isometry3f new_T = result.T * Eigen::Isometry3f(eigen_utils::lie::se3_exp(delta));

            const auto [new_error, inlier] = compute_error(source, target, this->neighbors_->at(0), new_T.matrix(),
                                                           max_correspondence_distance, robust_scale);

            if (this->params_.verbose) {
                std::cout << "iter [" << iter << "] ";
                std::cout << "inner: " << i << ", ";
                std::cout << "lambda: " << lambda << ", ";
                std::cout << "error: " << new_error << ", ";
                std::cout << "inlier: " << inlier << ", ";
                std::cout << "dt: " << delta.tail<3>().norm() << ", ";
                std::cout << "dr: " << delta.head<3>().norm() << std::endl;
            }
            if (new_error <= linearized_result.error) {
                result.converged = this->is_converged(delta);
                result.T = new_T;
                result.error = new_error;
                result.inlier = inlier;
                updated = true;

                lambda = std::clamp(lambda / this->params_.lm.lambda_factor, this->params_.lm.min_lambda,
                                    this->params_.lm.max_lambda);

                break;
            } else if (std::fabs(new_error - last_error) <= 1e-6f) {
                result.converged = this->is_converged(delta);
                result.T = new_T;
                result.error = new_error;
                result.inlier = inlier;
                updated = false;

                break;
            } else {
                lambda = std::clamp(lambda * this->params_.lm.lambda_factor, this->params_.lm.min_lambda,
                                    this->params_.lm.max_lambda);
            }
            last_error = new_error;
        }

        result.iterations = iter;
        result.H = H;
        result.b = g;
        return updated;
    }

    bool optimize_powell_dogleg(const PointCloudShared& source, const PointCloudShared& target,
                                float max_correspondence_distance, RegistrationResult& result,
                                const LinearizedResult& linearized_result, float& trust_region_radius, size_t iter,
                                float robust_scale) {
        bool updated = false;
        const auto& H = linearized_result.H;
        const auto& g = linearized_result.b;
        const float current_error = linearized_result.error;

        result.H = H;
        result.b = g;
        result.error = current_error;
        result.inlier = linearized_result.inlier;
        result.iterations = iter;

        const auto clamp_radius = [&](float radius) {
            return std::clamp(radius, this->params_.dogleg.min_trust_region_radius,
                              this->params_.dogleg.max_trust_region_radius);
        };

        trust_region_radius = clamp_radius(trust_region_radius);

        Eigen::Vector<float, 6> p_gn = Eigen::Vector<float, 6>::Zero();
        float norm_p_gn = 0.0f;
        bool has_valid_gn = false;
        const bool success = this->solve_linear_system(H, g, p_gn);
        if (success) {
            norm_p_gn = p_gn.norm();
            has_valid_gn = std::isfinite(norm_p_gn);
        }

        const float g_norm_sq = g.squaredNorm();
        const Eigen::Vector<float, 6> Hg = H * g;
        const float g_H_g = g.dot(Hg);
        Eigen::Vector<float, 6> p_sd = -g;
        if (g_H_g > std::numeric_limits<float>::epsilon()) {
            const float alpha = g_norm_sq / g_H_g;
            if (std::isfinite(alpha)) {
                p_sd = -alpha * g;
            }
        }
        const float norm_p_sd = p_sd.norm();

        Eigen::Vector<float, 6> p_dl = Eigen::Vector<float, 6>::Zero();
        float step_norm = 0.0f;
        if (has_valid_gn && norm_p_gn <= trust_region_radius) {
            p_dl = p_gn;
            step_norm = norm_p_gn;
        } else if (norm_p_sd >= trust_region_radius) {
            if (norm_p_sd > std::numeric_limits<float>::epsilon()) {
                p_dl = (trust_region_radius / norm_p_sd) * p_sd;
            }
            step_norm = trust_region_radius;
        } else if (has_valid_gn) {
            const Eigen::Vector<float, 6> diff = p_gn - p_sd;
            const float a = diff.squaredNorm();
            const float b = 2.0f * p_sd.dot(diff);
            const float c = p_sd.squaredNorm() - trust_region_radius * trust_region_radius;
            float discriminant = b * b - 4.0f * a * c;
            discriminant = std::max(discriminant, 0.0f);
            float tau = 0.0f;
            if (a > std::numeric_limits<float>::epsilon()) {
                tau = (-b + std::sqrt(discriminant)) / (2.0f * a);
            }
            tau = std::clamp(tau, 0.0f, 1.0f);
            p_dl = p_sd + tau * diff;
            step_norm = p_dl.norm();
        } else {
            p_dl = p_sd;
            if (norm_p_sd > trust_region_radius && norm_p_sd > std::numeric_limits<float>::epsilon()) {
                const float scale = trust_region_radius / norm_p_sd;
                p_dl *= scale;
                step_norm = trust_region_radius;
            } else {
                step_norm = norm_p_sd;
            }
        }

        const float predicted_reduction = -(g.dot(p_dl) + 0.5f * p_dl.dot(H * p_dl));

        if (predicted_reduction <= 0.0f) {
            trust_region_radius = clamp_radius(trust_region_radius * this->params_.dogleg.gamma_decrease);
            return updated;
        }

        const Eigen::Isometry3f new_T = result.T * Eigen::Isometry3f(eigen_utils::lie::se3_exp(p_dl));
        const auto [new_error, inlier] = compute_error(source, target, this->neighbors_->at(0), new_T.matrix(),
                                                       max_correspondence_distance, robust_scale);

        const float actual_reduction = current_error - new_error;
        const float rho = actual_reduction / predicted_reduction;

        if (this->params_.verbose) {
            std::cout << "iter [" << iter << "] ";
            std::cout << "radius: " << trust_region_radius << ", ";
            std::cout << "rho: " << rho << ", ";
            std::cout << "error: " << new_error << ", ";
            std::cout << "inlier: " << inlier << ", ";
            std::cout << "dt: " << p_dl.tail<3>().norm() << ", ";
            std::cout << "dr: " << p_dl.head<3>().norm() << std::endl;
        }

        if (rho < this->params_.dogleg.eta1) {
            trust_region_radius = clamp_radius(trust_region_radius * this->params_.dogleg.gamma_decrease);
            return updated;
        }

        result.converged = this->is_converged(p_dl);
        result.T = new_T;
        result.error = new_error;
        result.inlier = inlier;
        updated = true;

        if (rho > this->params_.dogleg.eta2 && step_norm >= trust_region_radius * 0.99f) {
            trust_region_radius = clamp_radius(trust_region_radius * this->params_.dogleg.gamma_increase);
        }

        return updated;
    }
};

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
