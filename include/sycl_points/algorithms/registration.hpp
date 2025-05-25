#pragma once

#include <sycl_points/algorithms/knn_search.hpp>
#include <sycl_points/algorithms/registration_factor.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {

namespace algorithms {

namespace registration {

struct RegistrationParams {
    size_t max_iterations = 20;                // max iteration
    float lambda = 1e-6f;                      // initial damping factor
    float max_correspondence_distance = 1.0f;  // max correspondence distance
    float translation_eps = 1e-3f;             // translation tolerance
    float rotation_eps = 1e-3f;                // rotation tolerance [rad]
    bool verbose = false;                      // If true, print debug messages
    // bool optimize_lm = false; // If true, use Levenberg-Marquardt method, else use Gauss-Newton method.
    // size_t max_inner_iterations = 10; // (for LM method)
    // float lambda_factor = 10.0f; // lambda increase factor (for LM method)
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

    sycl_utils::DeviceQueue queue;

    LinearlizedDevice(const sycl_utils::DeviceQueue& q) : queue(q) {
        H0 = sycl::malloc_shared<sycl::float16>(1, *this->queue.ptr);
        H1 = sycl::malloc_shared<sycl::float16>(1, *this->queue.ptr);
        H2 = sycl::malloc_shared<sycl::float4>(1, *this->queue.ptr);
        b0 = sycl::malloc_shared<sycl::float3>(1, *this->queue.ptr);
        b1 = sycl::malloc_shared<sycl::float3>(1, *this->queue.ptr);
        error = sycl::malloc_shared<float>(1, *this->queue.ptr);
    }
    ~LinearlizedDevice() {
        sycl_utils::free(H0, *this->queue.ptr);
        sycl_utils::free(H1, *this->queue.ptr);
        sycl_utils::free(H2, *this->queue.ptr);
        sycl_utils::free(b0, *this->queue.ptr);
        sycl_utils::free(b1, *this->queue.ptr);
        sycl_utils::free(error, *this->queue.ptr);
    }
    void setZero() {
        for (size_t i = 0; i < 16; ++i) {
            H0[0][i] = 0.0f;
            H1[0][i] = 0.0f;
        }
        for (size_t i = 0; i < 4; ++i) {
            H2[0][i] = 0.0f;
        }
        for (size_t i = 0; i < 3; ++i) {
            b0[0][i] = 0.0f;
            b1[0][i] = 0.0f;
        }
        error[0] = 0.0f;
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

        float lambda = this->params_.lambda;
        const float max_dist_2 = this->params_.max_correspondence_distance * this->params_.max_correspondence_distance;
        const auto verbose = this->params_.verbose;

        sycl_utils::events transform_events;
        for (size_t iter = 0; iter < this->params_.max_iterations; ++iter) {
            prev_T = result.T;

            // Nearest neighbor search on device
            auto knn_event = target_tree.knn_search_async<1>(this->aligned_->points->data(), N, 1, (*this->neighbors_)[0],
                                                             transform_events.evs);

            // Linearlize on device
            const LinearlizedResult linearlized_result =
                this->linearlize(source, *this->aligned_, target, result.T.matrix(), max_dist_2, knn_event.evs);

            // Optimize on Host
            this->optimize_gauss_newton(result, linearlized_result, lambda, verbose, iter);

            // Async transform source points on device
            transform_events =
                transform::transform_async(*this->aligned_, result.T.matrix() * prev_T.matrix().inverse());  // zero copy

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

    bool is_converged(const Eigen::Matrix<float, 6, 1>& delta) const {
        return delta.template head<3>().norm() < this->params_.rotation_eps &&
               delta.template tail<3>().norm() < this->params_.translation_eps;
    }

    LinearlizedResult linearlize_sequential_reduction(const PointCloudShared& source,
                                                      const PointCloudShared& transform_source,
                                                      const PointCloudShared& target, const Eigen::Matrix4f transT,
                                                      float max_correspondence_distance_2,
                                                      const std::vector<sycl::event>& depends) {
        const size_t N = source.size();
        sycl_utils::events events;
        events += this->queue_.ptr->submit([&](sycl::handler& h) {
            if (this->linearlized_on_host_->size()) {
                this->linearlized_on_host_->resize(N);
            }
            const size_t work_group_size = queue_.get_work_group_size();
            const size_t global_size = queue_.get_global_size(N);

            // convert to sycl::float4
            const auto cur_T = eigen_utils::to_sycl_vec(transT);

            // get pointers
            // input
            const auto source_ptr = source.points->data();
            const auto transform_source_cov_ptr = transform_source.covs_ptr();
            const auto target_ptr = target.points_ptr();
            const auto target_cov_ptr = target.covs_ptr();
            const auto neighbors_index_ptr = (*this->neighbors_)[0].indices->data();
            const auto neighbors_distances_ptr = (*this->neighbors_)[0].distances->data();
            // output
            const auto linearlized_ptr = this->linearlized_on_host_->data();

            // wait for knn search
            h.depends_on(depends);

            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;

                if (neighbors_distances_ptr[i] > max_correspondence_distance_2) {
                    linearlized_ptr[i].H.setZero();
                    linearlized_ptr[i].b.setZero();
                    linearlized_ptr[i].error = 0.0f;
                } else {
                    PointType transformed_source;
                    transform::kernel::transform_point(source_ptr[i], transformed_source, cur_T.data());
                    const auto cur_T_mat = eigen_utils::from_sycl_vec(cur_T);

                    linearlized_ptr[i] = kernel::linearlize<icp>(
                        cur_T_mat, source_ptr[i], transformed_source, transform_source_cov_ptr[i],
                        target_ptr[neighbors_index_ptr[i]], target_cov_ptr[neighbors_index_ptr[i]]);
                }
            });
        });
        events.wait();

        LinearlizedResult linearlized_result;
        linearlized_result.H.setZero();
        linearlized_result.b.setZero();
        linearlized_result.error = 0.0f;
        // reduction on host
        for (size_t i = 0; i < N; ++i) {
            linearlized_result.H += (*this->linearlized_on_host_)[i].H;
            linearlized_result.b += (*this->linearlized_on_host_)[i].b;
            linearlized_result.error += (*this->linearlized_on_host_)[i].error;
        }
        return linearlized_result;
    }

    LinearlizedResult linearlize_parallel_reduction(const PointCloudShared& source,
                                                    const PointCloudShared& transform_source,
                                                    const PointCloudShared& target, const Eigen::Matrix4f transT,
                                                    float max_correspondence_distance_2,
                                                    const std::vector<sycl::event>& depends) {
        const size_t N = source.size();
        sycl_utils::events events;
        events += this->queue_.ptr->submit([&](sycl::handler& h) {
            const size_t work_group_size = this->queue_.get_work_group_size_for_parallel_reduction();
            const size_t global_size = this->queue_.get_global_size_for_parallel_reduction(N);

            // convert to sycl::float4
            const auto cur_T = eigen_utils::to_sycl_vec(transT);

            // get pointers
            // input
            const auto source_ptr = source.points->data();
            const auto transform_source_cov_ptr = transform_source.covs_ptr();
            const auto target_ptr = target.points_ptr();
            const auto target_cov_ptr = target.covs_ptr();
            const auto neighbors_index_ptr = (*this->neighbors_)[0].indices->data();
            const auto neighbors_distances_ptr = (*this->neighbors_)[0].distances->data();

            // output
            this->linearlized_on_device_->setZero();
            const auto H0_ptr = this->linearlized_on_device_->H0;
            const auto H1_ptr = this->linearlized_on_device_->H1;
            const auto H2_ptr = this->linearlized_on_device_->H2;
            const auto b0_ptr = this->linearlized_on_device_->b0;
            const auto b1_ptr = this->linearlized_on_device_->b1;
            const auto error_ptr = this->linearlized_on_device_->error;

            // reduction
            auto sum_H0 = sycl::reduction(H0_ptr, sycl::plus<sycl::float16>());
            auto sum_H1 = sycl::reduction(H1_ptr, sycl::plus<sycl::float16>());
            auto sum_H2 = sycl::reduction(H2_ptr, sycl::plus<sycl::float4>());
            auto sum_b0 = sycl::reduction(b0_ptr, sycl::plus<sycl::float3>());
            auto sum_b1 = sycl::reduction(b1_ptr, sycl::plus<sycl::float3>());
            auto sum_error = sycl::reduction(error_ptr, sycl::plus<float>());

            // wait for knn search
            h.depends_on(depends);

            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),    // range
                           sum_H0, sum_H1, sum_H2, sum_b0, sum_b1, sum_error,  // reduction
                           [=](sycl::nd_item<1> item, auto& sum_H0_arg, auto& sum_H1_arg, auto& sum_H2_arg,
                               auto& sum_b0_arg, auto& sum_b1_arg, auto& sum_error_arg) {
                               const size_t i = item.get_global_id(0);
                               if (i >= N) return;

                               if (neighbors_distances_ptr[i] > max_correspondence_distance_2) {
                                   return;
                               } else {
                                   PointType transformed_source;
                                   transform::kernel::transform_point(source_ptr[i], transformed_source, cur_T.data());
                                   const auto cur_T_mat = eigen_utils::from_sycl_vec(cur_T);

                                   const LinearlizedResult result = kernel::linearlize<icp>(
                                       cur_T_mat, source_ptr[i], transformed_source, transform_source_cov_ptr[i],
                                       target_ptr[neighbors_index_ptr[i]], target_cov_ptr[neighbors_index_ptr[i]]);
                                   // reduction on device
                                   const auto& [H0, H1, H2] = eigen_utils::to_sycl_vec(result.H);
                                   const auto& [b0, b1] = eigen_utils::to_sycl_vec(result.b);
                                   sum_H0_arg += H0;
                                   sum_H1_arg += H1;
                                   sum_H2_arg += H2;
                                   sum_b0_arg += b0;
                                   sum_b1_arg += b1;
                                   sum_error_arg += result.error;
                               }
                           });
        });
        events.wait();

        const LinearlizedResult linearlized_result = {
            .H = eigen_utils::from_sycl_vec({this->linearlized_on_device_->H0[0],  //
                                             this->linearlized_on_device_->H1[0],  //
                                             this->linearlized_on_device_->H2[0]}),
            .b = eigen_utils::from_sycl_vec({this->linearlized_on_device_->b0[0],  //
                                             this->linearlized_on_device_->b1[0]}),
            .error = this->linearlized_on_device_->error[0]};
        return linearlized_result;
    }

    LinearlizedResult linearlize(const PointCloudShared& source, const PointCloudShared& transform_source,
                                 const PointCloudShared& target, const Eigen::Matrix4f transT,
                                 float max_correspondence_distance_2, const std::vector<sycl::event>& depends) {
        if (this->queue_.is_nvidia()) {
            return this->linearlize_parallel_reduction(source, transform_source, target, transT,
                                                       max_correspondence_distance_2, depends);
        } else {
            return this->linearlize_sequential_reduction(source, transform_source, target, transT,
                                                         max_correspondence_distance_2, depends);
        }
    }

    void optimize_gauss_newton(RegistrationResult& result, const LinearlizedResult& linearlized_result, float lambda,
                               bool verbose, size_t iter) {
        const Eigen::Matrix<float, 6, 1> delta =
            (linearlized_result.H + lambda * Eigen::Matrix<float, 6, 6>::Identity())
                .ldlt()
                .solve(-linearlized_result.b);
        result.converged = this->is_converged(delta);
        result.T = result.T * eigen_utils::lie::se3_exp(delta);
        result.iterations = iter;
        result.H = linearlized_result.H;
        result.b = linearlized_result.b;
        result.error = linearlized_result.error;
        if (verbose) {
            std::cout << "iter [" << iter << "] ";
            std::cout << "error: " << result.error << ", ";
            std::cout << "dt: " << delta.tail<3>().norm() << ", ";
            std::cout << "dr: " << delta.head<3>().norm() << std::endl;
        }
    }
};

using RegistrationPointToPoint = Registration<ICPType::POINT_TO_POINT>;
using RegistrationGICP = Registration<ICPType::GICP>;

}  // namespace registration

}  // namespace algorithms

}  // namespace sycl_points
