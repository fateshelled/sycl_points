#pragma once

#include <mutex>
#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/algorithms/downsampling.hpp>
#include <sycl_points/algorithms/knn_search.hpp>
#include <sycl_points/algorithms/registration_factor.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {

namespace algorithms {

struct RegistrationParams {
    size_t max_iterations = 20;
    //   size_t max_inner_iterations = 10; // LM method
    float lambda = 1e-6f;
    //   float lambda_factor = 10.0f; // LM method
    // bool optimize_lm = false;
    float max_correspondence_distance = 1.0f;
    float translation_eps = 1e-3f;
    float rotation_eps = 1e-3f;
    bool verbose = false;
};

struct RegistrationResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    bool converged = false;
    size_t iterations = 0;
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
    float error = std::numeric_limits<float>::max();
};

template <typename PointCloud = PointCloudShared, factor::ICPType icp = factor::ICPType::GICP>
class Registration {
public:
    Registration(const std::shared_ptr<sycl::queue>& queue_ptr, const RegistrationParams& params = RegistrationParams())
        : params_(params), queue_ptr_(queue_ptr) {
        this->cur_T_ = std::make_shared<shared_vector<TransformMatrix>>(
            1, TransformMatrix::Identity(), shared_allocator<TransformMatrix>(*this->queue_ptr_, {}));

        this->max_distance_ = std::make_shared<shared_vector<float>>(1, params_.max_correspondence_distance,
                                                                     shared_allocator<float>(*this->queue_ptr_, {}));

        this->neighbors_ = std::make_shared<shared_vector<KNNResultSYCL>>(
            1, KNNResultSYCL(), shared_allocator<KNNResultSYCL>(*this->queue_ptr_, {}));
        this->neighbors_->at(0).allocate(*this->queue_ptr_, 1, 1);

        this->linearlized_ = std::make_shared<shared_vector<factor::LinearlizedResult>>(
            1, factor::LinearlizedResult(), shared_allocator<factor::LinearlizedResult>(*this->queue_ptr_, {}));
    }

    RegistrationResult optimize(const PointCloud& source, const PointCloud& target, const KDTreeSYCL& target_tree,
                                const TransformMatrix& init_T = TransformMatrix::Identity()) {
        const size_t N = traits::pointcloud::size(source);
        RegistrationResult result;
        result.T.matrix() = init_T;

        if (N == 0) return result;

        Eigen::Isometry3f prev_T = Eigen::Isometry3f::Identity();
        const auto transform_source = transform_sycl_copy(source, init_T);

        float lambda = this->params_.lambda;
        const float max_dist_2 = this->params_.max_correspondence_distance * this->params_.max_correspondence_distance;
        const auto verbose = this->params_.verbose;

        // Optimize work group size
        const size_t work_group_size = sycl_utils::get_work_group_size(*this->queue_ptr_);
        const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;

        sycl_utils::events transform_events;

        {
            // memory allocation
            if (this->linearlized_->size() < N) {
                this->linearlized_->resize(N);
            }
            auto fill_event = this->queue_ptr_->submit([&](sycl::handler& h) {
                // get pointers
                const auto linearlized_ptr = this->linearlized_->data();
                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_global_id(0);
                    if (i >= N) return;
                    linearlized_ptr[i].b.setZero();
                    linearlized_ptr[i].H.setZero();
                    linearlized_ptr[i].error = 0.0f;
                });
            });
            fill_event.wait();

            this->cur_T_->data()[0] = result.T.matrix();
            this->max_distance_->at(0) = max_dist_2;

            for (size_t iter = 0; iter < this->params_.max_iterations; ++iter) {
                prev_T = result.T;
                (*this->cur_T_)[0] = result.T.matrix();

                // nearest neighbor search
                auto knn_event = target_tree.knn_search_async<1>(traits::pointcloud::points_ptr(transform_source),
                                                                 traits::pointcloud::size(transform_source), 1,
                                                                 (*this->neighbors_)[0], transform_events.evs);

                // linearlize
                factor::LinearlizedResult linearlized;
                linearlized.H.setZero();
                linearlized.b.setZero();
                linearlized.error = 0.0f;
                {
                    auto linearlize_event = this->queue_ptr_->submit([&](sycl::handler& h) {
                        // get pointers
                        const auto linearlized_ptr = this->linearlized_->data();
                        const auto max_dist_ptr = this->max_distance_->data();
                        const auto cur_T_ptr = this->cur_T_->data();

                        const auto source_ptr = traits::pointcloud::points_ptr(source);
                        const auto transform_source_ptr = traits::pointcloud::points_ptr(transform_source);
                        const auto transform_source_cov_ptr = traits::pointcloud::covs_ptr(transform_source);
                        const auto target_ptr = traits::pointcloud::points_ptr(target);
                        const auto target_cov_ptr = traits::pointcloud::covs_ptr(target);

                        const auto neighbors_index_ptr = (*this->neighbors_)[0].indices->data();
                        const auto neighbors_distances_ptr = (*this->neighbors_)[0].distances->data();

                        // wait for knn search
                        h.depends_on(knn_event.evs);

                        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                            const size_t i = item.get_global_id(0);
                            if (i >= N) return;

                            if (neighbors_distances_ptr[i] > max_dist_ptr[0]) {
                                linearlized_ptr[i].H.setZero();
                                linearlized_ptr[i].b.setZero();
                                linearlized_ptr[i].error = 0.0f;
                            } else {
                                if constexpr (icp == factor::ICPType::GICP) {
                                    linearlized_ptr[i] = factor::linearlize_gicp(
                                        cur_T_ptr[0], source_ptr[i], transform_source_ptr[i],
                                        transform_source_cov_ptr[i], target_ptr[neighbors_index_ptr[i]],
                                        target_cov_ptr[neighbors_index_ptr[i]]);
                                } else if constexpr (icp == factor::ICPType::POINT_TO_POINT) {
                                    linearlized_ptr[i] = factor::linearlize_point_to_point(
                                        cur_T_ptr[0], source_ptr[i], transform_source_ptr[i],
                                        transform_source_cov_ptr[i], target_ptr[neighbors_index_ptr[i]],
                                        target_cov_ptr[neighbors_index_ptr[i]]);
                                }
                            }
                        });
                    });
                    linearlize_event.wait();

                    // reduction
                    for (size_t i = 0; i < N; ++i) {
                        linearlized.H += (*this->linearlized_)[i].H;
                        linearlized.b += (*this->linearlized_)[i].b;
                        linearlized.error += (*this->linearlized_)[i].error;
                    }
                }

                // if (this->params_.optimize_lm) {
                // }
                // else
                {
                    const Eigen::Matrix<float, 6, 1> delta =
                        (linearlized.H + lambda * Eigen::Matrix<float, 6, 6>::Identity()).ldlt().solve(-linearlized.b);
                    // const auto delta = eigen_utils::solve_system_6x6(
                    //     linearlized.H + lambda * Eigen::Matrix<float, 6, 6>::Identity(), -linearlized.b);
                    result.converged = this->is_converged(delta);
                    result.T = result.T * eigen_utils::lie::se3_exp(delta);
                    result.iterations = iter;
                    result.H = linearlized.H;
                    result.b = linearlized.b;
                    result.error = linearlized.error;
                    if (verbose) {
                        std::cout << "iter [" << iter << "] ";
                        std::cout << "error: " << linearlized.error << ", ";
                        std::cout << "dt: " << delta.tail<3>().norm() << ", ";
                        std::cout << "dr: " << delta.head<3>().norm() << std::endl;
                    }
                }
                // transform source points
                transform_events =
                    transform_sycl_async(transform_source, result.T.matrix() * prev_T.matrix().inverse());  // zero copy
                if (result.converged) {
                    break;
                }
            }
            transform_events.wait();
        }
        return result;
    }

private:
    RegistrationParams params_;
    bool is_converged(const Eigen::Matrix<float, 6, 1>& delta) const {
        return delta.template head<3>().norm() < this->params_.rotation_eps &&
               delta.template tail<3>().norm() < this->params_.translation_eps;
    }
    std::shared_ptr<sycl::queue> queue_ptr_ = nullptr;
    std::shared_ptr<shared_vector<TransformMatrix>> cur_T_ = nullptr;
    std::shared_ptr<shared_vector<float>> max_distance_ = nullptr;
    std::shared_ptr<shared_vector<KNNResultSYCL>> neighbors_ = nullptr;
    std::shared_ptr<shared_vector<factor::LinearlizedResult>> linearlized_ = nullptr;
};

}  // namespace algorithms

}  // namespace sycl_points
