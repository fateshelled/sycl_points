#pragma once

#include "point_cloud.hpp"
#include "point_cloud_reader.hpp"
#include "knn_search.hpp"
#include "downsampling.hpp"
#include "covariance.hpp"

namespace sycl_points {

struct RegistrationParams {
  size_t max_iterations = 20;
  //   size_t max_inner_iterations = 10; // LM method
  float lambda = 1e-6f;
  //   float lambda_factor = 10.0f; // LM method
  bool lm_method = false;
  float max_correspondence_distance = 1.0f;
  float translation_eps = 1e-3f;
  float rotation_eps = 1e-3f;
  bool verbose = true;
};

struct LinearlizedResult {
  Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
  Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
  float error = 0.0;
};

struct RegistrationResult {
  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
  bool converged = false;
  size_t iterations = 0;
  Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
  Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
  float error = 0.0;
};

namespace factor {

SYCL_EXTERNAL inline LinearlizedResult
linearlize_point_to_point(const TransformMatrix& T, const PointType& source, const PointType& target, const Covariance& source_cov, const Covariance& target_cov) {
  LinearlizedResult ret;
  return ret;
}
SYCL_EXTERNAL inline LinearlizedResult
linearlize_gicp(const TransformMatrix& T, const PointType& source, const PointType& target, const Covariance& source_cov, const Covariance& target_cov) {
  LinearlizedResult ret;

  Covariance mahalanobis = Covariance::Zero();
  {
    const Eigen::Matrix3f RCR = eigen_utils::matrixAdd<3, 3>(eigen_utils::block3x3(source_cov), eigen_utils::block3x3(target_cov));
    const Eigen::Matrix3f RCR_inv = eigen_utils::inverse(RCR);

    mahalanobis(0, 0) = RCR_inv(0, 0);
    mahalanobis(0, 1) = RCR_inv(0, 1);
    mahalanobis(0, 2) = RCR_inv(0, 2);
    mahalanobis(1, 0) = RCR_inv(1, 0);
    mahalanobis(1, 1) = RCR_inv(1, 1);
    mahalanobis(1, 2) = RCR_inv(1, 2);
    mahalanobis(2, 0) = RCR_inv(2, 0);
    mahalanobis(2, 1) = RCR_inv(2, 1);
    mahalanobis(2, 2) = RCR_inv(2, 2);
  }

  const PointType residual(target.x() - source.x(), target.y() - source.y(), target.z() - source.z(), 0.0f);

  Eigen::Matrix<float, 4, 6> J = Eigen::Matrix<float, 4, 6>::Zero();
  {
    const Eigen::Matrix3f skewed = eigen_utils::skew(source);
    const Eigen::Matrix3f T_3x3 = eigen_utils::block3x3(T);
    Eigen::Matrix3f T_skewed = eigen_utils::matrixMultiply<3, 3, 3>(T_3x3, skewed);
    J(0, 0) = T_skewed(0, 0);
    J(0, 1) = T_skewed(0, 1);
    J(0, 2) = T_skewed(0, 2);
    J(1, 0) = T_skewed(1, 0);
    J(1, 1) = T_skewed(1, 1);
    J(1, 2) = T_skewed(1, 2);
    J(2, 0) = T_skewed(2, 0);
    J(2, 1) = T_skewed(2, 1);
    J(2, 2) = T_skewed(2, 2);

    J(0, 3) = -T(0, 0);
    J(0, 4) = -T(0, 1);
    J(0, 5) = -T(0, 2);
    J(1, 3) = -T(1, 0);
    J(1, 4) = -T(1, 1);
    J(1, 5) = -T(1, 2);
    J(2, 3) = -T(2, 0);
    J(2, 4) = -T(2, 1);
    J(2, 5) = -T(2, 2);
  }

  Eigen::Matrix<float, 6, 4> J_T_mah = eigen_utils::matrixMultiply<6, 4, 4>(eigen_utils::matrixTranspose<4, 6>(J), mahalanobis);

  // J.transpose() * mahalanobis * J;
  ret.H = eigen_utils::matrixMultiply<6, 4, 6>(J_T_mah, J);
  // J.transpose() * mahalanobis * residual;
  ret.b = eigen_utils::matrixVectorMultiply<6, 4>(J_T_mah, residual);
  // 0.5 * residual.transpose() * mahalanobis * residual;
  ret.error = 0.5f * (eigen_utils::vectorDot<4>(residual, eigen_utils::matrixVectorMultiply<4, 4>(mahalanobis, residual)));
  return ret;
}

}  // namespace factor

class Registration {
private:
  RegistrationParams params_;
  bool is_converged(const Eigen::Matrix<float, 6, 1>& delta) const {
    return delta.template head<3>().norm() < this->params_.rotation_eps && delta.template tail<3>().norm() < this->params_.translation_eps;
  }

public:
  Registration(const RegistrationParams& params = RegistrationParams()) : params_(params) {}
  RegistrationResult optimize(
    sycl::queue& queue,
    const PointCloudShared& source,
    const PointCloudShared& target,
    const KNNSearchSYCL& target_tree,
    const TransformMatrix& init_T = TransformMatrix::Identity()) const {
    const size_t N = source.size();
    RegistrationResult result;
    result.T.matrix() = init_T;

    Eigen::Isometry3f prev_T = Eigen::Isometry3f::Identity();
    PointCloudShared trans_source = source.transform_cpu_copy(init_T);

    float lambda = this->params_.lambda;
    const float max_dist_2 = this->params_.max_correspondence_distance * this->params_.max_correspondence_distance;
    const auto verbose = this->params_.verbose;

    for (size_t iter = 0; iter < this->params_.max_iterations; ++iter) {
      prev_T = result.T;
      // nearest neighbor search
      const auto neighbors = target_tree.searchKDTree_sycl(trans_source, 1);

      // linearlize
      LinearlizedResult linearlized;
      linearlized.H = Eigen::Matrix<float, 6, 6>::Zero();
      linearlized.b = Eigen::Matrix<float, 6, 1>::Zero();
      linearlized.error = 0.0;
      {
        shared_vector<int> target_indices(N, shared_allocator<int>(queue, {}));
        shared_vector<float> target_distances(N, shared_allocator<int>(queue, {}));
        shared_vector<TransformMatrix> cur_T(1, result.T.matrix(), shared_allocator<TransformMatrix>(queue, {}));
        shared_vector<LinearlizedResult> linearlized_results(N, LinearlizedResult(), shared_allocator<LinearlizedResult>(queue, {}));
        shared_vector<float> max_distance(1, max_dist_2, shared_allocator<float>(queue, {}));

        for (size_t j = 0; j < N; ++j) {
          target_indices[j] = neighbors.indices[j][0];
          target_distances[j] = neighbors.distances[j][0];
        }
        auto indices_ptr = target_indices.data();
        auto distances_ptr = target_distances.data();
        auto max_dist_ptr = max_distance.data();
        auto cur_T_ptr = cur_T.data();

        auto source_ptr = trans_source.points->data();
        auto target_ptr = target.points->data();
        auto source_cov_ptr = trans_source.covs->data();
        auto target_cov_ptr = target.covs->data();

        auto linearlized_ptr = linearlized_results.data();
        auto event = queue.submit([&](sycl::handler& h) {
          h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            const size_t i = idx[0];
            if (distances_ptr[i] > max_dist_ptr[0]) {
              return;
            }
            linearlized_ptr[i] = factor::linearlize_gicp(cur_T_ptr[0], source_ptr[i], target_ptr[indices_ptr[i]], source_cov_ptr[i], target_cov_ptr[indices_ptr[i]]);
            // linearlized_ptr[i] = factor::linearlize_point_to_point(cur_T_ptr[0], source_ptr[i], target_ptr[indices_ptr[i]], source_cov_ptr[i], target_cov_ptr[indices_ptr[i]]);

            // reduction here ?
          });
        });
        event.wait();

        // reduction
        for (size_t i = 0; i < N; ++i) {
          linearlized.H += linearlized_results[i].H;
          linearlized.b += linearlized_results[i].b;
          linearlized.error += linearlized_results[i].error;
        }
      }

      if (this->params_.lm_method) {
      } else {
        const Eigen::Matrix<float, 6, 1> delta = (linearlized.H + lambda * Eigen::Matrix<float, 6, 6>::Identity()).ldlt().solve(-linearlized.b);
        result.converged = this->is_converged(delta);
        result.T = result.T * eigen_utils::se3_exp(delta);
        result.iterations = iter;
        result.H = linearlized.H;
        result.b = linearlized.b;
        result.error = linearlized.error;
        if (verbose) {
          std::cout << "iter [" << iter << "] ";
          std::cout << "error: " << linearlized.error << ", ";
          std::cout << "dt: " << delta.tail<3>().norm() << ", ";
          std::cout << "dr: " << delta.head<3>().norm() << std::endl;
          // std::cout << "b: " << std::endl << linearlized.b.transpose() << std::endl;
          // std::cout << "H: " << std::endl << linearlized.H << std::endl;
          // std::cout << "trans: " << std::endl << result.T.matrix() << std::endl;
        }
      }
      if (result.converged) {
        break;
      }
      // transform source points
      trans_source.transform_sycl(result.T.matrix() * prev_T.matrix().inverse());  // zero copy
    }
    return result;
  }
};

}  // namespace sycl_points
