#pragma once

#include <Eigen/Dense>
#include <vector>
#include "eigen_utils.hpp"
#include "sycl_utils.hpp"

namespace sycl_points {

using PointType = Eigen::Vector4f;
using Covariance = Eigen::Matrix4f;
using TransformMatrix = Eigen::Matrix4f;

using PointContainerCPU = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
using PointContainerHost = host_vector<PointType>;
using PointContainerShared = shared_vector<PointType>;

using CovarianceContainerCPU = std::vector<Covariance, Eigen::aligned_allocator<Covariance>>;
using CovarianceContainerHost = host_vector<Covariance>;
using CovarianceContainerShared = shared_vector<Covariance>;

struct PointCloudCPU {
  PointContainerCPU points;
  CovarianceContainerCPU covs;

  PointCloudCPU() {}

  size_t size() const { return this->points.size(); }

  bool has_cov() const { return this->covs.size() > 0; }

  void transform(const TransformMatrix& trans) {
    const size_t N = this->points.size();

    for (size_t i = 0; i < N; ++i) {
      this->points[i] = trans * this->points[i];
    }
    if (this->has_cov()) {
      const TransformMatrix trans_T = trans.transpose();
      for (size_t i = 0; i < N; ++i) {
        this->covs[i] = trans * this->covs[i] * trans_T;
      }
    }
  };

  PointCloudCPU transform_copy(const TransformMatrix& trans) {
    const size_t N = this->points.size();

    PointCloudCPU transformed;
    transformed.points.resize(N);
    for (size_t i = 0; i < N; ++i) {
      transformed.points[i] = trans * this->points[i];
    }
    if (this->has_cov()) {
      transformed.covs.resize(N);
      for (size_t i = 0; i < N; ++i) {
        transformed.covs[i] = trans * this->covs[i] * trans.transpose();
      }
    }

    return transformed;
  };
};

SYCL_EXTERNAL inline void transform_covs(const Covariance& cov, Covariance& result, const sycl::vec<float, 4>* trans) {
  const auto cov_vec = eigen_utils::to_sycl_vec(eigen_utils::transpose<4, 4>(cov));

  const sycl::vec<float, 4> tmp0(sycl::dot(trans[0], cov_vec[0]), sycl::dot(trans[0], cov_vec[1]), sycl::dot(trans[0], cov_vec[2]), sycl::dot(trans[0], cov_vec[3]));
  const sycl::vec<float, 4> tmp1(sycl::dot(trans[1], cov_vec[0]), sycl::dot(trans[1], cov_vec[1]), sycl::dot(trans[1], cov_vec[2]), sycl::dot(trans[1], cov_vec[3]));
  const sycl::vec<float, 4> tmp2(sycl::dot(trans[2], cov_vec[0]), sycl::dot(trans[2], cov_vec[1]), sycl::dot(trans[2], cov_vec[2]), sycl::dot(trans[2], cov_vec[3]));
  const sycl::vec<float, 4> tmp3(sycl::dot(trans[3], cov_vec[0]), sycl::dot(trans[3], cov_vec[1]), sycl::dot(trans[3], cov_vec[2]), sycl::dot(trans[3], cov_vec[3]));

  result(0, 0) = sycl::dot(tmp0, trans[0]);
  result(0, 1) = sycl::dot(tmp0, trans[1]);
  result(0, 2) = sycl::dot(tmp0, trans[2]);
  result(0, 3) = sycl::dot(tmp0, trans[3]);

  result(1, 0) = sycl::dot(tmp1, trans[0]);
  result(1, 1) = sycl::dot(tmp1, trans[1]);
  result(1, 2) = sycl::dot(tmp1, trans[2]);
  result(1, 3) = sycl::dot(tmp1, trans[3]);

  result(2, 0) = sycl::dot(tmp2, trans[0]);
  result(2, 1) = sycl::dot(tmp2, trans[1]);
  result(2, 2) = sycl::dot(tmp2, trans[2]);
  result(2, 3) = sycl::dot(tmp2, trans[3]);

  result(3, 0) = sycl::dot(tmp3, trans[0]);
  result(3, 1) = sycl::dot(tmp3, trans[1]);
  result(3, 2) = sycl::dot(tmp3, trans[2]);
  result(3, 3) = sycl::dot(tmp3, trans[3]);
}

SYCL_EXTERNAL inline void transform_point(const PointType& point, PointType& result, const sycl::vec<float, 4>* trans) {
  const auto pt = eigen_utils::to_sycl_vec(point);
  result[0] = sycl::dot(trans[0], pt);
  result[1] = sycl::dot(trans[1], pt);
  result[2] = sycl::dot(trans[2], pt);
  result[3] = 1.0f;
}

struct PointCloudShared {
  std::shared_ptr<PointContainerShared> points = nullptr;
  std::shared_ptr<CovarianceContainerShared> covs = nullptr;
  std::shared_ptr<sycl::queue> queue_ptr = nullptr;
  // PointType* points_device_ptr_ = nullptr;
  // Covariance* covs_device_ptr_ = nullptr;
  sycl::property_list propeties = {sycl::property::no_init()};

  PointCloudShared(sycl::queue& q) : queue_ptr(std::make_shared<sycl::queue>(q)) {
    const sycl_points::shared_allocator<PointContainerShared> alloc_pc(*this->queue_ptr, propeties);
    this->points = std::make_shared<PointContainerShared>(0, alloc_pc);

    const sycl_points::shared_allocator<CovarianceContainerShared> alloc_cov(*this->queue_ptr, propeties);
    this->covs = std::make_shared<CovarianceContainerShared>(0, alloc_cov);
  }

  PointCloudShared(sycl::queue& q, const PointCloudCPU& cpu) : queue_ptr(std::make_shared<sycl::queue>(q)) {
    const sycl_points::shared_allocator<PointContainerShared> alloc_pc(*this->queue_ptr, propeties);
    this->points = std::make_shared<PointContainerShared>(cpu.points.size(), alloc_pc);
    const sycl_points::shared_allocator<CovarianceContainerShared> alloc_cov(*this->queue_ptr, propeties);
    this->covs = std::make_shared<CovarianceContainerShared>(cpu.covs.size(), alloc_cov);

    for (size_t i = 0; i < cpu.points.size(); ++i) {
      (*this->points)[i] = cpu.points[i];
    }

    for (size_t i = 0; i < cpu.covs.size(); ++i) {
      (*this->covs)[i] = cpu.covs[i];
    }
  }

  // copy constructor
  PointCloudShared(const PointCloudShared& other) : queue_ptr(other.queue_ptr) {
    const sycl_points::shared_allocator<PointContainerShared> alloc_pc(*this->queue_ptr, propeties);
    const sycl_points::shared_allocator<CovarianceContainerShared> alloc_cov(*this->queue_ptr, propeties);
    this->points = std::make_shared<PointContainerShared>(*other.points, alloc_pc);
    this->covs = std::make_shared<CovarianceContainerShared>(*other.covs, alloc_cov);
  }

  ~PointCloudShared() {}

  size_t size() const { return this->points->size(); }

  bool has_cov() const { return this->covs->size() > 0; }

  std::vector<sycl::event> copyToCPU(PointCloudCPU& cpu) {
    std::vector<sycl::event> events;
    cpu.points.resize(this->points->size());
    events.push_back(this->queue_ptr->memcpy(cpu.points.data(), this->points->data(), this->points->size() * sizeof(PointType)));
    if (this->has_cov()) {
      cpu.covs.resize(this->covs->size());
      events.push_back(this->queue_ptr->memcpy(cpu.covs.data(), this->covs->data(), this->covs->size() * sizeof(Covariance)));
    }
    return events;
  }

  void transform_cpu(const TransformMatrix& trans) {
    const size_t N = this->points->size();

    for (size_t i = 0; i < N; ++i) {
      (*this->points)[i] = trans * (*this->points)[i];
    }
    if (this->has_cov()) {
      const TransformMatrix trans_T = trans.transpose();
      for (size_t i = 0; i < N; ++i) {
        (*this->covs)[i] = trans * (*this->covs)[i] * trans_T;
      }
    }
  }

  PointCloudShared transform_cpu_copy(const TransformMatrix& trans) const {
    PointCloudShared ret(*this);  // copy
    ret.transform_cpu(trans);
    return ret;
  }

  // async transform on device
  sycl_utils::events transform_sycl_async(const TransformMatrix& trans) {
    const size_t N = this->points->size();

    shared_vector<sycl::vec<float, 4>> trans_vec_shared(4, shared_allocator<TransformMatrix>(*this->queue_ptr));
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        trans_vec_shared[i][j] = trans(i, j);
      }
    }

    // Optimize work group size
    const size_t work_group_size = std::min(sycl_utils::default_work_group_size, (size_t)this->queue_ptr->get_device().get_info<sycl::info::device::max_work_group_size>());
    const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;

    sycl::event covs_trans_event;
    if (this->has_cov()) {
      const auto covs = (*this->covs).data();
      const auto trans_vec_ptr = trans_vec_shared.data();

      /* Transform Covariance */
      covs_trans_event = this->queue_ptr->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(work_group_size)), [=](sycl::nd_item<1> item) {
          const size_t i = item.get_global_id(0);
          if (i >= N) return;
          transform_covs(covs[i], covs[i], trans_vec_ptr);
        });
      });
    }

    sycl::event points_trans_event;
    {
      auto points = (*this->points).data();
      const auto trans_vec_ptr = trans_vec_shared.data();

      /* Transform Points*/
      points_trans_event = this->queue_ptr->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(work_group_size)), [=](sycl::nd_item<1> item) {
          const size_t i = item.get_global_id(0);
          if (i >= N) return;
          // transform_point(points[i], points[i], trans_ptr[0]);
          transform_point(points[i], points[i], trans_vec_ptr);
        });
      });
    }

    sycl_utils::events ev;
    ev.events.push_back(covs_trans_event);
    ev.events.push_back(points_trans_event);
    return ev;
  }

  // transform on device
  void transform_sycl(const TransformMatrix& trans) { transform_sycl_async(trans).wait(); }

  // transform on device (too slow)
  PointCloudShared transform_sycl_copy(const TransformMatrix& trans) const {
    PointCloudShared ret(*this);  // copy
    ret.transform_sycl(trans);
    return ret;
  };
};

}  // namespace sycl_points
