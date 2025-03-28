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

SYCL_EXTERNAL inline void transform_covs(const Covariance& cov, Covariance& result, const TransformMatrix& trans) {
  Covariance tmp;
  const TransformMatrix trans_T = eigen_utils::transpose<4, 4>(trans);
  // eigen_utils::multiply<4, 4>(trans, cov, tmp);
  // eigen_utils::multiply<4, 4>(tmp, trans_T, result);

  tmp(0, 0) = trans(0, 0) * cov(0, 0) + trans(0, 1) * cov(1, 0) + trans(0, 2) * cov(2, 0) + trans(0, 3) * cov(3, 0);
  tmp(0, 1) = trans(0, 0) * cov(0, 1) + trans(0, 1) * cov(1, 1) + trans(0, 2) * cov(2, 1) + trans(0, 3) * cov(3, 1);
  tmp(0, 2) = trans(0, 0) * cov(0, 2) + trans(0, 1) * cov(1, 2) + trans(0, 2) * cov(2, 2) + trans(0, 3) * cov(3, 2);
  tmp(0, 3) = trans(0, 0) * cov(0, 3) + trans(0, 1) * cov(1, 3) + trans(0, 2) * cov(2, 3) + trans(0, 3) * cov(3, 3);

  tmp(1, 0) = trans(1, 0) * cov(0, 0) + trans(1, 1) * cov(1, 0) + trans(1, 2) * cov(2, 0) + trans(1, 3) * cov(3, 0);
  tmp(1, 1) = trans(1, 0) * cov(0, 1) + trans(1, 1) * cov(1, 1) + trans(1, 2) * cov(2, 1) + trans(1, 3) * cov(3, 1);
  tmp(1, 2) = trans(1, 0) * cov(0, 2) + trans(1, 1) * cov(1, 2) + trans(1, 2) * cov(2, 2) + trans(1, 3) * cov(3, 2);
  tmp(1, 3) = trans(1, 0) * cov(0, 3) + trans(1, 1) * cov(1, 3) + trans(1, 2) * cov(2, 3) + trans(1, 3) * cov(3, 3);

  tmp(2, 0) = trans(2, 0) * cov(0, 0) + trans(2, 1) * cov(1, 0) + trans(2, 2) * cov(2, 0) + trans(2, 3) * cov(3, 0);
  tmp(2, 1) = trans(2, 0) * cov(0, 1) + trans(2, 1) * cov(1, 1) + trans(2, 2) * cov(2, 1) + trans(2, 3) * cov(3, 1);
  tmp(2, 2) = trans(2, 0) * cov(0, 2) + trans(2, 1) * cov(1, 2) + trans(2, 2) * cov(2, 2) + trans(2, 3) * cov(3, 2);
  tmp(2, 3) = trans(2, 0) * cov(0, 3) + trans(2, 1) * cov(1, 3) + trans(2, 2) * cov(2, 3) + trans(2, 3) * cov(3, 3);

  tmp(3, 0) = trans(3, 0) * cov(0, 0) + trans(3, 1) * cov(1, 0) + trans(3, 2) * cov(2, 0) + trans(3, 3) * cov(3, 0);
  tmp(3, 1) = trans(3, 0) * cov(0, 1) + trans(3, 1) * cov(1, 1) + trans(3, 2) * cov(2, 1) + trans(3, 3) * cov(3, 1);
  tmp(3, 2) = trans(3, 0) * cov(0, 2) + trans(3, 1) * cov(1, 2) + trans(3, 2) * cov(2, 2) + trans(3, 3) * cov(3, 2);
  tmp(3, 3) = trans(3, 0) * cov(0, 3) + trans(3, 1) * cov(1, 3) + trans(3, 2) * cov(2, 3) + trans(3, 3) * cov(3, 3);


  result(0, 0) = cov(0, 0) * trans_T(0, 0) + cov(0, 1) * trans_T(1, 0) + cov(0, 2) * trans_T(2, 0) + cov(0, 3) * trans_T(3, 0);
  result(0, 1) = cov(0, 0) * trans_T(0, 1) + cov(0, 1) * trans_T(1, 1) + cov(0, 2) * trans_T(2, 1) + cov(0, 3) * trans_T(3, 1);
  result(0, 2) = cov(0, 0) * trans_T(0, 2) + cov(0, 1) * trans_T(1, 2) + cov(0, 2) * trans_T(2, 2) + cov(0, 3) * trans_T(3, 2);
  result(0, 3) = cov(0, 0) * trans_T(0, 3) + cov(0, 1) * trans_T(1, 3) + cov(0, 2) * trans_T(2, 3) + cov(0, 3) * trans_T(3, 3);

  result(1, 0) = cov(1, 0) * trans_T(0, 0) + cov(1, 1) * trans_T(1, 0) + cov(1, 2) * trans_T(2, 0) + cov(1, 3) * trans_T(3, 0);
  result(1, 1) = cov(1, 0) * trans_T(0, 1) + cov(1, 1) * trans_T(1, 1) + cov(1, 2) * trans_T(2, 1) + cov(1, 3) * trans_T(3, 1);
  result(1, 2) = cov(1, 0) * trans_T(0, 2) + cov(1, 1) * trans_T(1, 2) + cov(1, 2) * trans_T(2, 2) + cov(1, 3) * trans_T(3, 2);
  result(1, 3) = cov(1, 0) * trans_T(0, 3) + cov(1, 1) * trans_T(1, 3) + cov(1, 2) * trans_T(2, 3) + cov(1, 3) * trans_T(3, 3);

  result(2, 0) = cov(2, 0) * trans_T(0, 0) + cov(2, 1) * trans_T(1, 0) + cov(2, 2) * trans_T(2, 0) + cov(2, 3) * trans_T(3, 0);
  result(2, 1) = cov(2, 0) * trans_T(0, 1) + cov(2, 1) * trans_T(1, 1) + cov(2, 2) * trans_T(2, 1) + cov(2, 3) * trans_T(3, 1);
  result(2, 2) = cov(2, 0) * trans_T(0, 2) + cov(2, 1) * trans_T(1, 2) + cov(2, 2) * trans_T(2, 2) + cov(2, 3) * trans_T(3, 2);
  result(2, 3) = cov(2, 0) * trans_T(0, 3) + cov(2, 1) * trans_T(1, 3) + cov(2, 2) * trans_T(2, 3) + cov(2, 3) * trans_T(3, 3);

  result(3, 0) = cov(3, 0) * trans_T(0, 0) + cov(3, 1) * trans_T(1, 0) + cov(3, 2) * trans_T(2, 0) + cov(3, 3) * trans_T(3, 0);
  result(3, 1) = cov(3, 0) * trans_T(0, 1) + cov(3, 1) * trans_T(1, 1) + cov(3, 2) * trans_T(2, 1) + cov(3, 3) * trans_T(3, 1);
  result(3, 2) = cov(3, 0) * trans_T(0, 2) + cov(3, 1) * trans_T(1, 2) + cov(3, 2) * trans_T(2, 2) + cov(3, 3) * trans_T(3, 2);
  result(3, 3) = cov(3, 0) * trans_T(0, 3) + cov(3, 1) * trans_T(1, 3) + cov(3, 2) * trans_T(2, 3) + cov(3, 3) * trans_T(3, 3);
};

SYCL_EXTERNAL inline void transform_point(const PointType& point, PointType& result, const TransformMatrix& trans) {
  // eigen_utils::multiply<float, 4, 4>(trans, point, result); // to slow
  result[0] = trans(0, 0) * point[0] + trans(0, 1) * point[1] + trans(0, 2) * point[2] + trans(0, 3);
  result[1] = trans(1, 0) * point[0] + trans(1, 1) * point[1] + trans(1, 2) * point[2] + trans(1, 3);
  result[2] = trans(2, 0) * point[0] + trans(2, 1) * point[1] + trans(2, 2) * point[2] + trans(2, 3);
  result[3] = 1.0f;
};

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

  // transform on device
  void transform_sycl(const TransformMatrix& trans) {
    const size_t N = this->points->size();

    TransformMatrix* trans_shared = sycl::malloc_shared<TransformMatrix>(1, *this->queue_ptr);
    trans_shared[0] = trans;

    sycl::event covs_trans_event;
    if (this->has_cov()) {
      auto covs = (*this->covs).data();
      /* Transform Covariance */
      covs_trans_event = this->queue_ptr->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          transform_covs(covs[i], covs[i], trans_shared[0]);
        });
      });
    }

    sycl::event points_trans_event;
    {
      auto points = (*this->points).data();
      /* Transform Points*/
      points_trans_event = this->queue_ptr->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          transform_point(points[i], points[i], trans_shared[0]);
        });
      });
    }

    covs_trans_event.wait();
    points_trans_event.wait();

    // free
    sycl::free(trans_shared, *this->queue_ptr);
  }

  // transform on device (too slow)
  PointCloudShared transform_sycl_copy(const TransformMatrix& trans) const {
    PointCloudShared ret(*this);  // copy
    ret.transform_sycl(trans);
    return ret;
  };
};

}  // namespace sycl_points
