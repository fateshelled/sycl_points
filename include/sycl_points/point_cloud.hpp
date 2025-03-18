#pragma once

#include <Eigen/Dense>
#include <vector>
#include "eigen_utils.hpp"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace sycl_points {

using baseT = float;

template <typename T = baseT>
using PointType = Eigen::Vector4<T>;

template <typename T = baseT>
using Covariance = Eigen::Matrix4<T>;

template <typename T = baseT>
using TransformMatrix = Eigen::Matrix4<T>;


template <typename T>
using host_allocator = sycl::usm_allocator<T, sycl::usm::alloc::host>;

template <typename T>
using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename allocator, typename T = baseT>
using PointContainer = std::vector<PointType<T>, allocator>;

template <typename T = baseT>
using PointContainerCPU = std::vector<PointType<T>, Eigen::aligned_allocator<PointType<T>>>;

template <typename T = baseT>
using PointContainerHost = std::vector<PointType<T>, host_allocator<PointType<T>>>;

template <typename T = baseT>
using PointContainerShared = std::vector<PointType<T>, shared_allocator<T>>;

template <typename allocator, typename T = baseT>
using CovarianceContainer = std::vector<Covariance<T>, allocator>;

template <typename T = baseT>
using CovarianceContainerCPU = std::vector<Covariance<T>, Eigen::aligned_allocator<Covariance<T>>>;

template <typename T = baseT>
using CovarianceContainerHost = std::vector<Covariance<T>, host_allocator<Covariance<T>>>;

template <typename T = baseT>
using CovarianceContainerShared = std::vector<Covariance<T>, shared_allocator<Covariance<T>>>;


template <typename T = float>
struct PointCloudCPU {
  PointContainerCPU<T> points;
  CovarianceContainerCPU<T> covs;

  PointCloudCPU() {}

  size_t size() const { return this->points.size(); }

  bool has_cov() const { return this->covs.size() > 0; }

  void transform(const Eigen::Matrix4<T>& trans) {
    const size_t N = this->points.size();

    PointCloudCPU<T> transformed(N);
    for (size_t i = 0; i < N; ++i) {
      points[i] = trans * this->points[i];
    }
    if (this->has_cov()) {
      const Eigen::Matrix4<T> trans_T = trans.transpose();
      for (size_t i = 0; i < N; ++i) {
        this->covs[i] = trans * this->covs[i] * trans_T;
      }
    }
  };

  PointCloudCPU<T> transform_copy(const Eigen::Matrix4<T>& trans) {
    const size_t N = this->points.size();

    PointCloudCPU<T> transformed;
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



template <typename T = float>
SYCL_EXTERNAL inline void transform_covs (const Covariance<T>& civ, Covariance<T>& result, const TransformMatrix<T>& trans) {
  Covariance<T> tmp;
  TransformMatrix<T> dev_trans_T;
  eigen_utils::matrixMultiply<T, 4, 4>(trans, civ, tmp);
  eigen_utils::matrixTranspose<T, 4, 4>(trans, dev_trans_T);
  eigen_utils::matrixMultiply<T, 4, 4>(tmp, dev_trans_T, result);
};

template <typename T = float>
SYCL_EXTERNAL inline void transform_point (const PointType<T>& point, PointType<T>& result, const TransformMatrix<T>& trans) {
  eigen_utils::matrixVectorMultiply<T, 4, 4>(trans, point, result);
};

template <typename T = float>
struct PointCloudDevice{
  PointType<T>* points_ptr = nullptr;
  Covariance<T>* covs_ptr = nullptr;
  size_t size = 0;
  std::shared_ptr<sycl::queue> queue_ptr = nullptr;
  sycl::event points_to_device_event;
  sycl::event covs_to_device_event;

  PointCloudDevice() {}
  PointCloudDevice(sycl::queue& q, PointCloudCPU<T>& cpu, const bool wait = false): size(cpu.size()), queue_ptr(std::make_shared<sycl::queue>(q)) {
    if (cpu.has_cov()) {
      this->covs_ptr = sycl::malloc_device<Covariance<T>>(this->size, *this->queue_ptr);
      covs_to_device_event = this->queue_ptr->memcpy(this->covs_ptr, cpu.covs.data(), this->size * sizeof(Covariance<T>));
    } else {
      this->covs_ptr = nullptr;
    }
    this->points_ptr = sycl::malloc_device<PointType<T>>(this->size, *this->queue_ptr);
    points_to_device_event = this->queue_ptr->memcpy(this->points_ptr, cpu.points.data(), this->size * sizeof(PointType<T>));

    if (wait) {
      covs_to_device_event.wait();
      points_to_device_event.wait();
    }
  }

  ~PointCloudDevice() {
    if (this->points_ptr) {
      sycl::free(this->points_ptr, *this->queue_ptr);
      this->points_ptr = nullptr;
    }
    if (this->covs_ptr) {
      sycl::free(this->covs_ptr, *this->queue_ptr);
      this->covs_ptr = nullptr;
    }
  }

  bool has_cov() const {
    return this->covs_ptr;
  }

  std::vector<sycl::event> copyToHost(PointCloudCPU<T>& cpu) {
    std::vector<sycl::event> events;
    cpu.points.resize(this->size);
    events.push_back(this->queue_ptr->memcpy(cpu.points.data(), this->points_ptr, this->size * sizeof(PointType<T>)));
    if(this->has_cov()) {
      cpu.covs.resize(this->size);
      events.push_back(this->queue_ptr->memcpy(cpu.covs.data(), this->covs_ptr, this->size * sizeof(Covariance<T>)));
    }
    return events;
  }

  void transform_copy(const Eigen::Matrix4<T>& trans, PointCloudDevice& ret) {
    const size_t N = this->size;
    /* allocate device memory */
    PointType<T>* dev_result = sycl::malloc_device<PointType<T>>(N, *this->queue_ptr);
    TransformMatrix<T>* dev_trans = sycl::malloc_shared<TransformMatrix<T>>(1, *this->queue_ptr);

    PointType<T>* dev_points = sycl::malloc_device<PointType<T>>(N, *this->queue_ptr);
    Covariance<T>* dev_covs = sycl::malloc_device<Covariance<T>>(N, *this->queue_ptr);

    /* copy to device */
    auto points_copy_event = this->queue_ptr->memcpy(dev_points, this->points_ptr, N * sizeof(PointType<T>));
    sycl::event covs_copy_event;
    auto trans_copy_event = this->queue_ptr->memcpy(dev_trans, trans.data(), sizeof(Eigen::Matrix4<T>));
    Covariance<T>* dev_result_cov;
    if (this->has_cov()) {
      covs_copy_event = this->queue_ptr->memcpy(dev_covs, this->covs_ptr, N * sizeof(Covariance<T>));
      dev_result_cov = sycl::malloc_device<Covariance<T>>(N, *this->queue_ptr);
    }

    sycl::event covs_trans_event;
    if (this->has_cov()) {

      /* Transform Covariance */
      covs_trans_event = this->queue_ptr->submit([&](sycl::handler& h) {
        h.depends_on({trans_copy_event, this->covs_to_device_event, covs_copy_event});
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          transform_covs(dev_covs[i], dev_result_cov[i], *dev_trans);
        });
      });
    }

    sycl::event points_trans_event;
    {
      /* Transform Points*/
      points_trans_event = this->queue_ptr->submit([&](sycl::handler& h) {
        h.depends_on({trans_copy_event, this->points_to_device_event, points_copy_event});
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          transform_point(dev_points[i], dev_result[i], *dev_trans);
        });
      });
    }

    covs_trans_event.wait();
    points_trans_event.wait();

    {
      ret.points_ptr = dev_result;
      dev_result = nullptr;
      ret.queue_ptr = this->queue_ptr;
      ret.size = N;
      if (this->has_cov()) {
        ret.covs_ptr = dev_result_cov;
        dev_result_cov = nullptr;
      }
    }

    // free
    sycl::free(dev_trans, *this->queue_ptr);
    sycl::free(dev_points, *this->queue_ptr);
    sycl::free(dev_covs, *this->queue_ptr);
  };
};

}  // namespace sycl_points
