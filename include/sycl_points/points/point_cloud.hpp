#pragma once

#include <sycl_points/points/types.hpp>

namespace sycl_points {

struct PointCloudCPU {
    using Ptr = std::shared_ptr<PointCloudCPU>;
    using ConstPtr = std::shared_ptr<PointCloudCPU>;

    std::shared_ptr<PointContainerCPU> points = nullptr;
    std::shared_ptr<CovarianceContainerCPU> covs = nullptr;

    PointCloudCPU() {
        this->points = std::make_shared<PointContainerCPU>();
        this->covs = std::make_shared<CovarianceContainerCPU>();
    }

    size_t size() const { return this->points->size(); }

    bool has_cov() const { return this->covs != nullptr && this->covs->size() > 0; }

    void transform(const TransformMatrix& trans) {
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
    };

    PointCloudCPU transform_copy(const TransformMatrix& trans) const {
        const size_t N = this->points->size();

        PointCloudCPU transformed;
        transformed.points->resize(N);
        for (size_t i = 0; i < N; ++i) {
            (*transformed.points)[i] = trans * (*this->points)[i];
        }
        if (this->has_cov()) {
            transformed.covs->resize(N);
            for (size_t i = 0; i < N; ++i) {
                (*transformed.covs)[i] = trans * (*this->covs)[i] * trans.transpose();
            }
        }

        return transformed;
    };
};

struct PointCloudShared {
    using Ptr = std::shared_ptr<PointCloudShared>;
    using ConstPtr = std::shared_ptr<PointCloudShared>;

    std::shared_ptr<sycl::queue> queue_ptr = nullptr;
    const sycl::property_list propeties = {
        // sycl::property::no_init()
    };

    std::shared_ptr<PointContainerShared> points = nullptr;
    std::shared_ptr<CovarianceContainerShared> covs = nullptr;

    PointCloudShared(const std::shared_ptr<sycl::queue>& q) : queue_ptr(q) {
        const PointAllocatorShared alloc_pc(*this->queue_ptr, this->propeties);
        this->points = std::make_shared<PointContainerShared>(0, alloc_pc);

        const CovarianceAllocatorShared alloc_cov(*this->queue_ptr, this->propeties);
        this->covs = std::make_shared<CovarianceContainerShared>(0, alloc_cov);
    }

    // copy from cpu
    PointCloudShared(const std::shared_ptr<sycl::queue>& q, const PointCloudCPU& cpu) : queue_ptr(q) {
        const CovarianceAllocatorShared alloc_cov(*this->queue_ptr, this->propeties);
        const PointAllocatorShared alloc_pc(*this->queue_ptr, this->propeties);

        const size_t N = cpu.size();

        if (sycl_utils::is_cpu(*this->queue_ptr)) {
            sycl_utils::events copy_events;

            if (cpu.has_cov()) {
                this->covs = std::make_shared<CovarianceContainerShared>(N, alloc_cov);
                if (N > 0) {
                    copy_events +=
                        this->queue_ptr->memcpy(this->covs->data(), cpu.covs->data(), N * sizeof(Covariance));
                }
            } else {
                this->covs = std::make_shared<CovarianceContainerShared>(0, alloc_cov);
            }

            this->points = std::make_shared<PointContainerShared>(N, alloc_pc);
            if (N > 0) {
                copy_events += this->queue_ptr->memcpy(this->points->data(), cpu.points->data(), N * sizeof(PointType));
            }
            copy_events.wait();
        } else {
            if (cpu.has_cov()) {
                this->covs = std::make_shared<CovarianceContainerShared>(N, alloc_cov);
                for (size_t i = 0; i < N; ++i) {
                    this->covs->data()[i] = cpu.covs->data()[i];
                }
            } else {
                this->covs = std::make_shared<CovarianceContainerShared>(0, alloc_cov);
            }

            this->points = std::make_shared<PointContainerShared>(N, alloc_pc);
            for (size_t i = 0; i < N; ++i) {
                this->points->data()[i] = cpu.points->data()[i];
            }
        }
    }

    // copy constructor
    PointCloudShared(const PointCloudShared& other) : queue_ptr(other.queue_ptr) {
        const CovarianceAllocatorShared alloc_cov(*this->queue_ptr, this->propeties);
        const PointAllocatorShared alloc_pc(*this->queue_ptr, this->propeties);

        const size_t N = other.size();
        sycl_utils::events copy_events;

        if (other.has_cov()) {
            this->covs = std::make_shared<CovarianceContainerShared>(N, alloc_cov);
            if (other.covs->size() > 0) {
                copy_events += this->queue_ptr->memcpy(this->covs->data(), other.covs->data(), N * sizeof(Covariance));
            }
        } else {
            this->covs = std::make_shared<CovarianceContainerShared>(0, alloc_cov);
        }

        this->points = std::make_shared<PointContainerShared>(N, alloc_pc);
        if (N > 0) {
            copy_events += this->queue_ptr->memcpy(this->points->data(), other.points->data(), N * sizeof(PointType));
        }

        copy_events.wait();
    }

    ~PointCloudShared() {}

    size_t size() const { return this->points->size(); }
    bool has_cov() const { return this->covs->size() > 0; }
    PointType* points_ptr() const { return this->points->data(); }
    Covariance* covs_ptr() const { return this->covs->data(); }

    void resize_points(size_t N) const { this->points->resize(N); }
    void resize_covs(size_t N) const { this->covs->resize(N); }
};

}  // namespace sycl_points
