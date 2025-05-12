#pragma once

#include <sycl_points/points/types.hpp>

namespace sycl_points {

/// @brief CPU point cloud class with point and covariance container.
struct PointCloudCPU {
    using Ptr = std::shared_ptr<PointCloudCPU>;
    using ConstPtr = std::shared_ptr<PointCloudCPU>;

    /// @brief point container
    std::shared_ptr<PointContainerCPU> points = nullptr;
    /// @brief covariance container
    std::shared_ptr<CovarianceContainerCPU> covs = nullptr;

    /// @brief Constructor
    PointCloudCPU() {
        this->points = std::make_shared<PointContainerCPU>();
        this->covs = std::make_shared<CovarianceContainerCPU>();
    }

    /// @brief number of point
    size_t size() const { return this->points->size(); }
    /// @brief has covariance or not
    bool has_cov() const { return this->covs != nullptr && this->covs->size() > 0; }
};

/// @brief Shared memory point cloud class with point and covariance container.
struct PointCloudShared {
    using Ptr = std::shared_ptr<PointCloudShared>;
    using ConstPtr = std::shared_ptr<PointCloudShared>;

    /// @brief SYCL queue
    sycl_utils::DeviceQueue queue;
    /// @brief point container
    std::shared_ptr<PointContainerShared> points = nullptr;
    /// @brief covariance container
    std::shared_ptr<CovarianceContainerShared> covs = nullptr;

    /// @brief Constructor
    /// @param queue SYCL queue
    PointCloudShared(const sycl_utils::DeviceQueue& q) : queue(q) {
        this->points = std::make_shared<PointContainerShared>(0, *this->queue.ptr);
        this->covs = std::make_shared<CovarianceContainerShared>(0, *this->queue.ptr);
    }

    /// @brief Copy from CPU point cloud
    /// @param queue SYCL queue
    /// @param cpu CPU point cloud
    PointCloudShared(const sycl_utils::DeviceQueue& q, const PointCloudCPU& cpu) : queue(q) {
        const size_t N = cpu.size();

        sycl_utils::events copy_events;
        const bool is_cpu = sycl_utils::is_cpu(*this->queue.ptr);

        if (cpu.has_cov()) {
            this->covs = std::make_shared<CovarianceContainerShared>(N, *this->queue.ptr);
            if (is_cpu) {
                copy_events += this->queue.ptr->memcpy(this->covs->data(), cpu.covs->data(), N * sizeof(Covariance));
            } else {
                for (size_t i = 0; i < N; ++i) {
                    this->covs->data()[i] = cpu.covs->data()[i];
                }
            }
        } else {
            this->covs = std::make_shared<CovarianceContainerShared>(0, *this->queue.ptr);
        }

        this->points = std::make_shared<PointContainerShared>(N, *this->queue.ptr);
        if (N > 0) {
            if (is_cpu) {
                copy_events += this->queue.ptr->memcpy(this->points->data(), cpu.points->data(), N * sizeof(PointType));
            } else {
                for (size_t i = 0; i < N; ++i) {
                    this->points->data()[i] = cpu.points->data()[i];
                }
            }
        }
        copy_events.wait();
    }

    /// @brief Copy from other shared point cloud
    /// @param other shared point cloud
    PointCloudShared(const PointCloudShared& other) : queue(other.queue) {
        const size_t N = other.size();
        sycl_utils::events copy_events;

        if (other.has_cov()) {
            this->covs = std::make_shared<CovarianceContainerShared>(N, *this->queue.ptr);
            if (other.covs->size() > 0) {
                copy_events += this->queue.ptr->memcpy(this->covs->data(), other.covs->data(), N * sizeof(Covariance));
            }
        } else {
            this->covs = std::make_shared<CovarianceContainerShared>(0, *this->queue.ptr);
        }

        this->points = std::make_shared<PointContainerShared>(N, *this->queue.ptr);
        if (N > 0) {
            copy_events += this->queue.ptr->memcpy(this->points->data(), other.points->data(), N * sizeof(PointType));
        }

        copy_events.wait();
    }

    /// @brief destructor
    ~PointCloudShared() {}

    /// @brief number of points
    size_t size() const { return this->points->size(); }
    /// @brief has covariance or not
    bool has_cov() const { return this->covs->size() > 0; }
    /// @brief pointer of points
    PointType* points_ptr() const { return this->points->data(); }
    /// @brief pointer of covariances
    Covariance* covs_ptr() const { return this->covs->data(); }

    /// @brief resize point container
    /// @param N size
    void resize_points(size_t N) const { this->points->resize(N); }
    /// @brief resize covariance container
    /// @param N size
    void resize_covs(size_t N) const { this->covs->resize(N); }
};

}  // namespace sycl_points
