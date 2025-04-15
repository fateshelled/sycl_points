#pragma once

#include <sycl_points/points/container.hpp>
#include <sycl_points/points/point_cloud_traits.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

struct PointCloudCPU {
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

template <typename PointContainer, typename CovarianceContainer, typename PointAllocator, typename CovarianceAllocator>
struct PointCloudSYCL {
    std::shared_ptr<sycl::queue> queue_ptr = nullptr;
    const sycl::property_list propeties = {sycl::property::no_init()};

    std::shared_ptr<PointContainer> points = nullptr;
    std::shared_ptr<CovarianceContainer> covs = nullptr;

    PointCloudSYCL(const std::shared_ptr<sycl::queue>& q) : queue_ptr(q) {
        const PointAllocator alloc_pc(*this->queue_ptr, this->propeties);
        this->points = std::make_shared<PointContainer>(0, alloc_pc);

        const CovarianceAllocator alloc_cov(*this->queue_ptr, this->propeties);
        this->covs = std::make_shared<CovarianceContainer>(0, alloc_cov);
    }

    // copy from cpu
    PointCloudSYCL(const std::shared_ptr<sycl::queue>& q, const PointCloudCPU& cpu) : queue_ptr(q) {
        const CovarianceAllocator alloc_cov(*this->queue_ptr, this->propeties);
        const PointAllocator alloc_pc(*this->queue_ptr, this->propeties);

        const size_t N = cpu.size();
        sycl::event copy_cov_event;
        if (cpu.has_cov()) {
            this->covs = std::make_shared<CovarianceContainer>(N, alloc_cov);
            if (N > 0) {
                if (!sycl_utils::is_cpu(*queue_ptr) && traits::covariance::is_shared<CovarianceContainer>()) {
                    for (size_t i = 0; i < N; ++i) {
                        (*this->covs)[i] = (*cpu.covs)[i];
                    }
                } else {
                    copy_cov_event =
                        this->queue_ptr->memcpy(this->covs->data(), cpu.covs->data(), N * sizeof(Covariance));
                }
            }
        } else {
            this->covs = std::make_shared<CovarianceContainer>(0, alloc_cov);
        }
        this->points = std::make_shared<PointContainer>(N, alloc_pc);
        sycl::event copy_pt_event;
        if (N > 0) {
            if (!sycl_utils::is_cpu(*queue_ptr) && traits::point::is_shared<PointContainer>()) {
                for (size_t i = 0; i < N; ++i) {
                    (*this->points)[i] = (*cpu.points)[i];
                }
            } else {
                copy_pt_event =
                    this->queue_ptr->memcpy(this->points->data(), cpu.points->data(), N * sizeof(PointType));
            }
        }
        copy_cov_event.wait();
        copy_pt_event.wait();
    }

    // copy constructor
    PointCloudSYCL(const PointCloudSYCL& other) : queue_ptr(other.queue_ptr) {
        const CovarianceAllocator alloc_cov(*this->queue_ptr, this->propeties);
        const PointAllocator alloc_pc(*this->queue_ptr, this->propeties);

        const size_t N = other.size();
        sycl::event copy_cov_event;
        if (other.has_cov()) {
            this->covs = std::make_shared<CovarianceContainer>(N, alloc_cov);
            if (other.covs->size() > 0) {
                copy_cov_event =
                    this->queue_ptr->memcpy(this->covs->data(), other.covs->data(), N * sizeof(Covariance));
            }
        } else {
            this->covs = std::make_shared<CovarianceContainer>(0, alloc_cov);
        }
        this->points = std::make_shared<PointContainer>(N, alloc_pc);
        sycl::event copy_pt_event;
        if (N > 0) {
            copy_pt_event = this->queue_ptr->memcpy(this->points->data(), other.points->data(), N * sizeof(PointType));
        }
        copy_cov_event.wait();
        copy_pt_event.wait();
    }

    ~PointCloudSYCL() {}

    size_t size() const { return this->points->size(); }
    bool has_cov() const { return this->covs->size() > 0; }
    PointType* points_ptr() const { return this->points->data(); }
    Covariance* covs_ptr() const { return this->covs->data(); }

    void resize_points(size_t N) const { this->points->resize(N); }
    void resize_covs(size_t N) const { this->covs->resize(N); }
};

using PointCloudShared =
    PointCloudSYCL<PointContainerShared, CovarianceContainerShared, PointAllocatorShared, CovarianceAllocatorShared>;

struct PointCloudDevice {
    std::shared_ptr<sycl::queue> queue_ptr = nullptr;
    const sycl::property_list propeties = {sycl::property::no_init()};

    std::shared_ptr<PointContainerDevice> points = nullptr;
    std::shared_ptr<CovarianceContainerDevice> covs = nullptr;

    PointCloudDevice(const std::shared_ptr<sycl::queue>& q) : queue_ptr(q) {}

    ~PointCloudDevice() { this->free(); }

    void free() {
        this->points->free();
        this->covs->free();
        this->points = nullptr;
        this->covs = nullptr;
    }

    size_t size() const { return this->points->size; }
    bool has_cov() const { return this->covs->size > 0; }
    PointType* points_ptr() const { return this->points->data; }
    Covariance* covs_ptr() const { return this->covs->data; }

    void resize_points(size_t N) const { this->points->resize(N); }
    void resize_covs(size_t N) const { this->covs->resize(N); }
};

// inline PointCloudShared device_to_shared(const PointCloudDevice& device) {
//     PointCloudShared shared(device.queue_ptr);

//     const PointAllocatorShared alloc_pc(*shared.queue_ptr, shared.propeties);
//     shared.points = std::make_shared<PointContainerShared>(device.points->size, alloc_pc);

//     const CovarianceAllocatorShared alloc_cov(*shared.queue_ptr, shared.propeties);
//     shared.covs = std::make_shared<CovarianceContainerShared>(device.covs->size, alloc_cov);

//     auto copy_cov_event =
//         device.queue_ptr->memcpy(shared.covs->data(), device.covs->data, device.covs->size *
//         sizeof(Covariance));
//     auto copy_pt_event = device.queue_ptr->memcpy(shared.points->data(), device.points->data,
//                                                   device.points->size * sizeof(PointType));
//     copy_cov_event.wait();
//     copy_pt_event.wait();
//     return shared;
// }

// inline PointCloudDevice shared_to_device(const PointCloudShared& shared) {
//     PointCloudDevice device(shared.queue_ptr);
//     device.points = std::make_shared<PointContainerDevice>(shared.queue_ptr);
//     device.covs = std::make_shared<CovarianceContainerDevice>(shared.queue_ptr);
//     device.points->allocate(shared.points->size());
//     device.covs->allocate(shared.covs->size());

//     auto copy_cov_event = shared.queue_ptr->memcpy(device.covs->data, shared.covs->data(),
//                                                    shared.covs->size() * sizeof(Covariance));
//     auto copy_pt_event = shared.queue_ptr->memcpy(device.points->data, shared.points->data(),
//                                                   shared.points->size() * sizeof(PointType));
//     copy_cov_event.wait();
//     copy_pt_event.wait();
//     return device;
// }

namespace traits {

template <>
struct PointCloudTraits<PointCloudShared> {
    static constexpr bool is_shared() { return true; }
    static constexpr bool is_device() { return false; }
    static std::shared_ptr<PointCloudShared> constructor(const std::shared_ptr<sycl::queue>& queue_ptr) {
        return std::make_shared<PointCloudShared>(queue_ptr);
    }
    static std::shared_ptr<sycl::queue> queue_ptr(const PointCloudShared& pc) { return pc.queue_ptr; }
    static size_t size(const PointCloudShared& pc) { return pc.size(); }
    static bool has_cov(const PointCloudShared& pc) { return pc.has_cov(); }
    static PointType* points_ptr(const PointCloudShared& pc) { return pc.points_ptr(); }
    static Covariance* covs_ptr(const PointCloudShared& pc) { return pc.covs_ptr(); }
    static void resize_points(const PointCloudShared& pc, size_t N) { pc.resize_points(N); }
    static void resize_covs(const PointCloudShared& pc, size_t N) { pc.resize_covs(N); }
};

template <>
struct PointCloudTraits<PointCloudDevice> {
    static constexpr bool is_shared() { return false; }
    static constexpr bool is_device() { return true; }
    static std::shared_ptr<PointCloudDevice> constructor(const std::shared_ptr<sycl::queue>& queue_ptr) {
        return std::make_shared<PointCloudDevice>(queue_ptr);
    }
    static std::shared_ptr<sycl::queue> queue_ptr(const PointCloudDevice& pc) { return pc.queue_ptr; }
    static size_t size(const PointCloudDevice& pc) { return pc.size(); }
    static bool has_cov(const PointCloudDevice& pc) { return pc.has_cov(); }
    static PointType* points_ptr(const PointCloudDevice& pc) { return pc.points_ptr(); }
    static Covariance* covs_ptr(const PointCloudDevice& pc) { return pc.covs_ptr(); }
    static void resize_points(const PointCloudDevice& pc, size_t N) { pc.resize_points(N); }
    static void resize_covs(const PointCloudDevice& pc, size_t N) { pc.resize_covs(N); }
};
}  // namespace traits

}  // namespace sycl_points
