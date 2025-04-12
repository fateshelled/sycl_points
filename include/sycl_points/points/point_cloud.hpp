#pragma once

#include <sycl_points/points/container.hpp>
#include <sycl_points/points/traits.hpp>
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

    bool has_cov() const { return this->covs->size() > 0; }

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
    std::shared_ptr<sycl::queue> queue_ptr = nullptr;
    const sycl::property_list propeties = {sycl::property::no_init()};

    std::shared_ptr<PointContainerShared> points = nullptr;
    std::shared_ptr<CovarianceContainerShared> covs = nullptr;

    PointCloudShared(const std::shared_ptr<sycl::queue>& q) : queue_ptr(q) {
        const PointAllocatorShared alloc_pc(*this->queue_ptr, this->propeties);
        this->points = std::make_shared<PointContainerShared>(0, alloc_pc);

        const CovarianceAllocatorShared alloc_cov(*this->queue_ptr, this->propeties);
        this->covs = std::make_shared<CovarianceContainerShared>(0, alloc_cov);
    }

    PointCloudShared(const std::shared_ptr<sycl::queue>& q, const PointCloudCPU& cpu) : queue_ptr(q) {
        const PointAllocatorShared alloc_pc(*this->queue_ptr, this->propeties);
        this->points = std::make_shared<PointContainerShared>(cpu.size(), alloc_pc);

        for (size_t i = 0; i < cpu.points->size(); ++i) {
            (*this->points)[i] = (*cpu.points)[i];
        }

        const CovarianceAllocatorShared alloc_cov(*this->queue_ptr, this->propeties);
        this->covs = std::make_shared<CovarianceContainerShared>(cpu.covs->size(), alloc_cov);
        for (size_t i = 0; i < cpu.covs->size(); ++i) {
            (*this->covs)[i] = (*cpu.covs)[i];
        }
    }

    // copy constructor
    PointCloudShared(const PointCloudShared& other) : queue_ptr(other.queue_ptr) {
        const CovarianceAllocatorShared alloc_cov(*this->queue_ptr, this->propeties);
        const PointAllocatorShared alloc_pc(*this->queue_ptr, this->propeties);

        const size_t N = other.size();
        sycl::event copy_cov_event;
        if (other.has_cov()) {
            this->covs = std::make_shared<CovarianceContainerShared>(N, alloc_cov);
            copy_cov_event = this->queue_ptr->memcpy(this->covs->data(), other.covs->data(), N * sizeof(Covariance));
        } else {
            this->covs = std::make_shared<CovarianceContainerShared>(0, alloc_cov);
        }
        this->points = std::make_shared<PointContainerShared>(N, alloc_pc);
        sycl::event copy_pt_event =
            this->queue_ptr->memcpy(this->points->data(), other.points->data(), N * sizeof(PointType));
        copy_cov_event.wait();
        copy_pt_event.wait();
    }

    ~PointCloudShared() {}

    size_t size() const { return this->points->size(); }
    bool has_cov() const { return this->covs->size() > 0; }
    PointType* points_ptr() const { return this->points->data(); }
    Covariance* covs_ptr() const { return this->covs->data(); }

    void resize_points(size_t N) const { this->points->resize(N); }
    void resize_covs(size_t N) const { this->covs->resize(N); }

    // sycl_utils::events copy_to_cpu_async(PointCloudCPU& cpu) const {
    //     sycl_utils::events events;
    //     cpu.points.resize(this->points->size());
    //     events.push_back(
    //         this->queue_ptr->memcpy(cpu.points.data(), this->points->data(), this->points->size() *
    //         sizeof(PointType)));
    //     if (this->has_cov()) {
    //         cpu.covs.resize(this->covs->size());
    //         events.push_back(
    //             this->queue_ptr->memcpy(cpu.covs.data(), this->covs->data(), this->covs->size() *
    //             sizeof(Covariance)));
    //     }
    //     return events;
    // }
};

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
    PointType* points_ptr() const { return this->points->device_ptr; }
    Covariance* covs_ptr() const { return this->covs->device_ptr; }

    void resize_points(size_t N) const { this->points->allocate(N); }
    void resize_covs(size_t N) const { this->covs->allocate(N); }
};

// inline PointCloudShared device_to_shared(const PointCloudDevice& device) {
//     PointCloudShared shared(device.queue_ptr);

//     const PointAllocatorShared alloc_pc(*shared.queue_ptr, shared.propeties);
//     shared.points = std::make_shared<PointContainerShared>(device.points->size, alloc_pc);

//     const CovarianceAllocatorShared alloc_cov(*shared.queue_ptr, shared.propeties);
//     shared.covs = std::make_shared<CovarianceContainerShared>(device.covs->size, alloc_cov);

//     auto copy_cov_event =
//         device.queue_ptr->memcpy(shared.covs->data(), device.covs->device_ptr, device.covs->size *
//         sizeof(Covariance));
//     auto copy_pt_event = device.queue_ptr->memcpy(shared.points->data(), device.points->device_ptr,
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

//     auto copy_cov_event = shared.queue_ptr->memcpy(device.covs->device_ptr, shared.covs->data(),
//                                                    shared.covs->size() * sizeof(Covariance));
//     auto copy_pt_event = shared.queue_ptr->memcpy(device.points->device_ptr, shared.points->data(),
//                                                   shared.points->size() * sizeof(PointType));
//     copy_cov_event.wait();
//     copy_pt_event.wait();
//     return device;
// }

namespace traits {
// template <>
// struct Traits<PointCloudCPU> {
//     static std::shared_ptr<sycl::queue> queue_ptr(const PointCloudCPU& pc) { return nullptr; }
//     static size_t size(const PointCloudCPU& pc) { return pc.size(); }
//     static bool has_cov(const PointCloudCPU& pc) { return pc.has_cov(); }
//     static PointType* points_ptr(const PointCloudCPU& pc) { return pc.points->data(); }
//     static Covariance* covs_ptr(const PointCloudCPU& pc) { return pc.covs->data(); }
//     static void resize_points(PointCloudCPU& pc, size_t N) { pc.points->resize(N); }
//     static void resize_covs(PointCloudCPU& pc, size_t N) { pc.covs->resize(N); }
// };

template <>
struct Traits<PointCloudShared> {
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
struct Traits<PointCloudDevice> {
    static std::shared_ptr<PointCloudDevice>  constructor(const std::shared_ptr<sycl::queue>& queue_ptr) {
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
