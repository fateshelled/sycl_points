#pragma once

#include <Eigen/Dense>
#include <sycl_points/points/container_traits.hpp>
#include <sycl_points/utils/sycl_utils.hpp>
#include <vector>

namespace sycl_points {

namespace traits {

template <>
struct PointContainerTraits<PointContainerShared> {
    static constexpr bool is_shared() { return true; }
    static constexpr bool is_device() { return false; }
    static size_t size(const PointContainerShared& pc) { return pc.size(); }
    static const PointType* const_data_ptr(const PointContainerShared& pc) { return pc.data(); }
    static PointType* data_ptr(const PointContainerShared& pc) { return const_cast<PointType*>(pc.data()); }
    static void resize(PointContainerShared& pc, size_t N) { pc.resize(N); }
};

template <>
struct PointContainerTraits<PointContainerDevice> {
    static constexpr bool is_shared() { return false; }
    static constexpr bool is_device() { return true; }
    static size_t size(const PointContainerDevice& pc) { return pc.size; }
    static const PointType* const_data_ptr(const PointContainerDevice& pc) { return pc.device_ptr; }
    static PointType* data_ptr(PointContainerDevice& pc) { return pc.device_ptr; }
    static void resize(PointContainerDevice& pc, size_t N) { pc.resize(N); }
};

template <>
struct CovarianceContainerTraits<CovarianceContainerShared> {
    static constexpr bool is_shared() { return true; }
    static constexpr bool is_device() { return false; }
    static size_t size(const CovarianceContainerShared& cc) { return cc.size(); }
    static const Covariance* const_data_ptr(const CovarianceContainerShared& cc) { return cc.data(); }
    static Covariance* data_ptr(CovarianceContainerShared& cc) { return const_cast<Covariance*>(cc.data()); }
    static void resize(CovarianceContainerShared& cc, size_t N) { cc.resize(N); }
};

template <>
struct CovarianceContainerTraits<CovarianceContainerDevice> {
    static constexpr bool is_shared() { return false; }
    static constexpr bool is_device() { return true; }
    static size_t size(const CovarianceContainerDevice& cc) { return cc.size; }
    static const Covariance* const_data_ptr(const CovarianceContainerDevice& cc) { return cc.device_ptr; }
    static Covariance* data_ptr(CovarianceContainerDevice& cc) { return cc.device_ptr; }
    static void resize(CovarianceContainerDevice& cc, size_t N) { cc.resize(N); }
};


}  // namespace traits
}  // namespace sycl_points
