#pragma once

#include <sycl_points/points/types.hpp>
#include <sycl_points/utils/sycl_utils.hpp>
#include <type_traits>

namespace sycl_points {

namespace traits {

template <typename T>
struct PointContainerTraits;
namespace point {

template <typename T>
constexpr bool is_shared() {
    return PointContainerTraits<std::remove_cv_t<T>>::is_shared();
}

template <typename T>
constexpr bool is_device() {
    return PointContainerTraits<std::remove_cv_t<T>>::is_device();
}

template <typename T>
size_t size(const T& container) {
    return PointContainerTraits<T>::size(container);
}

template <typename T>
const PointType* const_data_ptr(const T& container) {
    return PointContainerTraits<T>::const_data_ptr(container);
}

template <typename T>
PointType* data_ptr(const T& container) {
    return PointContainerTraits<T>::data_ptr(container);
}

template <typename T>
void resize(T& container, size_t N) {
    PointContainerTraits<T>::resize(container, N);
}
}  // namespace point

template <typename T>
struct CovarianceContainerTraits;

namespace covariance {
template <typename T>
constexpr bool is_shared() {
    return CovarianceContainerTraits<std::remove_cv_t<T>>::is_shared();
}

template <typename T>
constexpr bool is_device() {
    return CovarianceContainerTraits<std::remove_cv_t<T>>::is_device();
}

template <typename T>
size_t size(const T& container) {
    return CovarianceContainerTraits<T>::size(container);
}

template <typename T>
const Covariance* const_data_ptr(const T& container) {
    return CovarianceContainerTraits<T>::const_data_ptr(container);
}

template <typename T>
Covariance* data_ptr(T& container) {
    return CovarianceContainerTraits<T>::data_ptr(container);
}

template <typename T>
void resize(T& container, size_t N) {
    CovarianceContainerTraits<T>::resize(container, N);
}

}  // namespace covariance

}  // namespace traits

}  // namespace sycl_points
