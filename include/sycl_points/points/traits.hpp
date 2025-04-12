#pragma once

#include <Eigen/Dense>
#include <sycl_points/utils/sycl_utils.hpp>
#include <type_traits>
#include <vector>

namespace sycl_points {

namespace traits {

template <typename T>
struct Traits;

template <typename T>
inline auto constructor(const std::shared_ptr<sycl::queue>& queue_ptr) {
    return Traits<std::remove_cv_t<T>>::constructor(queue_ptr);
}

template <typename T>
inline std::shared_ptr<sycl::queue> queue_ptr(const T& cloud) {
    return Traits<T>::queue_ptr(cloud);
}

template <typename T>
inline size_t size(const T& cloud) {
    return Traits<T>::size(cloud);
}

template <typename T>
inline bool has_cov(const T& cloud) {
    return Traits<T>::has_cov(cloud);
}

template <typename T>
inline auto points_ptr(const T& cloud) {
    return Traits<T>::points_ptr(cloud);
}

template <typename T>
inline auto covs_ptr(const T& cloud) {
    return Traits<T>::covs_ptr(cloud);
}

template <typename T>
inline void resize_points(const T& cloud, size_t N) {
    Traits<T>::resize_points(cloud, N);
    // Traits<std::remove_cv_t<T>>::
}

template <typename T>
inline void resize_covs(const T& cloud, size_t N) {
    Traits<T>::resize_covs(cloud, N);
}

}  // namespace traits

}  // namespace sycl_points
