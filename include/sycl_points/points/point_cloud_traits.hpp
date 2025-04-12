#pragma once

#include <Eigen/Dense>
#include <sycl_points/utils/sycl_utils.hpp>
#include <type_traits>
#include <vector>

namespace sycl_points {

namespace traits {

template <typename T>
struct PointCloudTraits;

namespace pointcloud {

template <typename T>
bool is_shared() {
    return PointCloudTraits<std::remove_cv_t<T>>::is_shared();
}

template <typename T>
bool is_device() {
    return PointCloudTraits<std::remove_cv_t<T>>::is_device();
}

template <typename T>
auto constructor(const std::shared_ptr<sycl::queue>& queue_ptr) {
    return PointCloudTraits<std::remove_cv_t<T>>::constructor(queue_ptr);
}

template <typename T>
std::shared_ptr<sycl::queue> queue_ptr(const T& cloud) {
    return PointCloudTraits<T>::queue_ptr(cloud);
}

template <typename T>
size_t size(const T& cloud) {
    return PointCloudTraits<T>::size(cloud);
}

template <typename T>
bool has_cov(const T& cloud) {
    return PointCloudTraits<T>::has_cov(cloud);
}

template <typename T>
auto points_ptr(const T& cloud) {
    return PointCloudTraits<T>::points_ptr(cloud);
}

template <typename T>
auto covs_ptr(const T& cloud) {
    return PointCloudTraits<T>::covs_ptr(cloud);
}

template <typename T>
void resize_points(const T& cloud, size_t N) {
    PointCloudTraits<T>::resize_points(cloud, N);
}

template <typename T>
void resize_covs(const T& cloud, size_t N) {
    PointCloudTraits<T>::resize_covs(cloud, N);
}

}  // namespace pointcloud

}  // namespace traits

}  // namespace sycl_points
