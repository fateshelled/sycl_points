#pragma once

#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/algorithms/knn_search.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace color_gradient {

namespace kernel {

SYCL_EXTERNAL inline RGBType RGB_to_XYZ(const RGBType& rgb) {
    const float lr = rgb.x() < 0.04045f ? rgb.x() / 12.92f : std::pow((rgb.x() + 0.055f) / 1.055f, 2.4f);
    const float lg = rgb.y() < 0.04045f ? rgb.y() / 12.92f : std::pow((rgb.y() + 0.055f) / 1.055f, 2.4f);
    const float lb = rgb.z() < 0.04045f ? rgb.z() / 12.92f : std::pow((rgb.z() + 0.055f) / 1.055f, 2.4f);

    const float x = 0.4124f * lr + 0.3576f * lg + 0.1805f * lb;
    const float y = 0.2126f * lr + 0.7152f * lg + 0.0722f * lb;
    const float z = 0.0193f * lr + 0.1192f * lg + 0.9505f * lb;

    return RGBType(x, y, z, 1.0f);
}

SYCL_EXTERNAL inline RGBType XYZ_to_Lab(const Eigen::Vector4f& xyz) {
    // D65
    const float x = xyz.x() / 95.047f;
    const float y = xyz.y() / 100.000f;
    const float z = xyz.z() / 108.883f;

    const float fx = x > 0.008856f ? std::pow(x, 1.0f / 3.0f) : 7.787f * x + 16.0f / 116.0f;
    const float fy = y > 0.008856f ? std::pow(y, 1.0f / 3.0f) : 7.787f * y + 16.0f / 116.0f;
    const float fz = z > 0.008856f ? std::pow(z, 1.0f / 3.0f) : 7.787f * z + 16.0f / 116.0f;

    const float l = 116.0f * fy - 16.0f;
    const float a = 500.0 * (fx - fy);
    const float b = 200.0 * (fy - fz);
    return RGBType(l, a, b, 1.0f);
}

SYCL_EXTERNAL inline RGBType RGB_to_Lab(const RGBType& rgb) {
    const auto xyz = RGB_to_XYZ(rgb);
    return XYZ_to_Lab(xyz);
}

SYCL_EXTERNAL inline void compute_gradient(ColorGradient& ret, const PointType* points, const RGBType* colors,
                                           const int32_t* index_ptr, size_t k, size_t i) {
    const auto p0 = points[i];
    const auto c0 = colors[i];

    Eigen::Matrix3f A = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f B = Eigen::Matrix3f::Zero();
    for (size_t j = 0; j < k; ++j) {
        const auto idx = index_ptr[i * k + j];
        const auto pj = points[idx];
        const auto cj = colors[idx];

        const Eigen::Vector3f dp(pj[0] - p0[0], pj[1] - p0[1], pj[2] - p0[2]);
        const Eigen::Vector3f dc(cj[0] - c0[0], cj[1] - c0[1], cj[2] - c0[2]);

        eigen_utils::add_inplace<3, 3>(A, eigen_utils::outer<3>(dp, dp));
        eigen_utils::add_inplace<3, 3>(B, eigen_utils::outer<3>(dp, dc));
    }

    const Eigen::Matrix3f L = eigen_utils::cholesky_3x3(A);
    for (size_t c = 0; c < 3; ++c) {
        const Eigen::Vector3f b = B.col(c);
        const Eigen::Vector3f g = eigen_utils::solve_cholesky_3x3(L, b);
        ret(c, 0) = g(0);
        ret(c, 1) = g(1);
        ret(c, 2) = g(2);
    }
}

}  // namespace kernel

inline sycl_utils::events compute_color_gradients_async(const PointCloudShared& cloud,
                                                        const knn_search::KNNResult& neighbors) {
    const size_t N = cloud.size();
    if (!cloud.has_rgb()) {
        throw std::runtime_error("RGB field not found");
    }

    if (cloud.color_gradients->size() != N) {
        cloud.resize_color_gradients(N);
    }
    if (N == 0) return sycl_utils::events();

    const size_t work_group_size = cloud.queue.get_work_group_size();
    const size_t global_size = cloud.queue.get_global_size(N);

    sycl_utils::events events;
    events += cloud.queue.ptr->submit([&](sycl::handler& h) {
        const auto pt_ptr = cloud.points_ptr();
        const auto color_ptr = cloud.rgb_ptr();
        const auto grad_ptr = cloud.color_gradients_ptr();
        const auto index_ptr = neighbors.indices->data();
        const size_t k = neighbors.k;
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t idx = item.get_global_id(0);
            if (idx >= N) return;

            kernel::compute_gradient(grad_ptr[idx], pt_ptr, color_ptr, index_ptr, k, idx);
        });
    });
    return events;
}

inline sycl_utils::events compute_color_gradients_async(const PointCloudShared& cloud, const knn_search::KDTree& kdtree,
                                                        size_t k_correspondences) {
    const auto neighbors = kdtree.knn_search(cloud, k_correspondences);
    return compute_color_gradients_async(cloud, neighbors);
}

}  // namespace color_gradient

}  // namespace algorithms

}  // namespace sycl_points
