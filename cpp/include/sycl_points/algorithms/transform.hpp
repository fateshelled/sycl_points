#pragma once

#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace transform {

namespace kernel {

SYCL_EXTERNAL inline void transform_covs(const Covariance& cov, Covariance& result,
                                         const std::array<sycl::vec<float, 4>, 4>& trans) {
    // trans * cov * trans.T
    const Eigen::Matrix4f trans_mat = eigen_utils::from_sycl_vec(trans);
    const Eigen::Matrix4f trans_mat_T = eigen_utils::transpose<4, 4>(trans_mat);
    const Eigen::Matrix4f ret = eigen_utils::multiply<4, 4, 4>(trans_mat, eigen_utils::multiply<4, 4, 4>(cov, trans_mat_T));
    eigen_utils::copy<4, 4>(ret, result);
}

SYCL_EXTERNAL inline void transform_normal(const Normal& normal, Normal& result,
                                           const std::array<sycl::vec<float, 4>, 4>& trans) {
    const Eigen::Matrix4f trans_mat = eigen_utils::from_sycl_vec(trans);
    const Normal ret = eigen_utils::multiply<4, 4>(trans_mat, normal);
    eigen_utils::normalize<4>(ret);
    eigen_utils::copy<4, 1>(ret, result);
}

SYCL_EXTERNAL inline void transform_point(const PointType& point, PointType& result,
                                          const std::array<sycl::vec<float, 4>, 4>& trans) {
    const Eigen::Matrix4f trans_mat = eigen_utils::from_sycl_vec(trans);
    const PointType ret = eigen_utils::multiply<4, 4>(trans_mat, point);
    eigen_utils::copy<4, 1>(ret, result);
}

}  // namespace kernel

/// @brief Async transform point cloud
/// @param cloud Point Cloud
/// @param trans transform matrix
/// @return events
inline sycl_utils::events transform_async(PointCloudShared& cloud, const TransformMatrix& trans) {
    const size_t N = cloud.size();
    if (N == 0) return sycl_utils::events();

    const size_t work_group_size = cloud.queue.get_work_group_size();
    const size_t global_size = cloud.queue.get_global_size(N);
    const auto trans_vec = eigen_utils::to_sycl_vec(trans);

    sycl_utils::events events;
    if (cloud.has_cov()) {
        const auto covs = cloud.covs_ptr();

        /* Transform Covariance */
        events += cloud.queue.ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;
                kernel::transform_covs(covs[i], covs[i], trans_vec);
            });
        });
    }

    if (cloud.has_normal()) {
        const auto normals = cloud.normals_ptr();

        /* Transform Normals */
        events += cloud.queue.ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;
                kernel::transform_normal(normals[i], normals[i], trans_vec);
            });
        });
    }

    {
        const auto point_ptr = cloud.points_ptr();

        /* Transform Points*/
        events += cloud.queue.ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;
                kernel::transform_point(point_ptr[i], point_ptr[i], trans_vec);
            });
        });
    }

    return events;
}

/// @brief Transform point cloud
/// @param cloud Point Cloud
/// @param trans transform matrix
inline void transform(PointCloudShared& cloud, const TransformMatrix& trans) { transform_async(cloud, trans).wait_and_throw(); }

/// @brief Transform point cloud
/// @param cloud Point Cloud
/// @param trans transform matrix
/// @return Transformed Point Cloud
PointCloudShared transform_copy(const PointCloudShared& cloud, const TransformMatrix& trans) {
    std::shared_ptr<PointCloudShared> ret = std::make_shared<PointCloudShared>(cloud.queue);
    ret->resize_points(cloud.size());
    if (cloud.size() == 0) {
        return *ret;
    }

    sycl_utils::events events;
    if (cloud.has_cov()) {
        ret->resize_covs(cloud.size());
        events += cloud.queue.ptr->submit([&](sycl::handler& h) {
            const auto covs = cloud.covs_ptr();
            const auto output_covs = ret->covs_ptr();
            h.memcpy(output_covs, covs, cloud.size() * sizeof(Covariance));
        });
    }
    events += cloud.queue.ptr->memcpy(ret->points_ptr(), cloud.points_ptr(), cloud.size() * sizeof(PointType));
    events.wait();

    transform(*ret, trans);
    return *ret;
}

/// @brief Transform on CPU
/// @param cloud Point Cloud
/// @param trans transform matrix
inline void transform_cpu(PointCloudShared& cloud, const TransformMatrix& trans) {
    const size_t N = cloud.size();
    if (N == 0) return;

    for (size_t i = 0; i < N; ++i) {
        (*cloud.points)[i] = trans * (*cloud.points)[i];
    }
    if (cloud.has_cov()) {
        const TransformMatrix trans_T = trans.transpose();
        for (size_t i = 0; i < N; ++i) {
            (*cloud.covs)[i] = trans * (*cloud.covs)[i] * trans_T;
        }
    }
}

/// @brief Transform on CPU
/// @param cloud Point Cloud
/// @param trans transform matrix
/// @return Transformed Point Cloud
sycl_points::PointCloudShared transform_cpu_copy(sycl_points::PointCloudShared& cloud, const TransformMatrix& trans) {
    sycl_points::PointCloudShared ret(cloud);  // copy
    transform_cpu(ret, trans);
    return ret;
}

}  // namespace transform

}  // namespace algorithms

}  // namespace sycl_points
