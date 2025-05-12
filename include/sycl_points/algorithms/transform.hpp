#pragma once

#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace transform {

namespace kernel {

SYCL_EXTERNAL inline void transform_covs(const Covariance& cov, Covariance& result, const sycl::vec<float, 4>* trans) {
    const auto cov_vec = eigen_utils::to_sycl_vec(eigen_utils::transpose<4, 4>(cov));

    {
        const sycl::vec<float, 4> tmp0(sycl::dot(trans[0], cov_vec[0]), sycl::dot(trans[0], cov_vec[1]),
                                       sycl::dot(trans[0], cov_vec[2]), sycl::dot(trans[0], cov_vec[3]));
        result(0, 0) = sycl::dot(tmp0, trans[0]);
        result(0, 1) = sycl::dot(tmp0, trans[1]);
        result(0, 2) = sycl::dot(tmp0, trans[2]);
        result(0, 3) = sycl::dot(tmp0, trans[3]);
    }

    {
        const sycl::vec<float, 4> tmp1(sycl::dot(trans[1], cov_vec[0]), sycl::dot(trans[1], cov_vec[1]),
                                       sycl::dot(trans[1], cov_vec[2]), sycl::dot(trans[1], cov_vec[3]));
        result(1, 0) = sycl::dot(tmp1, trans[0]);
        result(1, 1) = sycl::dot(tmp1, trans[1]);
        result(1, 2) = sycl::dot(tmp1, trans[2]);
        result(1, 3) = sycl::dot(tmp1, trans[3]);
    }

    {
        const sycl::vec<float, 4> tmp2(sycl::dot(trans[2], cov_vec[0]), sycl::dot(trans[2], cov_vec[1]),
                                       sycl::dot(trans[2], cov_vec[2]), sycl::dot(trans[2], cov_vec[3]));
        result(2, 0) = sycl::dot(tmp2, trans[0]);
        result(2, 1) = sycl::dot(tmp2, trans[1]);
        result(2, 2) = sycl::dot(tmp2, trans[2]);
        result(2, 3) = sycl::dot(tmp2, trans[3]);
    }

    {
        const sycl::vec<float, 4> tmp3(sycl::dot(trans[3], cov_vec[0]), sycl::dot(trans[3], cov_vec[1]),
                                       sycl::dot(trans[3], cov_vec[2]), sycl::dot(trans[3], cov_vec[3]));
        result(3, 0) = sycl::dot(tmp3, trans[0]);
        result(3, 1) = sycl::dot(tmp3, trans[1]);
        result(3, 2) = sycl::dot(tmp3, trans[2]);
        result(3, 3) = sycl::dot(tmp3, trans[3]);
    }
}

SYCL_EXTERNAL inline void transform_point(const PointType& point, PointType& result, const sycl::vec<float, 4>* trans) {
    const auto pt = eigen_utils::to_sycl_vec(point);
    result[0] = sycl::dot(trans[0], pt);
    result[1] = sycl::dot(trans[1], pt);
    result[2] = sycl::dot(trans[2], pt);
    result[3] = 1.0f;
}

}  // namespace kernel

/// @brief Async transform point cloud
/// @param cloud Point Cloud
/// @param trans transform matrix
/// @return events
inline sycl_utils::events transform_sycl_async(PointCloudShared& cloud, const TransformMatrix& trans) {
    const size_t N = cloud.size();
    if (N == 0) return sycl_utils::events();

    shared_vector<sycl::vec<float, 4>> trans_vec_shared(4, shared_allocator<TransformMatrix>(*cloud.queue.ptr));

#pragma unroll 4
    for (size_t i = 0; i < 4; ++i) {
#pragma unroll 4
        for (size_t j = 0; j < 4; ++j) {
            trans_vec_shared[i][j] = trans(i, j);
        }
    }

    const size_t work_group_size = cloud.queue.work_group_size;
    const size_t global_size = cloud.queue.get_global_size(N);

    sycl_utils::events events;
    if (cloud.has_cov()) {
        const auto covs = cloud.covs_ptr();
        const auto trans_vec_ptr = trans_vec_shared.data();

        /* Transform Covariance */
        events += cloud.queue.ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;
                kernel::transform_covs(covs[i], covs[i], trans_vec_ptr);
            });
        });
    }

    {
        const auto point_ptr = cloud.points_ptr();
        const auto trans_vec_ptr = trans_vec_shared.data();

        /* Transform Points*/
        events += cloud.queue.ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;
                kernel::transform_point(point_ptr[i], point_ptr[i], trans_vec_ptr);
            });
        });
    }

    return events;
}

/// @brief Transform point cloud
/// @param cloud Point Cloud
/// @param trans transform matrix
inline void transform_sycl(PointCloudShared& cloud, const TransformMatrix& trans) {
    transform_sycl_async(cloud, trans).wait();
}

/// @brief Transform point cloud
/// @param cloud Point Cloud
/// @param trans transform matrix
/// @return Transformed Point Cloud
PointCloudShared transform_sycl_copy(const PointCloudShared& cloud, const TransformMatrix& trans) {
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

    transform_sycl(*ret, trans);
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
