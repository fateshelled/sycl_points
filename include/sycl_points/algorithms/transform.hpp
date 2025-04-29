#pragma once

#include <sycl_points/points/container.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {

namespace algorithms {

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

template <typename PointCloud>
inline sycl_utils::events transform_sycl_async(PointCloud& cloud, const TransformMatrix& trans) {
    const size_t N = traits::pointcloud::size(cloud);
    if (N == 0) return sycl_utils::events();

    const auto queue_ptr = traits::pointcloud::queue_ptr(cloud);

    shared_vector<sycl::vec<float, 4>> trans_vec_shared(4, shared_allocator<TransformMatrix>(*queue_ptr));

#pragma unroll 4
    for (size_t i = 0; i < 4; ++i) {
#pragma unroll 4
        for (size_t j = 0; j < 4; ++j) {
            trans_vec_shared[i][j] = trans(i, j);
        }
    }

    // Optimize work group size
    const size_t work_group_size = sycl_utils::get_work_group_size(*queue_ptr);
    const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;

    sycl_utils::events events;
    if (traits::pointcloud::has_cov(cloud)) {
        const auto covs = traits::pointcloud::covs_ptr(cloud);
        const auto trans_vec_ptr = trans_vec_shared.data();

        /* Transform Covariance */
        events += queue_ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;
                kernel::transform_covs(covs[i], covs[i], trans_vec_ptr);
            });
        });
    }

    {
        const auto point_ptr = traits::pointcloud::points_ptr(cloud);
        const auto trans_vec_ptr = trans_vec_shared.data();

        /* Transform Points*/
        events += queue_ptr->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= N) return;
                kernel::transform_point(point_ptr[i], point_ptr[i], trans_vec_ptr);
            });
        });
    }

    return events;
}

// transform on device
template <typename PointCloud>
inline void transform_sycl(PointCloud& cloud, const TransformMatrix& trans) {
    transform_sycl_async(cloud, trans).wait();
}

template <typename PointCloud>
PointCloud transform_sycl_copy(PointCloud& cloud, const TransformMatrix& trans) {
    const auto queue_ptr = traits::pointcloud::queue_ptr(cloud);

    std::shared_ptr<PointCloud> ret = traits::pointcloud::constructor<PointCloud>(queue_ptr);
    traits::pointcloud::resize_points(*ret, traits::pointcloud::size(cloud));

    sycl_utils::events events;
    if (traits::pointcloud::has_cov(cloud)) {
        traits::pointcloud::resize_covs(*ret, traits::pointcloud::size(cloud));
        events += queue_ptr->submit([&](sycl::handler& h) {
            const auto covs = traits::pointcloud::covs_ptr(cloud);
            const auto output_covs = traits::pointcloud::covs_ptr(*ret);
            h.memcpy(output_covs, covs, traits::pointcloud::size(cloud) * sizeof(Covariance));
        });
    }
    events += queue_ptr->memcpy(traits::pointcloud::points_ptr(*ret), traits::pointcloud::points_ptr(cloud),
                                traits::pointcloud::size(cloud) * sizeof(PointType));
    events.wait();

    transform_sycl(*ret, trans);
    return *ret;
}

inline void transform_cpu(sycl_points::PointCloudShared& cloud, const TransformMatrix& trans) {
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

sycl_points::PointCloudShared transform_cpu_copy(sycl_points::PointCloudShared& cloud, const TransformMatrix& trans) {
    sycl_points::PointCloudShared ret(cloud);  // copy
    transform_cpu(ret, trans);
    return ret;
}

}  // namespace algorithms

}  // namespace sycl_points
