#pragma once

#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace intensity_zscore {

namespace kernel {

/// @brief Compute intensity z-score for one point using its KNN neighborhood.
///        Uses the one-pass variance formula: Var = E[X²] - E[X]²
///        Returns 0 if the local standard deviation is below sigma_min (flat region).
SYCL_EXTERNAL inline float compute_zscore(const float* intensities, const int32_t* index_ptr, size_t k, size_t i,
                                          float sigma_min) {
    float sum_I = 0.0f;
    float sum_I2 = 0.0f;
    for (size_t j = 0; j < k; ++j) {
        const float Ij = intensities[index_ptr[i * k + j]];
        sum_I += Ij;
        sum_I2 += Ij * Ij;
    }
    const float kf = static_cast<float>(k);
    const float mean = sum_I / kf;
    const float var = sycl::fmax(sum_I2 / kf - mean * mean, 0.0f);
    const float sigma = sycl::sqrt(var);
    if (sigma < sigma_min) return 0.0f;
    return (intensities[i] - mean) / sigma;
}

}  // namespace kernel

/// @brief Compute intensity z-score and overwrite intensities in-place.
///        Uses a temporary buffer to avoid read/write race conditions.
///        After this call, cloud.intensities contains z-score values.
///        Call before intensity gradient computation so gradients represent ∇z.
inline void compute(PointCloudShared& cloud, const knn::KNNResult& neighbors, float sigma_min = 0.01f) {
    const size_t N = cloud.size();
    if (N == 0) return;
    if (!cloud.has_intensity()) {
        throw std::runtime_error("[intensity_zscore::compute] Intensity field not found");
    }

    const size_t work_group_size = cloud.queue.get_work_group_size();
    const size_t global_size = cloud.queue.get_global_size(N);
    const auto indices = neighbors.indices;
    const size_t k = neighbors.k;

    // Write z-scores into a temporary buffer to avoid in-place race conditions.
    auto tmp = std::make_shared<shared_vector<float>>(N, 0.0f, *cloud.queue.ptr);

    auto event = cloud.queue.ptr->submit([&, indices, k](sycl::handler& h) {
        const auto intensity_ptr = cloud.intensities_ptr();
        const auto index_ptr = indices->data();
        const auto tmp_ptr = tmp->data();
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
            const size_t i = item.get_global_id(0);
            if (i >= N) return;
            tmp_ptr[i] = kernel::compute_zscore(intensity_ptr, index_ptr, k, i, sigma_min);
        });
    });
    event.wait_and_throw();

    // Swap: O(1) pointer exchange, old intensity buffer released automatically.
    std::swap(cloud.intensities, tmp);
}

}  // namespace intensity_zscore
}  // namespace algorithms
}  // namespace sycl_points
