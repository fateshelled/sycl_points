#pragma once

#include <sycl_points/algorithms/common/filter_by_flags.hpp>
#include <sycl_points/algorithms/knn/kdtree.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace filter {

/// @brief Class for removing outliers from a point cloud.
class OutlierRemoval {
public:
    using Ptr = std::shared_ptr<OutlierRemoval>;

    /// @brief Constructor.
    /// @param queue SYCL queue to be used for computations.
    OutlierRemoval(const sycl_utils::DeviceQueue& queue) : queue_(queue) {
        this->filter_ = std::make_shared<FilterByFlags>(this->queue_);
        this->flags_ = std::make_shared<shared_vector<uint8_t>>(*this->queue_.ptr);
        this->indices_ = std::make_shared<shared_vector<int32_t>>(*this->queue_.ptr);
        this->local_mean_distance_ = std::make_shared<shared_vector<float>>(*this->queue_.ptr);
        this->distance_threshold_ = std::make_shared<shared_vector<float>>(*this->queue_.ptr);

        this->neighbors_ = std::make_shared<knn_search::KNNResult>();
    }

    /// @brief Filters outliers from a point cloud based on statistical analysis of the neighborhood of each point.
    /// @details For each point, it computes the mean distance to its k-nearest neighbors. Points whose mean distance is
    /// outside a threshold (mean + std_dev_multiplier * std_dev) are considered outliers and removed.
    /// @param cloud The point cloud to filter (modified in-place).
    /// @param tree The KD-Tree for nearest neighbor search.
    /// @param mean_k The number of nearest neighbors to use for mean distance calculation.
    /// @param stddev_mul_thresh The standard deviation multiplier threshold. A point is an outlier if its mean distance
    /// is > (global_mean + stddev_mul_thresh * global_stddev).
    /// @param remove_from_tree If true, removes the outlier nodes from the KD-Tree as well.
    void statistical(PointCloudShared& cloud, knn_search::KDTree& tree, size_t mean_k, float stddev_mul_thresh,
                     bool remove_from_tree = false) {
        const size_t N = cloud.size();
        if (N < mean_k) {
            std::cerr << "Not enough points in the cloud [ points = " << N << ", mean_k = " << mean_k << " ]"
                      << std::endl;
            return;
        }
        auto knn_event = tree.knn_search_async(cloud, mean_k, *this->neighbors_);

        this->flags_->resize(cloud.size());
        this->local_mean_distance_->resize(cloud.size());

        shared_vector<float> global_distance_sum_vec(1, 0.0f, *this->queue_.ptr);

        // compute knn mean distance
        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(N);

                const size_t k = mean_k;

                // get pointers
                const auto neighbors_index_ptr = this->neighbors_->indices->data();
                const auto neighbors_distances_ptr = this->neighbors_->distances->data();
                const auto local_mean_distance_ptr = this->local_mean_distance_->data();

                auto global_distance_sum_reduction =
                    sycl::reduction(global_distance_sum_vec.data(), sycl::plus<float>());

                // wait for knn search
                h.depends_on(knn_event.evs);

                h.parallel_for(                                       //
                    sycl::nd_range<1>(global_size, work_group_size),  //
                    global_distance_sum_reduction,                    //
                    [=](sycl::nd_item<1> item, auto& global_sum) {
                        const size_t index = item.get_global_id(0);
                        if (index >= N) return;
                        float sum_dist = 0.0f;
                        for (size_t i = 0; i < k; ++i) {
                            const auto nb_idx = neighbors_index_ptr[index * k + i];
                            const auto dist = neighbors_distances_ptr[index * k + i];
                            sum_dist += dist;
                        }
                        const float mean_distance = sum_dist / k;
                        local_mean_distance_ptr[index] = mean_distance;
                        global_sum += mean_distance;
                    });
            })
            .wait();

        // compute variance
        shared_vector<float> global_variance_vec(1, 0.0f, *this->queue_.ptr);
        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(N);

                auto global_var_reduction = sycl::reduction(global_variance_vec.data(), sycl::plus<float>());

                // get pointers
                const auto local_mean_distance_ptr = this->local_mean_distance_->data();
                const float global_mean_distance = global_distance_sum_vec[0] / static_cast<float>(N);

                h.parallel_for(  //
                    sycl::nd_range<1>(global_size, work_group_size),
                    global_var_reduction,  //
                    [=](sycl::nd_item<1> item, auto& global_var) {
                        const size_t index = item.get_global_id(0);
                        if (index >= N) return;
                        const auto sub = global_mean_distance - local_mean_distance_ptr[index];
                        global_var += sub * sub;
                    });
            })
            .wait();

        // compute flags
        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(N);

                // get pointers
                const auto local_mean_distance_ptr = this->local_mean_distance_->data();
                const auto flag_ptr = this->flags_->data();

                const float global_mean_distance = global_distance_sum_vec[0] / static_cast<float>(N);
                const float global_stddev = std::sqrt(global_variance_vec[0] / static_cast<float>(N));
                const float threshold = global_mean_distance + stddev_mul_thresh * global_stddev;

                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const size_t index = item.get_global_id(0);
                    if (index >= N) return;
                    const float mean_dist = local_mean_distance_ptr[index];
                    flag_ptr[index] = (mean_dist > threshold) ? REMOVE_FLAG : INCLUDE_FLAG;
                });
            })
            .wait();

        // filter
        this->filter_by_flags(cloud);

        if (remove_from_tree) {
            tree.remove_nodes_by_flags(this->get_flags(), this->calculate_indices());
        }
    }

    /// @brief Filters outliers from a point cloud based on the number of neighbors within a given radius.
    /// @details For each point, it counts the number of neighbors within a specified radius. If the number of neighbors
    /// is less than a given threshold, the point is considered an outlier and removed.
    /// @param cloud The point cloud to filter (modified in-place).
    /// @param tree The KD-Tree for nearest neighbor search.
    /// @param min_k The minimum number of neighbors a point must have within the radius to be considered an inlier.
    /// @param radius The radius to search for neighbors.
    /// @param remove_from_tree If true, removes the outlier nodes from the KD-Tree as well.
    void radius(PointCloudShared& cloud, knn_search::KDTree& tree, size_t min_k, float radius,
                bool remove_from_tree = false) {
        const size_t N = cloud.size();
        if (N < min_k) {
            std::cerr << "Not enough points in the cloud [ points = " << N << ", min_k = " << min_k << " ]"
                      << std::endl;
            return;
        }
        // tree include the point itself, so search min_k + 1
        auto knn_event = tree.knn_search_async(cloud, min_k + 1, *this->neighbors_);

        this->flags_->resize(cloud.size());

        // compute flags
        this->queue_.ptr
            ->submit([&](sycl::handler& h) {
                const size_t work_group_size = this->queue_.get_work_group_size();
                const size_t global_size = this->queue_.get_global_size(N);

                // get pointers
                const auto neighbors_index_ptr = this->neighbors_->indices->data();
                const auto neighbors_distances_ptr = this->neighbors_->distances->data();
                const auto flag_ptr = this->flags_->data();

                const auto threshold = radius;

                // wait for knn search
                h.depends_on(knn_event.evs);

                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const size_t index = item.get_global_id(0);
                    if (index >= N) return;

                    const auto max_dist = neighbors_distances_ptr[index * (min_k + 1) + min_k];
                    flag_ptr[index] = (max_dist > threshold) ? REMOVE_FLAG : INCLUDE_FLAG;
                });
            })
            .wait();

        // filter
        this->filter_by_flags(cloud);

        if (remove_from_tree) {
            tree.remove_nodes_by_flags(this->get_flags(), this->calculate_indices());
        }
    }

    /// @brief Gets the flags indicating which points were kept or removed.
    /// @return A constant reference to the flags vector. `INCLUDE_FLAG` for inliers, `REMOVE_FLAG` for outliers.
    const shared_vector<uint8_t>& get_flags() const { return *this->flags_; }

    /// @brief Calculates the new indices for the inlier points.
    /// @details This is useful for remapping other data structures that correspond to the point cloud.
    /// @return A constant reference to the indices vector. Inlier points get a new compact index, outliers get -1.
    const shared_vector<int32_t>& calculate_indices() const {
        this->filter_->calculate_indices(*this->flags_, *this->indices_);
        return *this->indices_;
    }

private:
    sycl_utils::DeviceQueue queue_;
    knn_search::KNNResult::Ptr neighbors_;
    FilterByFlags::Ptr filter_;
    shared_vector_ptr<uint8_t> flags_;
    shared_vector_ptr<int32_t> indices_;
    shared_vector_ptr<float> local_mean_distance_;
    shared_vector_ptr<float> distance_threshold_;

    /// @brief Applies filtering based on the current flags
    /// @param data Point cloud to be filtered (modified in-place)
    void filter_by_flags(PointCloudShared& data) {
        if (data.has_cov()) {
            this->filter_->filter_by_flags(*data.covs, *this->flags_);
        }
        if (data.has_normal()) {
            this->filter_->filter_by_flags(*data.normals, *this->flags_);
        }
        if (data.has_rgb()) {
            this->filter_->filter_by_flags(*data.rgb, *this->flags_);
        }
        if (data.has_intensity()) {
            this->filter_->filter_by_flags(*data.intensities, *this->flags_);
        }
        this->filter_->filter_by_flags(*data.points, *this->flags_);
    }
};

}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
