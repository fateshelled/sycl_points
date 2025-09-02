#pragma once

#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace filter {

constexpr uint8_t REMOVE_FLAG = 0;
constexpr uint8_t INCLUDE_FLAG = 1;

/// @brief Filter class for processing data based on flags
class FilterByFlags {
public:
    using Ptr = std::shared_ptr<FilterByFlags>;

    /// @brief Constructor
    /// @param queue SYCL queue
    FilterByFlags(const sycl_utils::DeviceQueue& queue) : queue_(queue) {}

    /// @brief Filter data synchronously on host
    /// @tparam T Data type (PointType or Covariance)
    /// @tparam AllocSize Optional allocator size
    /// @param data Data to be filtered
    /// @param flags Flags indicating which elements to keep (INCLUDE_FLAG) or remove
    template <typename T, size_t AllocSize = 0>
    void filter_by_flags(shared_vector<T, AllocSize>& data, const shared_vector<uint8_t>& flags) const {
        const size_t N = data.size();
        if (N == 0) return;

        // mem_advise to host
        {
            this->queue_.set_accessed_by_host(data.data(), N);
            this->queue_.set_accessed_by_host(flags.data(), N);
        }

        // Filter data on host (compact elements with INCLUDE_FLAG)
        size_t new_size = 0;
        for (size_t i = 0; i < N; ++i) {
            if (flags[i] == INCLUDE_FLAG) {
                data[new_size++] = data[i];
            }
        }
        // mem_advise clear
        {
            this->queue_.clear_accessed_by_host(data.data(), N);
            this->queue_.clear_accessed_by_host(flags.data(), N);
        }
        data.resize(new_size);
    }

    /// @brief Calculates new indices based on flags.
    /// @param flags Flags indicating which elements to keep (INCLUDE_FLAG) or remove.
    /// @param indices Output vector to store the new indices for elements to be kept, or -1 for elements to be removed.
    void calculate_indices(const shared_vector<uint8_t>& flags, shared_vector<int32_t>& indices) const {
        const size_t N = flags.size();
        if (N == 0) return;

        indices.resize(N);

        // mem_advise to host
        {
            this->queue_.set_accessed_by_host(flags.data(), N);
            this->queue_.set_accessed_by_host(indices.data(), N);
        }

        // Calculate indices on host
        int32_t count = 0;
        for (size_t i = 0; i < N; ++i) {
            indices[i] = (flags[i] == INCLUDE_FLAG) ? count++ : -1;
        }

        // mem_advise clear
        {
            this->queue_.clear_accessed_by_host(flags.data(), N);
            this->queue_.clear_accessed_by_host(indices.data(), N);
        }
    }

private:
    sycl_utils::DeviceQueue queue_;  // SYCL queue
};

}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
