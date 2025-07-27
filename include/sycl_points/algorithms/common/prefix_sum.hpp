#pragma once
#include <numeric>
#include <random>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {

namespace algorithms {

namespace common {

class PrefixSum {
public:
    using Ptr = std::shared_ptr<PrefixSum>;

    /// @brief Constructor
    /// @param queue SYCL queue
    PrefixSum(const sycl_utils::DeviceQueue& queue) : queue_(queue) {
        this->prefix_sum_ptr_ = std::make_shared<shared_vector<uint32_t>>(*this->queue_.ptr);
        this->group_sums_ptr_ = std::make_shared<shared_vector<uint32_t>>(*this->queue_.ptr);
        this->group_prefix_ptr_ = std::make_shared<shared_vector<uint32_t>>(*this->queue_.ptr);
    }

    /// @brief get computed prefix sum
    /// @return
    const shared_vector<uint32_t>& get_prefix_sum() const { return *this->prefix_sum_ptr_; }

    /// @brief compute prefix sum
    /// @param flags
    /// @return new_size
    size_t compute(const shared_vector<uint8_t>& flags) {
        const size_t N = flags.size();
        if (N == 0) return 0;

        if (this->prefix_sum_ptr_->size() < N) {
            this->prefix_sum_ptr_->resize(N);
        }

        const size_t wg_size = this->queue_.get_work_group_size();
        const size_t num_groups = (N + wg_size - 1) / wg_size;

        // Resize and initialize group sum buffers
        if (this->group_sums_ptr_->size() < num_groups) {
            this->group_sums_ptr_->resize(num_groups);
            this->group_prefix_ptr_->resize(num_groups);
        }
        // mem_advise to device
        {
            this->queue_.set_accessed_by_device(this->prefix_sum_ptr_->data(), N);
            this->queue_.set_accessed_by_device(this->group_sums_ptr_->data(), num_groups);
            this->queue_.set_accessed_by_device(this->group_prefix_ptr_->data(), num_groups);
        }

        // Initialize buffers to zero
        sycl_utils::events init_event;
        {
            init_event += this->queue_.ptr->fill(this->group_sums_ptr_->data(), (uint32_t)0, num_groups);
            init_event += this->queue_.ptr->fill(this->group_prefix_ptr_->data(), (uint32_t)0, num_groups);
        }
        init_event.wait();

        // Step 1: Inclusive scan within each work group
        {
            auto event1 = this->queue_.ptr->submit([&](sycl::handler& h) {
                const auto flags_ptr = flags.data();
                auto prefix_sum_ptr = this->prefix_sum_ptr_->data();
                auto group_sums_ptr = this->group_sums_ptr_->data();
                const auto N_capture = N;
                const auto wg_size_capture = wg_size;

                h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size), [=](sycl::nd_item<1> item) {
                    const size_t gid = item.get_global_id(0);
                    const size_t lid = item.get_local_id(0);
                    // Calculate group ID
                    const size_t group_id = gid / wg_size_capture;
                    auto group = item.get_group();

                    // Each thread loads a value (0 if out of range)
                    uint32_t value = (gid < N_capture) ? static_cast<uint32_t>(flags_ptr[gid]) : 0;

                    // Inclusive scan within the work group
                    uint32_t scan_result = sycl::inclusive_scan_over_group(group, value, sycl::plus<uint32_t>());

                    // Write results to global memory
                    if (gid < N_capture) {
                        prefix_sum_ptr[gid] = scan_result;
                    }

                    // Only the last element of each work group saves the sum
                    if (lid == wg_size_capture - 1 || gid == N_capture - 1) {
                        group_sums_ptr[group_id] = scan_result;
                    }
                });
            });
            event1.wait();
        }

        // Complete if only one group
        if (num_groups == 1) return get_new_size(N);
        ;

        // Step 2: Perform an exclusive scan on group sums
        {
            // 1. Shift group_sum to create exclusive scan input
            auto event2a = this->queue_.ptr->submit([&](sycl::handler& h) {
                auto group_sums_ptr = this->group_sums_ptr_->data();
                auto group_prefix_ptr = this->group_prefix_ptr_->data();
                const auto num_groups_capture = num_groups;

                h.parallel_for(sycl::range<1>(num_groups_capture), [=](sycl::id<1> idx) {
                    const size_t i = idx[0];
                    if (i == 0) {
                        group_prefix_ptr[i] = 0;  // First element is zero
                    } else {
                        group_prefix_ptr[i] = group_sums_ptr[i - 1];  // Shift
                    }
                });
            });
            event2a.wait_and_throw();

            // 2. Sequential scan for stability
            auto event2b = this->queue_.ptr->submit([&](sycl::handler& h) {
                auto group_prefix_ptr = this->group_prefix_ptr_->data();
                const auto num_groups_capture = num_groups;

                h.single_task([=]() {
                    for (size_t i = 1; i < num_groups_capture; ++i) {
                        group_prefix_ptr[i] += group_prefix_ptr[i - 1];
                    }
                });
            });
            event2b.wait_and_throw();

            // Copy result back to group_sums for use in step 3
            this->queue_.ptr
                ->memcpy(this->group_sums_ptr_->data(), this->group_prefix_ptr_->data(), num_groups * sizeof(uint32_t))
                .wait_and_throw();
        }

        // Step 3: Add the appropriate group offset to each element
        {
            auto event3 = this->queue_.ptr->submit([&](sycl::handler& h) {
                auto prefix_sum_ptr = this->prefix_sum_ptr_->data();
                auto group_sums_ptr = this->group_sums_ptr_->data();
                const auto N_capture = N;
                const auto wg_size_capture = wg_size;

                h.parallel_for(sycl::range<1>(N_capture), [=](sycl::id<1> idx) {
                    const size_t i = idx[0];
                    const size_t group_id = i / wg_size_capture;

                    // Add the appropriate group offset
                    if (group_id > 0) {
                        prefix_sum_ptr[i] += group_sums_ptr[group_id];
                    }
                });
            });
            event3.wait();
        }
        // mem_advise clear
        {
            this->queue_.clear_accessed_by_device(this->prefix_sum_ptr_->data(), N);
            this->queue_.clear_accessed_by_device(this->group_sums_ptr_->data(), num_groups);
            this->queue_.clear_accessed_by_device(this->group_prefix_ptr_->data(), num_groups);
        }

        return get_new_size(N);
    }

private:
    sycl_utils::DeviceQueue queue_;
    shared_vector_ptr<uint32_t> prefix_sum_ptr_;    // Prefix sum results
    shared_vector_ptr<uint32_t> group_sums_ptr_;    // Group sums for prefix calculation
    shared_vector_ptr<uint32_t> group_prefix_ptr_;  // Group prefix sums

    size_t get_new_size(size_t N) const { return this->prefix_sum_ptr_->at(N - 1); }
};

}  // namespace common
}  // namespace algorithms
}  // namespace sycl_points
