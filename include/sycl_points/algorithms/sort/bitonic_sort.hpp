#pragma once

#include <cmath>
#include <sycl_points/utils/sycl_utils.hpp>
#include <vector>

namespace sycl_points {

namespace algorithms {

namespace sort {

template <typename T, size_t Alignment = 0>
class BitonicSortShared {
public:
    BitonicSortShared(const sycl_utils::DeviceQueue &queue) : queue_(queue) {
        constexpr size_t INITIAL_SIZE = 32768;
        this->padded_data_ = std::make_shared<shared_vector<T, Alignment>>(
            INITIAL_SIZE, shared_allocator<T, Alignment>(*this->queue_.ptr, {}));
        this->padded_data_indices_ =
            std::make_shared<shared_vector<uint64_t>>(INITIAL_SIZE, shared_allocator<uint64_t>(*this->queue_.ptr));
    }

    std::vector<uint64_t> get_sorted_indices() {
        std::vector<uint64_t> indices(this->last_N_);
        this->queue_.ptr->memcpy(indices.data(), this->padded_data_indices_->data(), this->last_N_ * sizeof(uint64_t))
            .wait();
        return indices;
    }

    template <typename Allocator = std::allocator<T>>
    void sort(std::vector<T, Allocator> &input) {
        const auto N = input.size();
        this->last_N_ = N;
        if (N == 0) return;

        size_t pow2_size = 1;
        while (pow2_size < N) {
            pow2_size *= 2;
        }
        if (this->padded_data_->size() < pow2_size) {
            this->padded_data_->resize(pow2_size);
            this->padded_data_indices_->resize(pow2_size);
        }

        sycl_utils::events events;

        std::vector<uint64_t> series(pow2_size);
        std::iota(series.begin(), series.end(), 0);
        events +=
            this->queue_.ptr->memcpy(this->padded_data_indices_->data(), series.data(), pow2_size * sizeof(uint64_t));
        if (pow2_size - N > 0) {
            events +=
                this->queue_.ptr->fill(this->padded_data_->data() + N, std::numeric_limits<T>::max(), pow2_size - N);
        }
        events += this->queue_.ptr->memcpy(this->padded_data_->data(), input.data(), N * sizeof(T));
        events.wait_and_throw();

        // work group size
        const size_t max_local_size = this->queue_.get_work_group_size();

        const auto data_ptr = this->padded_data_->data();
        const auto indices_ptr = this->padded_data_indices_->data();
        // sort
        for (size_t k = 2; k <= pow2_size; k *= 2) {
            for (size_t j = k / 2; j > 0; j /= 2) {
                const size_t local_size = std::min(max_local_size, pow2_size / 2);

                auto event = this->queue_.ptr->submit([&](sycl::handler &h) {
                    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(pow2_size / 2), sycl::range<1>(local_size)),
                                   [=](sycl::nd_item<1> item) {
                                       const auto gid = item.get_global_id(0);

                                       if (gid >= pow2_size / 2) return;

                                       const uint64_t i = 2 * gid - (gid & (j - 1));
                                       const uint64_t ixj = i ^ j;

                                       if (ixj > i && i < pow2_size && ixj < pow2_size) {
                                           // necessary swap or not
                                           const auto data_i = data_ptr[i];
                                           const auto data_ixj = data_ptr[ixj];
                                           const bool is_equal = (data_i == data_ixj);
                                           const auto index_i = indices_ptr[i];
                                           const auto index_ixj = indices_ptr[ixj];
                                           const bool idx_compare = (index_i > index_ixj);
                                           const bool val_compare = (data_i > data_ixj);

                                           const bool should_swap = is_equal ? (idx_compare == ((i & k) == 0))
                                                                             : (val_compare == ((i & k) == 0));
                                           if (should_swap) {
                                               data_ptr[i] = data_ixj;
                                               data_ptr[ixj] = data_i;
                                               indices_ptr[i] = index_ixj;
                                               indices_ptr[ixj] = index_i;
                                           }
                                       }
                                   });
                });
                event.wait_and_throw();
            }
        }

        this->queue_.ptr->memcpy(input.data(), this->padded_data_->data(), N * sizeof(T)).wait_and_throw();
    }

private:
    sycl_utils::DeviceQueue queue_;
    shared_vector_ptr<T, Alignment> padded_data_ = nullptr;
    shared_vector_ptr<uint64_t> padded_data_indices_ = nullptr;
    size_t last_N_ = 0;
};

}  // namespace sort

}  // namespace algorithms

}  // namespace sycl_points
