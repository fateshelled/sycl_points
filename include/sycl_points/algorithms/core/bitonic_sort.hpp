#pragma once

#include <cmath>
#include <sycl_points/utils/sycl_utils.hpp>
#include <vector>

namespace sycl_points {

namespace algorithms {

namespace core {

template <typename T>
class BitonicSortDevice {
public:
    BitonicSortDevice(const std::shared_ptr<sycl::queue> &queue_ptr) : queue_ptr_(queue_ptr) {
        this->padded_data_ = std::make_shared<ContainerDevice<T>>(this->queue_ptr_);
        this->invalid_value_ = T();
    }
    void setInvalidValue(T value) { this->invalid_value_ = value; }

    template <typename Allocator = std::allocator<T>>
    void sort(const std::vector<T, Allocator> &input, std::vector<T, Allocator> &result) {
        const auto N = input.size();

        size_t pow2_size = 1;
        while (pow2_size < N) {
            pow2_size *= 2;
        }

        if (this->padded_data_->size < pow2_size) {
            this->padded_data_->resize(pow2_size);
        }
        sycl_utils::events events;
        events += this->padded_data_->memset_async(this->invalid_value_);
        events += this->padded_data_->memcpy_async(input.data(), input.size());
        events.wait();

        // work group size
        const size_t local_size = sycl_utils::get_work_group_size(*this->queue_ptr_);

        // sort
        for (size_t k = 2; k <= pow2_size; k *= 2) {
            for (size_t j = k / 2; j > 0; j /= 2) {
                auto event = this->queue_ptr_->submit([&](sycl::handler &h) {
                    const auto data_ptr = this->padded_data_->data;
                    //  h.depends_on(events.evs);  // too slow
                    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(pow2_size / 2), sycl::range<1>(local_size)),
                                   [=](sycl::nd_item<1> item) {
                                       const auto gid = item.get_global_id(0);

                                       // if (gid >= pow2_size / 2) return;

                                       const size_t i = 2 * gid - (gid & (j - 1));
                                       const size_t ixj = i ^ j;

                                       if (ixj > i && i < pow2_size && ixj < pow2_size) {
                                           const bool direction = ((i & k) == 0);

                                           if ((data_ptr[i] > data_ptr[ixj]) == direction) {
                                               // swap
                                               const T temp = data_ptr[i];
                                               data_ptr[i] = data_ptr[ixj];
                                               data_ptr[ixj] = temp;
                                           }
                                       }
                                   });
                });
                //  event.wait();
                events += event;
            }
        }
        events.wait();

        result.resize(N);
        this->queue_ptr_->memcpy(result.data(), this->padded_data_->data, N * sizeof(T)).wait();
    }

private:
    std::shared_ptr<sycl::queue> queue_ptr_ = nullptr;
    std::shared_ptr<ContainerDevice<T>> padded_data_ = nullptr;
    T invalid_value_;
};

template <typename T>
class BitonicSortShared {
public:
    BitonicSortShared(const std::shared_ptr<sycl::queue> &queue_ptr) : queue_ptr_(queue_ptr) {
        constexpr size_t INITIAL_SIZE = 65536;
        this->padded_data_ = std::make_shared<shared_vector<T>>(INITIAL_SIZE, std::numeric_limits<T>::max(),
                                                                shared_allocator<T>(*this->queue_ptr_, {}));
        this->invalid_value_ = T();
    }
    void setInvalidValue(T value) { this->invalid_value_ = value; }

    template <typename Allocator = std::allocator<T>>
    void sort(const std::vector<T, Allocator> &input, std::vector<T, Allocator> &result) {
        const auto N = input.size();

        size_t pow2_size = 1;
        while (pow2_size < N) {
            pow2_size *= 2;
        }
        if (this->padded_data_->size() < pow2_size) {
            this->padded_data_->resize(pow2_size);
        }
        sycl_utils::events events;
        events += this->queue_ptr_->memset(this->padded_data_->data() + N, this->invalid_value_, pow2_size - N);
        events += this->queue_ptr_->memcpy(this->padded_data_->data(), input.data(), N * sizeof(T));
        events.wait();

        // work group size
        const size_t local_size = sycl_utils::get_work_group_size(*this->queue_ptr_);

        // sort
        for (size_t k = 2; k <= pow2_size; k *= 2) {
            for (size_t j = k / 2; j > 0; j /= 2) {
                auto event = this->queue_ptr_->submit([&](sycl::handler &h) {
                    const auto data_ptr = this->padded_data_->data();
                    //  h.depends_on(events.evs);  // too slow
                    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(pow2_size / 2), sycl::range<1>(local_size)),
                                   [=](sycl::nd_item<1> item) {
                                       const auto gid = item.get_global_id(0);

                                       // if (gid >= pow2_size / 2) return;

                                       const size_t i = 2 * gid - (gid & (j - 1));
                                       const size_t ixj = i ^ j;

                                       if (ixj > i && i < pow2_size && ixj < pow2_size) {
                                           const bool direction = ((i & k) == 0);

                                           if ((data_ptr[i] > data_ptr[ixj]) == direction) {
                                               // swap
                                               const T temp = data_ptr[i];
                                               data_ptr[i] = data_ptr[ixj];
                                               data_ptr[ixj] = temp;
                                           }
                                       }
                                   });
                });
                events += event;
            }
        }
        events.wait();

        result.resize(N);
        this->queue_ptr_->memcpy(result.data(), this->padded_data_->data(), N * sizeof(T)).wait();
    }

private:
    std::shared_ptr<sycl::queue> queue_ptr_ = nullptr;
    std::shared_ptr<shared_vector<T>> padded_data_ = nullptr;
    T invalid_value_;
};

}  // namespace core

}  // namespace algorithms

}  // namespace sycl_points
