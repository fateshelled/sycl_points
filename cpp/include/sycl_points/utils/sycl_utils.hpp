#pragma once

#include <Eigen/Dense>  // Must include before <sycl/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sycl/sycl.hpp>

#include "device_utils.hpp"

#ifdef SYCL_IMPL_ADAPTIVECPP
#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif
#endif

namespace sycl_points {

namespace sycl_utils {

/// @brief get device optimized work_group_size
/// @param device SYCL device
/// @param work_group_size if greater than 0, return this value clamped at upper limit.
/// @return optimized work_group_size
inline size_t get_work_group_size(const sycl::device& device, size_t work_group_size = 0UL) {
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    if (work_group_size > 0UL) {
        return std::min(work_group_size, max_work_group_size);
    }

    const size_t max_compute_unit = static_cast<size_t>(device.get_info<sycl::info::device::max_compute_units>());
    if (device.is_cpu()) {
        // CPU's max_compute_units is number of total thread.
        return std::min(max_compute_unit, max_work_group_size);
    }
    return std::min(max_compute_unit * 4UL, max_work_group_size);

    // const auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
    // switch (vendor_id) {
    //     case VENDOR_ID::INTEL:
    //         // optimize for iGPU
    //         return std::min(max_compute_unit * 4, max_work_group_size);
    //         break;
    //     case VENDOR_ID::NVIDIA:
    //         return std::min(max_compute_unit * 4, max_work_group_size);
    //         break;
    // }
}

/// @brief get device optimized work_group_size
/// @param queue SYCL queue
/// @param work_group_size if greater than 0, return this value clamped at upper limit.
/// @return optimized work_group_size
inline size_t get_work_group_size(const sycl::queue& queue, size_t work_group_size = 0UL) {
    return get_work_group_size(queue.get_device(), work_group_size);
}

inline size_t get_work_group_size_for_parallel_reduction(const sycl::device& device, size_t work_group_size = 0UL) {
    const size_t max_work_group_size = 128UL;  // conservative value
    if (work_group_size > 0) {
        return std::min(work_group_size, max_work_group_size);
    }
    const size_t max_compute_unit = static_cast<size_t>(device.get_info<sycl::info::device::max_compute_units>());
    if (device.is_cpu()) {
        return std::min(max_compute_unit, max_work_group_size);
    }
    return std::min(max_compute_unit * 4UL, max_work_group_size);
}

inline size_t get_work_group_size_for_parallel_reduction(const sycl::queue& queue, size_t work_group_size = 0UL) {
    return get_work_group_size_for_parallel_reduction(queue.get_device(), work_group_size);
}

/// @brief Get the maximum supported sub-group size of a device.
/// @param device SYCL device
/// @return Maximum supported sub-group size, or 1 when the device does not report any.
inline size_t get_max_sub_group_size(const sycl::device& device) {
    try {
        const auto sub_group_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
        if (!sub_group_sizes.empty()) {
            return *std::max_element(sub_group_sizes.begin(), sub_group_sizes.end());
        }
    } catch (...) {
        // sub_group_sizes is not supported on every backend; fall back to 1.
    }
    return 1UL;
}

/// @brief Calculate global_size for a kernel execution based on total number of elements and work_group_size.
/// @param N total number of elements to process.
/// @param work_group_size size of each work group.
/// @return global_size
inline size_t get_global_size(size_t N, size_t work_group_size) {
    return ((N + work_group_size - 1) / work_group_size) * work_group_size;
}

/// @brief Compute a work-group size for kernels that allocate a local (SLM) reduction buffer.
/// @param device SYCL device.
/// @param slm_entry_size Size in bytes of the per work-item local memory entry.
/// @return Work-group size that fits in the device's shared local memory and is aligned
///         to a multiple of the maximum supported sub-group size.
/// @note The work-group size is first derived from a device heuristic (vendor and compute unit
///       count) and then capped so that `wg_size * slm_entry_size` does not exceed the device's
///       local memory. A work-group requesting more SLM than available is rejected by the JIT
///       compiler (e.g. Level Zero returns ZE_RESULT_ERROR_INVALID_ARGUMENT at zeModuleCreate).
inline size_t compute_work_group_size_for_slm(const sycl::device& device, const size_t slm_entry_size) {
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t compute_units = static_cast<size_t>(device.get_info<sycl::info::device::max_compute_units>());
    const bool cpu = is_cpu(device);

    size_t wg_size = std::min<size_t>(128, max_work_group_size);
    if (is_nvidia(device)) {
        wg_size = std::min(max_work_group_size, size_t{64});
    } else if (is_intel(device) && !cpu) {
        wg_size = std::min(max_work_group_size, compute_units * size_t{8});
    } else if (cpu) {
        wg_size = std::min(max_work_group_size, compute_units * size_t{100});
    }

    // Cap the work-group size so the local reduction buffer fits in SLM.
    // The kernel consumes additional SLM beyond the accessor (e.g. for barriers and
    // sub-group shuffles). Measured overhead on Intel GPUs is ~1 KB, so reserve a
    // 2 KB safety margin below the device limit.
    const size_t local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    const size_t slm_margin = 2UL * 1024UL;
    if (local_mem_size > slm_margin) {
        const size_t max_by_slm = std::max<size_t>(1UL, (local_mem_size - slm_margin) / slm_entry_size);
        wg_size = std::min(wg_size, max_by_slm);
    }

    // Align the work-group size down to a multiple of the sub-group size.
    const size_t sub_group_size = std::max<size_t>(1UL, get_max_sub_group_size(device));
    wg_size -= wg_size % sub_group_size;
    return std::max<size_t>(1UL, wg_size);
}

/// @brief free with nullptr check
/// @param data_ptr pointer of data
/// @param queue SYCL queue
inline void free(void* data_ptr, const sycl::queue& queue) {
    if (data_ptr != nullptr) {
        sycl::free(data_ptr, queue);
    }
}

/// @brief sycl::event container
struct events {
    /// @brief events
    std::vector<sycl::event> evs;
    /// @brief resources kept alive until events complete
    std::vector<std::shared_ptr<void>> keep_alive;

    /// @brief add event
    /// @param event event
    void push_back(const sycl::event& event) { this->evs.push_back(event); }

    /// @brief wait all events
    void wait() {
        for (auto& event : this->evs) {
            event.wait();
        }
        this->clear();
    }

    /// @brief wait_and_throw all events
    void wait_and_throw() {
        for (auto& event : this->evs) {
            event.wait_and_throw();
        }
        this->clear();
    }

    /// @brief clear all events
    void clear() {
        this->evs.clear();
        this->keep_alive.clear();
    }
    /// @brief add event
    /// @param event event
    void operator+=(const sycl::event& event) { this->evs.push_back(event); }
    /// @brief add events
    /// @param e events
    void operator+=(const events& e) {
        std::copy(e.evs.begin(), e.evs.end(), std::back_inserter(this->evs));
        std::copy(e.keep_alive.begin(), e.keep_alive.end(), std::back_inserter(this->keep_alive));
    }

    /// @brief keep a resource alive while events are pending
    template <typename T>
    void add_resource(const std::shared_ptr<T>& resource) {
        this->keep_alive.emplace_back(resource);
    }
};

/// @brief shared memory location advise to underlying runtime
namespace mem_advise {

#ifdef SYCL_IMPL_INTEL_DPCPP

/// @brief Hints that data will be accessed from the device. set flag UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE.
/// @tparam T data type
/// @param queue SYCL queue
/// @param data_ptr shared memory pointer of data
/// @param N number of data
template <typename T>
void set_accessed_by_device(sycl::queue& queue, T* data_ptr, size_t N) {
    queue.mem_advise(data_ptr, sizeof(T) * N, ur_usm_advice_flag_t::UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE);
}

/// @brief Remove affects of `set_accessed_by_device`. set flag UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE
/// @tparam T data type
/// @param queue SYCL queue
/// @param data_ptr shared memory pointer of data
/// @param N number of data
template <typename T>
void clear_accessed_by_device(sycl::queue& queue, T* data_ptr, size_t N) {
    queue.mem_advise(data_ptr, sizeof(T) * N, ur_usm_advice_flag_t::UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE);
}

/// @brief Hints that data will be accessed from the host. set flag UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST
/// @tparam T data type
/// @param queue SYCL queue
/// @param data_ptr shared memory pointer of data
/// @param N number of data
template <typename T>
void set_accessed_by_host(sycl::queue& queue, T* data_ptr, size_t N) {
    queue.mem_advise(data_ptr, sizeof(T) * N, ur_usm_advice_flag_t::UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST);
}

/// @brief Remove affects of `set_accessed_by_host`. set flag UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_HOST
/// @tparam T data type
/// @param queue SYCL queue
/// @param data_ptr shared memory pointer of data
/// @param N number of data
template <typename T>
void clear_accessed_by_host(sycl::queue& queue, T* data_ptr, size_t N) {
    queue.mem_advise(data_ptr, sizeof(T) * N, ur_usm_advice_flag_t::UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_HOST);
}

/// @brief Hint that memory will be read from frequently and written to rarely
/// @tparam T data type
/// @param queue SYCL queue
/// @param data_ptr shared memory pointer of data
/// @param N number of data
template <typename T>
void set_read_mostly(sycl::queue& queue, T* data_ptr, size_t N) {
    queue.mem_advise(data_ptr, sizeof(T) * N, ur_usm_advice_flag_t::UR_USM_ADVICE_FLAG_SET_READ_MOSTLY);
}

/// @brief Remove affects of `set_read_mostly`. set flag UR_USM_ADVICE_FLAG_CLEAR_READ_MOSTLY
/// @tparam T data type
/// @param queue SYCL queue
/// @param data_ptr shared memory pointer of data
/// @param N number of data
template <typename T>
void clear_read_mostly(sycl::queue& queue, T* data_ptr, size_t N) {
    queue.mem_advise(data_ptr, sizeof(T) * N, ur_usm_advice_flag_t::UR_USM_ADVICE_FLAG_CLEAR_READ_MOSTLY);
}

#else  // AdaptiveCpp: mem_advise hints are not available, use no-ops

template <typename T>
void set_accessed_by_device(sycl::queue&, T*, size_t) {}
template <typename T>
void clear_accessed_by_device(sycl::queue&, T*, size_t) {}
template <typename T>
void set_accessed_by_host(sycl::queue&, T*, size_t) {}
template <typename T>
void clear_accessed_by_host(sycl::queue&, T*, size_t) {}
template <typename T>
void set_read_mostly(sycl::queue&, T*, size_t) {}
template <typename T>
void clear_read_mostly(sycl::queue&, T*, size_t) {}

#endif  // SYCL_IMPL_INTEL_DPCPP

}  // namespace mem_advise

/// @brief Submit a host-side callable with event dependencies.
/// @details For Intel DPC++: uses handler::host_task for proper SYCL event integration.
///          For AdaptiveCpp: waits for dependencies synchronously and executes on CPU.
/// @param q SYCL queue to submit to
/// @param deps Events to wait for before executing the callable
/// @param f Callable to execute on the host
/// @return SYCL event representing task completion
template <typename Func>
inline sycl::event submit_host_task(sycl::queue& q, const std::vector<sycl::event>& deps, Func&& f) {
#ifdef SYCL_IMPL_ADAPTIVECPP
    sycl::event::wait_and_throw(deps);
    f();
    return sycl::event{};
#else
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.host_task(std::forward<Func>(f));
    });
#endif
}

/// @brief Represents a SYCL queue with device-specific optimizations and management capabilities
class DeviceQueue {
private:
    size_t work_group_size;
    size_t work_group_size_for_parallel_reduction;
    size_t max_sub_group_size;

public:
    using Ptr = std::shared_ptr<DeviceQueue>;

    std::shared_ptr<sycl::queue> ptr = nullptr;  // sycl::queue pointer

    /// @brief constructor
    /// @param device sycl::device class
    DeviceQueue(const sycl::device& device)
        : ptr(std::make_shared<sycl::queue>(device, [](sycl::exception_list el) {
              for (auto& e : el) {
                  try {
                      std::rethrow_exception(e);
                  } catch (const std::exception& ex) {
                      std::cerr << "[SYCL async exception] " << ex.what() << std::endl;
                  } catch (...) {
                      std::cerr << "[SYCL async exception] unknown exception" << std::endl;
                  }
              }
          })) {
        if (!sycl_utils::device_selector::is_supported_device(device)) {
            const std::string device_name = device.get_info<sycl::info::device::name>();
            const std::string error_msg =
                device_name + " [" + get_backend_name(device.get_backend()) + "]" + " is not supported.";
            throw std::runtime_error("[DeviceQueue::DeviceQueue] " + error_msg);
        }
        this->work_group_size = sycl_utils::get_work_group_size(device);
        this->work_group_size_for_parallel_reduction = sycl_utils::get_work_group_size_for_parallel_reduction(device);
        this->max_sub_group_size = sycl_utils::get_max_sub_group_size(device);
    }

    /// @brief Print device info
    void print_device_info() const { sycl_utils::print_device_info(*this->ptr); }
    /// @brief SYCL device this queue was constructed with.
    sycl::device get_device() const { return this->ptr->get_device(); }
    /// @brief device is CPU or not
    bool is_cpu() const { return sycl_utils::is_cpu(*this->ptr); }
    /// @brief device is GPU or not
    bool is_gpu() const { return sycl_utils::is_gpu(*this->ptr); }
    /// @brief device vendor is Intel or not
    bool is_intel() const { return sycl_utils::is_intel(*this->ptr); }
    /// @brief device vendor is NVIDIA or not
    bool is_nvidia() const { return sycl_utils::is_nvidia(*this->ptr); }
    /// @brief device vendor is AMD or not
    bool is_amd() const { return sycl_utils::is_amd(*this->ptr); }
    /// @brief device support double precision or not
    bool is_supported_double() const { return this->ptr->get_device().has(sycl::aspect::fp64); }

    /// @brief get work group size
    /// @return work group size
    size_t get_work_group_size() const { return this->work_group_size; }

    /// @brief set work group size
    /// @param wg_size work group size
    void set_work_group_size(size_t wg_size) {
        this->work_group_size = sycl_utils::get_work_group_size(*this->ptr, wg_size);
    }

    /// @brief get work group size for parallel reduction
    /// @return work group size for parallel reduction
    size_t get_work_group_size_for_parallel_reduction() const { return this->work_group_size_for_parallel_reduction; }

    /// @brief Get the maximum supported sub-group size of the device.
    /// @return maximum supported sub-group size
    size_t get_max_sub_group_size() const { return this->max_sub_group_size; }

    /// @brief set work group size for parallel reduction
    /// @param wg_size work group size
    void set_work_group_size_for_parallel_reduction(size_t wg_size) {
        this->work_group_size_for_parallel_reduction = sycl_utils::get_work_group_size(*this->ptr, wg_size);
    }

    /// @brief Calculate global_size for a kernel execution based on total number of elements and work_group_size.
    /// @param N total number of elements to process.
    /// @return global size
    size_t get_global_size(size_t N) const { return sycl_utils::get_global_size(N, this->work_group_size); }

    /// @brief Calculate global_size for parallel reduction.
    /// @param N total number of elements to process.
    /// @return global size for parallel reduction
    size_t get_global_size_for_parallel_reduction(size_t N) const {
        return sycl_utils::get_global_size(N, this->work_group_size_for_parallel_reduction);
    }

    /// @brief Hints that data will be accessed from the device. set flag UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE.
    /// @tparam T data type
    /// @param data_ptr shared memory pointer of data
    /// @param N number of data
    template <typename T>
    void set_accessed_by_device(T* data_ptr, size_t N) const {
        sycl_utils::mem_advise::set_accessed_by_device<T>(*this->ptr, data_ptr, N);
    }

    /// @brief Remove affects of `set_accessed_by_device`. set flag UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE
    /// @tparam T data type
    /// @param data_ptr shared memory pointer of data
    /// @param N number of data
    template <typename T>
    void clear_accessed_by_device(T* data_ptr, size_t N) const {
        sycl_utils::mem_advise::clear_accessed_by_device<T>(*this->ptr, data_ptr, N);
    }

    /// @brief Hints that data will be accessed from the host. set flag UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST
    /// @tparam T data type
    /// @param data_ptr shared memory pointer of data
    /// @param N number of data
    template <typename T>
    void set_accessed_by_host(T* data_ptr, size_t N) const {
        sycl_utils::mem_advise::set_accessed_by_host<T>(*this->ptr, data_ptr, N);
    }

    /// @brief Remove affects of `set_accessed_by_host`. set flag UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_HOST
    /// @tparam T data type
    /// @param data_ptr shared memory pointer of data
    /// @param N number of data
    template <typename T>
    void clear_accessed_by_host(T* data_ptr, size_t N) const {
        sycl_utils::mem_advise::clear_accessed_by_host<T>(*this->ptr, data_ptr, N);
    }

    /// @brief Hint that memory will be read from frequently and written to rarely
    /// @tparam T data type
    /// @param queue SYCL queue
    /// @param data_ptr shared memory pointer of data
    /// @param N number of data
    template <typename T>
    void set_read_mostly(T* data_ptr, size_t N) const {
        sycl_utils::mem_advise::set_read_mostly<T>(*this->ptr, data_ptr, N);
    }

    /// @brief Remove affects of `set_read_mostly`. set flag UR_USM_ADVICE_FLAG_CLEAR_READ_MOSTLY
    /// @tparam T data type
    /// @param data_ptr shared memory pointer of data
    /// @param N number of data
    template <typename T>
    void clear_read_mostly(T* data_ptr, size_t N) const {
        sycl_utils::mem_advise::clear_read_mostly<T>(*this->ptr, data_ptr, N);
    }
};

}  // namespace sycl_utils

template <typename T, size_t Alignment = 0>
using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared, Alignment>;
template <typename T>
using shared_vector = std::vector<T, shared_allocator<T, alignof(T)>>;
template <typename T>
using shared_vector_ptr = std::shared_ptr<shared_vector<T>>;

}  // namespace sycl_points
