#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace sycl_points {

namespace sycl_utils {

namespace VENDOR_ID {
constexpr uint32_t INTEL = 0x8086;   // 32902
constexpr uint32_t NVIDIA = 0x10de;  // 4318
constexpr uint32_t AMD = 0x1002;     // 4098
};  // namespace VENDOR_ID

/// @brief get device optimized work_group_size
/// @param device SYCL device
/// @param work_group_size if greater than 0, return this value clamped at upper limit.
/// @return optimized work_group_size
inline size_t get_work_group_size(const sycl::device& device, size_t work_group_size = 0) {
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    if (work_group_size > 0) {
        return std::min(work_group_size, max_work_group_size);
    }

    const size_t max_compute_unit = static_cast<size_t>(device.get_info<sycl::info::device::max_compute_units>());
    if (device.is_cpu()) {
        // CPU's max_compute_units is number of total thread.
        return std::min(max_compute_unit, max_work_group_size);
    }

    const auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
    switch (vendor_id) {
        case VENDOR_ID::INTEL:
            // optimize for iGPU
            return std::min(max_compute_unit * 2, max_work_group_size);
            break;
        case VENDOR_ID::NVIDIA:
            return std::min(max_compute_unit * 4, max_work_group_size);
            break;
    }
    return std::min(max_compute_unit * 3, max_work_group_size);
}

/// @brief get device optimized work_group_size
/// @param queue SYCL queue
/// @param work_group_size if greater than 0, return this value clamped at upper limit.
/// @return optimized work_group_size
inline size_t get_work_group_size(const sycl::queue& queue, size_t work_group_size = 0) {
    return get_work_group_size(queue.get_device(), work_group_size);
}

/// @brief get device optimized work_group_size for parallel reduction
/// @param device SYCL device
/// @return optimized work_group_size
inline size_t get_work_group_size_for_parallel_reduction(const sycl::device& device) {
    constexpr size_t max_work_group_size = 281;
    const size_t work_group_size = get_work_group_size(device);
    return std::min(work_group_size, max_work_group_size);
}

/// @brief get device optimized work_group_size for parallel reduction
/// @param queue SYCL queue
/// @return optimized work_group_size
inline size_t get_work_group_size_for_parallel_reduction(const sycl::queue& queue) {
    return get_work_group_size_for_parallel_reduction(queue.get_device());
}

/// @brief Calculate global_size for a kernel execution based on total number of elements and work_group_size.
/// @param N total number of elements to process.
/// @param work_group_size size of each work group.
/// @return global_size
inline size_t get_global_size(size_t N, size_t work_group_size) {
    return ((N + work_group_size - 1) / work_group_size) * work_group_size;
}

/// @brief free with nullptr check
/// @param data_ptr pointer of data
/// @param queue SYCL queue
inline void free(void* data_ptr, const sycl::queue& queue) {
    if (data_ptr != nullptr) {
        sycl::free(data_ptr, queue);
    }
}

/// @brief Print device info
/// @param device SYCL device
inline void print_device_info(const sycl::device& device) {
    const auto platform = device.get_platform();
    std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;

    for (auto device : platform.get_devices()) {
        std::cout << "\tDevice: " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "\ttype: " << (device.is_cpu() ? "CPU" : "GPU") << std::endl;
        std::cout << "\tVendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
        std::cout << "\tVendorID: " << device.get_info<sycl::info::device::vendor_id>() << std::endl;
        std::cout << "\tBackend name: " << device.get_backend() << std::endl;
        std::cout << "\tBackend version: " << device.get_info<sycl::info::device::backend_version>() << std::endl;
        std::cout << "\tDriver version: " << device.get_info<sycl::info::device::driver_version>() << std::endl;
        std::cout << "\tGlobal Memory Size: "
                  << device.get_info<sycl::info::device::global_mem_size>() / 1024.0 / 1024.0 / 1024.0 << " GB"
                  << std::endl;
        std::cout << "\tLocal Memory Size: " << device.get_info<sycl::info::device::local_mem_size>() / 1024.0 << " KB"
                  << std::endl;
        std::cout << "\tMax Memory Allocation Size: "
                  << device.get_info<sycl::info::device::max_mem_alloc_size>() / 1024.0 / 1024.0 / 1024.0 << " GB"
                  << std::endl;

        std::cout << "\tMax Work Group Size: " << device.get_info<sycl::info::device::max_work_group_size>()
                  << std::endl;
        std::cout << "\tMax Work Item Sizes: [";
        std::cout << device.get_info<sycl::info::device::max_work_item_sizes<1>>().dimensions << ", ";
        std::cout << device.get_info<sycl::info::device::max_work_item_sizes<1>>().dimensions << ", ";
        std::cout << device.get_info<sycl::info::device::max_work_item_sizes<1>>().dimensions << "]" << std::endl;
        std::cout << "\tMax Sub Groups num: " << device.get_info<sycl::info::device::max_num_sub_groups>() << std::endl;
        std::cout << "\tSub Group Sizes: [";
        const auto subgroup_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
        for (size_t i = 0; i < subgroup_sizes.size(); ++i) {
            std::cout << subgroup_sizes[i];
            if (i < subgroup_sizes.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;

        std::cout << "\tMax compute units: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;

        std::cout << "\tMax Clock Frequency: " << device.get_info<sycl::info::device::max_clock_frequency>() / 1000.0
                  << " GHz" << std::endl;
        std::cout << "\tDouble precision support: " << (device.has(sycl::aspect::fp64) ? "true" : "false") << std::endl;

        std::cout << "\tUSM host allocations: " << (device.has(sycl::aspect::usm_host_allocations) ? "true" : "false")
                  << std::endl;
        std::cout << "\tUSM device allocations: "
                  << (device.has(sycl::aspect::usm_device_allocations) ? "true" : "false") << std::endl;
        std::cout << "\tUSM shared allocations: "
                  << (device.has(sycl::aspect::usm_shared_allocations) ? "true" : "false") << std::endl;

        std::cout << "\tUSM atomic shared allocations: "
                  << (device.has(sycl::aspect::usm_atomic_shared_allocations) ? "true" : "false") << std::endl;

        std::cout << "\tAvailable: " << (device.get_info<sycl::info::device::is_available>() ? "true" : "false")
                  << std::endl;
        std::cout << std::endl;
    }
}

/// @brief Print selected device info
/// @param queue SYCL queue
inline void print_device_info(const sycl::queue& queue) { print_device_info(queue.get_device()); }

/// @brief device is CPU or not
inline bool is_cpu(const sycl::queue& queue) { return queue.get_device().is_cpu(); }

/// @brief device is iGPU/dGPU or not
inline bool is_gpu(const sycl::queue& queue) { return queue.get_device().is_gpu(); }

/// @brief device is FPGA or not
inline bool is_accelerator(const sycl::queue& queue) { return queue.get_device().is_accelerator(); }

/// @brief device is NVIDIA or not
inline bool is_nvidia(const sycl::queue& queue) {
    const auto device = queue.get_device();
    const auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
    return vendor_id == VENDOR_ID::NVIDIA;
}

/// @brief device is INTEL or not
inline bool is_intel(const sycl::queue& queue) {
    const auto device = queue.get_device();
    const auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
    return vendor_id == VENDOR_ID::INTEL;
}

/// @brief device is AMD or not
inline bool is_amd(const sycl::queue& queue) {
    const auto device = queue.get_device();
    const auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
    return vendor_id == VENDOR_ID::AMD;
}

inline bool enable_shared_allocations(const sycl::device& device) {
    return device.has(sycl::aspect::usm_shared_allocations);
}

/// @brief sycl::event container
struct events {
    /// @brief events
    std::vector<sycl::event> evs;

    /// @brief add event
    /// @param event event
    void push_back(const sycl::event& event) { this->evs.push_back(event); }

    /// @brief wait all events
    void wait() {
        while (evs.size() > 0) {
            auto& event = this->evs.back();
            event.wait();
            evs.pop_back();
        }
    }

    /// @brief clear all events
    void clear() { this->evs.clear(); }
    /// @brief add event
    /// @param event event
    void operator+=(const sycl::event& event) { this->evs.push_back(event); }
    /// @brief add events
    /// @param e events
    void operator+=(const events& e) { std::copy(e.evs.begin(), e.evs.end(), std::back_inserter(this->evs)); }
};

/// @brief shared memory location advise to underlying runtime
namespace mem_advise {

/// @brief Hints that data will be accessed from the device. set flag UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE.
/// @tparam T data type
/// @param queue SYCL queue
/// @param ptr shared memory pointer of data
/// @param N number of data
template <typename T>
void set_accessed_by_device(sycl::queue& queue, T* ptr, size_t N) {
    queue.mem_advise(ptr, sizeof(T) * N, ur_usm_advice_flag_t::UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE);
}

/// @brief Remove affects of `set_accessed_by_device`. set flag UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE
/// @tparam T data type
/// @param queue SYCL queue
/// @param ptr shared memory pointer of data
/// @param N number of data
template <typename T>
void clear_accessed_by_device(sycl::queue& queue, T* ptr, size_t N) {
    queue.mem_advise(ptr, sizeof(T) * N, ur_usm_advice_flag_t::UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE);
}

/// @brief Hints that data will be accessed from the host. set flag UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST
/// @tparam T data type
/// @param queue SYCL queue
/// @param ptr shared memory pointer of data
/// @param N number of data
template <typename T>
void set_accessed_by_host(sycl::queue& queue, T* ptr, size_t N) {
    queue.mem_advise(ptr, sizeof(T) * N, ur_usm_advice_flag_t::UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST);
}

/// @brief Remove affects of `set_accessed_by_host`. set flag UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_HOST
/// @tparam T data type
/// @param queue SYCL queue
/// @param ptr shared memory pointer of data
/// @param N number of data
template <typename T>
void clear_accessed_by_host(sycl::queue& queue, T* ptr, size_t N) {
    queue.mem_advise(ptr, sizeof(T) * N, ur_usm_advice_flag_t::UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_HOST);
}

}  // namespace mem_advise

namespace device_selector {

inline int supported_selector_v(const sycl::device& dev) {
    const auto backend = dev.get_backend();
    bool supported = true;
    supported &= (backend == sycl::backend::opencl) || (backend == sycl::backend::ext_oneapi_cuda);
    supported &= enable_shared_allocations(dev);
    if (supported) {
        return 1;
    } else {
        return -1;
    }
}

inline int intel_cpu_selector_v(const sycl::device& dev) {
    if (supported_selector_v(dev) < 0) {
        return -1;
    }
    const auto vendor_id = dev.get_info<sycl::info::device::vendor_id>();
    return dev.is_cpu() && (vendor_id == VENDOR_ID::INTEL);
}

inline int intel_gpu_selector_v(const sycl::device& dev) {
    if (supported_selector_v(dev) < 0) {
        return -1;
    }
    const auto vendor_id = dev.get_info<sycl::info::device::vendor_id>();
    return dev.is_gpu() && (vendor_id == VENDOR_ID::INTEL);
}

inline int nvidia_gpu_selector_v(const sycl::device& dev) {
    if (supported_selector_v(dev) < 0) {
        return -1;
    }
    const auto vendor_id = dev.get_info<sycl::info::device::vendor_id>();
    return dev.is_gpu() && (vendor_id == VENDOR_ID::NVIDIA);
}

}  // namespace device_selector

}  // namespace sycl_utils

template <typename T, size_t Alignment = 0>
using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared, Alignment>;
template <typename T, size_t Alignment = 0>
using shared_vector = std::vector<T, shared_allocator<T, Alignment>>;

}  // namespace sycl_points
