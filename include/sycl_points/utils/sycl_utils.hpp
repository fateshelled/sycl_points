#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>

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
        std::cout << "\tGlobal Memory Cache Size: "
                  << device.get_info<sycl::info::device::global_mem_cache_size>() / 1024.0 / 1024.0 << " MB"
                  << std::endl;
        std::cout << "\tGlobal Memory Cache Line Size: "
                  << device.get_info<sycl::info::device::global_mem_cache_line_size>() << " byte" << std::endl;

        std::cout << "\tMax Memory Allocation Size: "
                  << device.get_info<sycl::info::device::max_mem_alloc_size>() / 1024.0 / 1024.0 / 1024.0 << " GB"
                  << std::endl;

        std::cout << "\tMax Work Group Size: " << device.get_info<sycl::info::device::max_work_group_size>()
                  << std::endl;
        std::cout << "\tMax Work Item Dimensions: " << device.get_info<sycl::info::device::max_work_item_dimensions>()
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

/// @brief device vendor is NVIDIA or not
inline bool is_nvidia(const sycl::queue& queue) {
    const auto device = queue.get_device();
    const auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
    return vendor_id == VENDOR_ID::NVIDIA;
}

/// @brief device vendor is Intel or not
inline bool is_intel(const sycl::queue& queue) {
    const auto device = queue.get_device();
    const auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
    return vendor_id == VENDOR_ID::INTEL;
}

/// @brief device vendor is AMD or not
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

    /// @brief wait_and_throw all events
    void wait_and_throw() {
        while (evs.size() > 0) {
            auto& event = this->evs.back();
            event.wait_and_throw();
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

}  // namespace mem_advise

namespace device_selector {

bool is_supported_device(const sycl::device& dev) {
    const auto backend = dev.get_backend();
    bool supported = true;
    supported &= (backend == sycl::backend::opencl) || (backend == sycl::backend::ext_oneapi_cuda);
    supported &= enable_shared_allocations(dev);
    return supported;
}

inline int default_selector_v(const sycl::device& dev) { return is_supported_device(dev); }

inline int intel_cpu_selector_v(const sycl::device& dev) {
    const auto vendor_id = dev.get_info<sycl::info::device::vendor_id>();
    return dev.is_cpu() && (vendor_id == VENDOR_ID::INTEL) && is_supported_device(dev);
}

inline int intel_gpu_selector_v(const sycl::device& dev) {
    const auto vendor_id = dev.get_info<sycl::info::device::vendor_id>();
    return dev.is_gpu() && (vendor_id == VENDOR_ID::INTEL) && is_supported_device(dev);
}

inline int nvidia_gpu_selector_v(const sycl::device& dev) {
    const auto vendor_id = dev.get_info<sycl::info::device::vendor_id>();
    return dev.is_gpu() && (vendor_id == VENDOR_ID::NVIDIA) && is_supported_device(dev);
}

sycl::device select_device(const std::string& device_vendor, const std::string& device_type) {
    std::string device_vendor_lower = device_vendor;
    std::transform(device_vendor_lower.begin(), device_vendor_lower.end(), device_vendor_lower.begin(),
                   [](char c) { return std::tolower(c); });
    std::string device_type_lower = device_type;
    std::transform(device_type_lower.begin(), device_type_lower.end(), device_type_lower.begin(),
                   [](char c) { return std::tolower(c); });

    uint32_t vendor_id = 0;
    if (device_vendor_lower == "intel") {
        vendor_id = VENDOR_ID::INTEL;
    } else if (device_vendor_lower == "nvidia") {
        vendor_id = VENDOR_ID::NVIDIA;
    } else if (device_vendor_lower == "amd") {
        vendor_id = VENDOR_ID::AMD;
    } else {
        throw std::runtime_error("invalid device vendor: " + device_vendor);
    }

    const bool select_cpu = device_type_lower == "cpu" ? true : false;
    const bool select_gpu = device_type_lower == "gpu" ? true : false;
    if (!select_cpu && !select_gpu) {
        throw std::runtime_error("not support device type: " + device_type);
    }

    for (auto platform : sycl::platform::get_platforms()) {
        for (auto device : platform.get_devices()) {
            if (vendor_id == device.get_info<sycl::info::device::vendor_id>()) {
                if (device.get_backend() == sycl::backend::ext_oneapi_level_zero) {
                    // level_zero is not support
                    continue;
                }
                if (select_cpu && device.is_cpu()) {
                    return device;
                }
                if (select_gpu && device.is_gpu()) {
                    return device;
                }
            }
        }
    }
    throw std::runtime_error("not found device: " + device_vendor + "/" + device_type);
}

}  // namespace device_selector

/// @brief Represents a SYCL queue with device-specific optimizations and management capabilities
class DeviceQueue {
private:
    size_t work_group_size;
    size_t work_group_size_for_parallel_reduction;

public:
    using Ptr = std::shared_ptr<DeviceQueue>;

    std::shared_ptr<sycl::queue> ptr = nullptr;  // sycl::queue pointer

    /// @brief constructor
    /// @param device sycl::device class
    DeviceQueue(const sycl::device& device) : ptr(std::make_shared<sycl::queue>(device)) {
        if (!sycl_utils::device_selector::is_supported_device(device)) {
            const std::string device_name = device.get_info<sycl::info::device::name>();
            const std::string backend_name = sycl::detail::get_backend_name_no_vendor(device.get_backend()).data();
            const std::string error_msg = device_name + " [" + backend_name + "]" + " is not supported.";
            throw std::runtime_error(error_msg);
        }
        this->work_group_size = sycl_utils::get_work_group_size(device);
        this->work_group_size_for_parallel_reduction = sycl_utils::get_work_group_size(device);
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
template <typename T, size_t Alignment = 0>
using shared_vector = std::vector<T, shared_allocator<T, Alignment>>;
template <typename T, size_t Alignment = 0>
using shared_vector_ptr = std::shared_ptr<shared_vector<T, Alignment>>;

}  // namespace sycl_points
