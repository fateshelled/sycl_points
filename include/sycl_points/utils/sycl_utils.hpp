#pragma once

#include <cassert>

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
};                                   // namespace VENDOR_ID

inline constexpr bool is_not_power_of_two(size_t alignment) { return (alignment & (alignment - 1)) != 0; }

inline size_t get_work_group_size(const sycl::device& device, size_t work_group_size = 0) {
    if (work_group_size > 0) {
        return std::min(work_group_size, device.get_info<sycl::info::device::max_work_group_size>());
    }

    const auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
    switch (vendor_id) {
        case 32902:  // Intel
            if (device.is_cpu()) {
                return std::min((size_t)16, device.get_info<sycl::info::device::max_work_group_size>());
            } else {
                return std::min((size_t)32, device.get_info<sycl::info::device::max_work_group_size>());
            }
            break;
        case 4318:  // NVIDIA
            return std::min((size_t)256, device.get_info<sycl::info::device::max_work_group_size>());
            break;
    }
    return std::min((size_t)128, device.get_info<sycl::info::device::max_work_group_size>());
}

inline size_t get_work_group_size(const sycl::queue& queue, size_t work_group_size = 0) {
    return get_work_group_size(queue.get_device(), work_group_size);
}

inline void print_device_info(const sycl::device& device) {
    const auto platform = device.get_platform();
    std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;

    for (auto device : platform.get_devices()) {
        std::cout << "\tDevice: " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "\ttype: " << (device.is_cpu() ? "CPU" : "GPU") << std::endl;
        std::cout << "\tVendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
        std::cout << "\tVendorID: " << device.get_info<sycl::info::device::vendor_id>() << std::endl;
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

        std::cout << "\tUSM host allocations: " << (device.has(sycl::aspect::usm_host_allocations) ? "true" : "false")<< std::endl;
        std::cout << "\tUSM device allocations: " << (device.has(sycl::aspect::usm_device_allocations) ? "true" : "false")<< std::endl;
        std::cout << "\tUSM shared allocations: " << (device.has(sycl::aspect::usm_shared_allocations) ? "true" : "false")<< std::endl;

        std::cout << "\tAvailable: " << (device.get_info<sycl::info::device::is_available>() ? "true" : "false")
                  << std::endl;
        std::cout << std::endl;
    }
}

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

struct events {
    std::vector<sycl::event> evs;

    void push_back(const sycl::event& event) { evs.push_back(event); }
    void wait() {
        while (evs.size() > 0) {
            auto& event = evs.back();
            event.wait();
            evs.pop_back();
        }
    }
    void operator+=(const sycl::event event) { evs.push_back(event); }
    void operator+=(const events& e) { std::copy(e.evs.begin(), e.evs.end(), std::back_inserter(evs)); }
};

namespace device_selector {

inline int intel_cpu_selector_v(const sycl::device& dev) {
    const auto vendor_id = dev.get_info<sycl::info::device::vendor_id>();
    return dev.is_cpu() && (vendor_id == VENDOR_ID::INTEL);
}

inline int intel_gpu_selector_v(const sycl::device& dev) {
    const auto vendor_id = dev.get_info<sycl::info::device::vendor_id>();
    return dev.is_gpu() && (vendor_id == VENDOR_ID::INTEL);
}

inline int nvidia_gpu_selector_v(const sycl::device& dev) {
    const auto vendor_id = dev.get_info<sycl::info::device::vendor_id>();
    return dev.is_gpu() && (vendor_id == VENDOR_ID::NVIDIA);
}

}  // namespace device_selector

}  // namespace sycl_utils

template <typename T, size_t Alignment = 0>
using host_allocator = sycl::usm_allocator<T, sycl::usm::alloc::host, Alignment>;

template <typename T, size_t Alignment = 0>
using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared, Alignment>;

// template <typename T, size_t Alignment = 0>
// using device_allocator = sycl::usm_allocator<T, sycl::usm::alloc::device, Alignment>;

template <typename T, size_t Alignment = 0>
using host_vector = std::vector<T, host_allocator<T, Alignment>>;

template <typename T, size_t Alignment = 0>
using shared_vector = std::vector<T, shared_allocator<T, Alignment>>;

// template <typename T, size_t Alignment = 0>
// using device_vector = std::vector<T, device_allocator<T, Alignment>>;

template <typename T, size_t Alignment = 0>
struct ContainerDevice {
    T* data = nullptr;
    size_t size = 0;
    std::shared_ptr<sycl::queue> queue_ptr = nullptr;
    const sycl::property_list propeties = {
        // sycl::property::no_init()
    };

    ContainerDevice(const std::shared_ptr<sycl::queue>& q) : queue_ptr(q) {
        if constexpr (sycl_utils::is_not_power_of_two(Alignment)) {
            static_assert("Alignment must be power of two");
        }
    }
    ~ContainerDevice() { free(); }

    // copy
    ContainerDevice(const ContainerDevice&) = delete;
    ContainerDevice& operator=(const ContainerDevice&) = delete;

    // move
    ContainerDevice(ContainerDevice&& other) noexcept
        : data(other.data), size(other.size), queue_ptr(std::move(other.queue_ptr)) {
        other.data = nullptr;
        other.size = 0;
    }

    // move
    ContainerDevice& operator=(ContainerDevice&& other) noexcept {
        if (this != &other) {
            free();
            data = other.data;
            size = other.size;
            queue_ptr = std::move(other.queue_ptr);
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void resize(size_t N) {
        if (this->size == N) return;
        this->free();
        this->size = N;
        this->data = sycl::aligned_alloc_device<T>(Alignment, N, *this->queue_ptr, this->propeties);
    }

    void free() {
        if (this->data != nullptr) {
            sycl::free(this->data, *this->queue_ptr);
            this->data = nullptr;
            this->size = 0;
        }
    }

    sycl_utils::events memset_async(const T& value, size_t start_index = 0, size_t count = 0) {
        sycl_utils::events events;
        if (this->data == nullptr || this->size == 0 || start_index >= this->size) return events;

        const size_t actual_count =
            (count == 0 || start_index + count > this->size) ? (this->size - start_index) : count;

        events += this->queue_ptr->memset(this->data + start_index, value, actual_count * sizeof(T));
        return events;
    }

    // copy to device memory
    sycl_utils::events memcpy_async(const T* ptr, size_t src_size, size_t src_start = 0, size_t dst_start = 0,
                                    size_t count = 0) {
        sycl_utils::events events;
        if (src_size == 0 || src_start >= src_size) return events;
        if (this->data == nullptr || this->size == 0 || dst_start >= this->size) return events;

        const size_t max_copy_size = src_size - src_start;
        const size_t actual_count = (count == 0 || count > max_copy_size) ? max_copy_size : count;

        events += this->queue_ptr->memcpy(this->data + dst_start, ptr, actual_count * sizeof(T));
        return events;
    }
};

}  // namespace sycl_points
