#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <sycl/sycl.hpp>

namespace sycl_points {

namespace sycl_utils {

namespace VENDOR_ID {
constexpr uint32_t INTEL = 0x8086;   // 32902
constexpr uint32_t NVIDIA = 0x10de;  // 4318
constexpr uint32_t AMD = 0x1002;     // 4098
#ifdef SYCL_IMPL_ADAPTIVECPP
constexpr uint32_t OMP = 0xffffffff;  // 4294967295
#endif
};  // namespace VENDOR_ID

/// @brief Get backend name as string
/// @param backend sycl::backend value
/// @return backend name string
inline std::string get_backend_name(sycl::backend backend) {
#ifdef SYCL_IMPL_ADAPTIVECPP
    // hipsycl::rt::backend_id: cuda=0, hip=1, level_zero=2, ocl=3, omp=4
    static constexpr const char* BackendNames[] = {"CUDA", "HIP", "Level Zero", "OpenCL", "OpenMP"};
    const auto idx = static_cast<int>(backend);
    if (idx >= 0 && idx < static_cast<int>(sizeof(BackendNames) / sizeof(BackendNames[0]))) {
        return BackendNames[idx];
    }
    return "Unknown";
#else
    std::ostringstream oss;
    oss << backend;
    return oss.str();
#endif
}

/// @brief Get device info
/// @param device SYCL device
inline std::string get_device_info(const sycl::device& device) {
    const auto platform = device.get_platform();
    std::ostringstream oss;
    oss << "Platform: " << platform.get_info<sycl::info::platform::name>() << "\n";

    for (auto device : platform.get_devices()) {
        oss << "\tDevice: " << device.get_info<sycl::info::device::name>() << "\n";
        oss << "\ttype: " << (device.is_cpu() ? "CPU" : "GPU") << "\n";
        oss << "\tVendor: " << device.get_info<sycl::info::device::vendor>() << "\n";
        oss << "\tVendorID: " << device.get_info<sycl::info::device::vendor_id>() << "\n";
        oss << "\tBackend name: " << get_backend_name(device.get_backend()) << "\n";
#ifndef SYCL_IMPL_ADAPTIVECPP
        oss << "\tBackend version: " << device.get_info<sycl::info::device::backend_version>() << "\n";
#endif
        // NOTE: driver_version cannot be obtained on the Level Zero backend (AdaptiveCpp bug:
        // uninitialized ze_driver_properties_t is passed to zeDriverGetProperties, causing SEGV).
        // oss << "\tDriver version: " << device.get_info<sycl::info::device::driver_version>() << "\n";
        oss << "\tGlobal Memory Size: "
            << device.get_info<sycl::info::device::global_mem_size>() / 1024.0 / 1024.0 / 1024.0 << " GB" << "\n";
        oss << "\tLocal Memory Size: " << device.get_info<sycl::info::device::local_mem_size>() / 1024.0 << " KB"
            << "\n";
        oss << "\tGlobal Memory Cache Size: "
            << device.get_info<sycl::info::device::global_mem_cache_size>() / 1024.0 / 1024.0 << " MB" << "\n";
        oss << "\tGlobal Memory Cache Line Size: " << device.get_info<sycl::info::device::global_mem_cache_line_size>()
            << " byte" << "\n";

        oss << "\tMax Memory Allocation Size: "
            << device.get_info<sycl::info::device::max_mem_alloc_size>() / 1024.0 / 1024.0 / 1024.0 << " GB"
            << "\n";

        oss << "\tMax Work Group Size: " << device.get_info<sycl::info::device::max_work_group_size>() << "\n";
        oss << "\tMax Work Item Dimensions: " << device.get_info<sycl::info::device::max_work_item_dimensions>()
            << "\n";
        oss << "\tMax Work Item Sizes: [";
        oss << device.get_info<sycl::info::device::max_work_item_sizes<1>>().dimensions << ", ";
        oss << device.get_info<sycl::info::device::max_work_item_sizes<1>>().dimensions << ", ";
        oss << device.get_info<sycl::info::device::max_work_item_sizes<1>>().dimensions << "]" << "\n";
        oss << "\tMax Sub Groups num: " << device.get_info<sycl::info::device::max_num_sub_groups>() << "\n";
        oss << "\tSub Group Sizes: [";
        const auto subgroup_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
        for (size_t i = 0; i < subgroup_sizes.size(); ++i) {
            oss << subgroup_sizes[i];
            if (i < subgroup_sizes.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]" << "\n";

        oss << "\tMax compute units: " << device.get_info<sycl::info::device::max_compute_units>() << "\n";

        oss << "\tMax Clock Frequency: " << device.get_info<sycl::info::device::max_clock_frequency>() / 1000.0
            << " GHz" << "\n";

        oss << "\tDouble precision support: " << (device.has(sycl::aspect::fp64) ? "true" : "false") << "\n";

        oss << "\tAtomic 64bit support: " << (device.has(sycl::aspect::atomic64) ? "true" : "false") << "\n";

        oss << "\tUSM host allocations: " << (device.has(sycl::aspect::usm_host_allocations) ? "true" : "false")
            << "\n";
        oss << "\tUSM device allocations: " << (device.has(sycl::aspect::usm_device_allocations) ? "true" : "false")
            << "\n";
        oss << "\tUSM shared allocations: " << (device.has(sycl::aspect::usm_shared_allocations) ? "true" : "false")
            << "\n";

        oss << "\tUSM atomic shared allocations: "
            << (device.has(sycl::aspect::usm_atomic_shared_allocations) ? "true" : "false") << "\n";

        oss << "\tAvailable: " << (device.get_info<sycl::info::device::is_available>() ? "true" : "false") << "\n";
    }
    return oss.str();
}

/// @brief Get device info
/// @param queue SYCL queue
inline std::string get_device_info(const sycl::queue& queue) { return get_device_info(queue.get_device()); }

/// @brief Print selected device info
/// @param queue SYCL queue
inline void print_device_info(const sycl::device& device) { std::cout << get_device_info(device) << std::endl; }

/// @brief Print selected device info
/// @param queue SYCL queue
inline void print_device_info(const sycl::queue& queue) { print_device_info(queue.get_device()); }

/// @brief device is CPU or not
inline bool is_cpu(const sycl::device& device) { return device.is_cpu(); }
inline bool is_cpu(const sycl::queue& queue) { return is_cpu(queue.get_device()); }

/// @brief device is iGPU/dGPU or not
inline bool is_gpu(const sycl::device& device) { return device.is_gpu(); }
inline bool is_gpu(const sycl::queue& queue) { return queue.get_device().is_gpu(); }

/// @brief device is FPGA or not
inline bool is_accelerator(const sycl::queue& queue) { return queue.get_device().is_accelerator(); }

/// @brief device vendor is NVIDIA or not
inline bool is_nvidia(const sycl::device& device) {
    const auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
    return vendor_id == VENDOR_ID::NVIDIA;
}
inline bool is_nvidia(const sycl::queue& queue) {
    const auto device = queue.get_device();
    return is_nvidia(device);
}

/// @brief device vendor is Intel or not
inline bool is_intel(const sycl::device& device) {
    const auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
    return vendor_id == VENDOR_ID::INTEL;
}
inline bool is_intel(const sycl::queue& queue) {
    const auto device = queue.get_device();
    return is_intel(device);
}

/// @brief device vendor is AMD or not
inline bool is_amd(const sycl::device& device) {
    const auto vendor_id = device.get_info<sycl::info::device::vendor_id>();
    return vendor_id == VENDOR_ID::AMD;
}
inline bool is_amd(const sycl::queue& queue) {
    const auto device = queue.get_device();
    return is_amd(device);
}

inline bool enable_shared_allocations(const sycl::device& device) {
    return device.has(sycl::aspect::usm_shared_allocations);
}

namespace device_selector {

inline bool is_supported_device(const sycl::device& dev) { return enable_shared_allocations(dev); }

inline int default_selector_v(const sycl::device& dev) {
    if (!is_supported_device(dev)) return -1;

    // AdaptiveCpp always exposes its OpenMP host device, even when a GPU
    // visibility mask is set. Prefer a GPU so a visible Level Zero device is
    // actually selected instead of silently falling back to the host.
    return dev.is_gpu() ? 2 : 1;
}

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

inline sycl::device select_device(const std::string& device_vendor, const std::string& device_type) {
    std::string device_vendor_upper = device_vendor;
    std::transform(device_vendor_upper.begin(), device_vendor_upper.end(), device_vendor_upper.begin(),
                   [](u_char c) { return std::toupper(c); });
    std::string device_type_upper = device_type;
    std::transform(device_type_upper.begin(), device_type_upper.end(), device_type_upper.begin(),
                   [](u_char c) { return std::toupper(c); });

    uint32_t vendor_id = 0;
    if (device_vendor_upper == "INTEL") {
        vendor_id = VENDOR_ID::INTEL;
    } else if (device_vendor_upper == "NVIDIA") {
        vendor_id = VENDOR_ID::NVIDIA;
    } else if (device_vendor_upper == "AMD") {
        vendor_id = VENDOR_ID::AMD;
#ifdef SYCL_IMPL_ADAPTIVECPP
    } else if (device_vendor_upper == "OMP") {
        vendor_id = VENDOR_ID::OMP;
#endif
    } else if (device_vendor_upper == "DEFAULT") {
        const auto device_selector = sycl_points::sycl_utils::device_selector::default_selector_v;
        return sycl::device{device_selector};
    } else {
        throw std::runtime_error("[device_selector::select_device] invalid device vendor: " + device_vendor);
    }

    const bool select_cpu = device_type_upper == "CPU" ? true : false;
    const bool select_gpu = device_type_upper == "GPU" ? true : false;
    if (!select_cpu && !select_gpu) {
        throw std::runtime_error("[device_selector::select_device] not support device type: " + device_type);
    }

    for (auto platform : sycl::platform::get_platforms()) {
        for (auto device : platform.get_devices()) {
            const auto dev_vendor_id = device.get_info<sycl::info::device::vendor_id>();
            if (vendor_id == dev_vendor_id) {
#ifdef SYCL_IMPL_ADAPTIVECPP
                if (vendor_id == VENDOR_ID::OMP) {
                    return device;
                }
#endif
                if (select_cpu && device.is_cpu()) {
                    return device;
                }
                if (select_gpu && device.is_gpu()) {
                    return device;
                }
            }
#ifdef SYCL_IMPL_ADAPTIVECPP
            // AdaptiveCpp: any CPU vendor is exposed via OMP backend
            else if (select_cpu && dev_vendor_id == VENDOR_ID::OMP) {
                return device;
            }
#endif
        }
    }
    throw std::runtime_error("[device_selector::select_device] not found device: " + device_vendor + "/" + device_type);
}

}  // namespace device_selector

}  // namespace sycl_utils

}  // namespace sycl_points
