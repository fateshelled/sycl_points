#pragma once

#include <cassert>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace sycl_points {

template <typename T>
using host_allocator = sycl::usm_allocator<T, sycl::usm::alloc::host>;

template <typename T>
using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template <typename T>
using host_vector = std::vector<T, host_allocator<T>>;

template <typename T>
using shared_vector = std::vector<T, shared_allocator<T>>;

namespace sycl_utils {

void print_device_info(const sycl::device& device) {
  const auto platform =device.get_platform();
  std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;

  for (auto device : platform.get_devices()) {
    std::cout << "\tDevice: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "\ttype: " << (device.is_cpu() ? "CPU" : "GPU") << std::endl;
    std::cout << "\tVendorID: " << device.get_info<sycl::info::device::vendor_id>() << std::endl;
    std::cout << "\tBackend version: " << device.get_info<sycl::info::device::backend_version>() << std::endl;
    std::cout << "\tDriver version: " << device.get_info<sycl::info::device::driver_version>() << std::endl;
    std::cout << "\tGlobal Memory Size: " << device.get_info<sycl::info::device::global_mem_size>() / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
    std::cout << "\tMax Clock Frequency: " << device.get_info<sycl::info::device::max_clock_frequency>() / 1000.0 << " GHz" << std::endl;
    std::cout << "\tDouble precision support: " << (device.has(sycl::aspect::fp64) ? "true" : "false") << std::endl;
    std::cout << "\tAvailable: " << (device.get_info<sycl::info::device::is_available>() ? "true" : "false") << std::endl;
  }
}

void print_device_info(const sycl::queue& queue) {
  print_device_info(queue.get_device());
}

}  // namespace sycl_utils
}  // namespace sycl_points
