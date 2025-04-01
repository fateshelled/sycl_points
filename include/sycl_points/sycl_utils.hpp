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

constexpr size_t default_work_group_size = 256;

void print_device_info(const sycl::device& device) {
  const auto platform = device.get_platform();
  std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;

  for (auto device : platform.get_devices()) {
    std::cout << "\tDevice: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "\ttype: " << (device.is_cpu() ? "CPU" : "GPU") << std::endl;
    std::cout << "\tVendorID: " << device.get_info<sycl::info::device::vendor_id>() << std::endl;
    std::cout << "\tBackend version: " << device.get_info<sycl::info::device::backend_version>() << std::endl;
    std::cout << "\tDriver version: " << device.get_info<sycl::info::device::driver_version>() << std::endl;
    std::cout << "\tGlobal Memory Size: " << device.get_info<sycl::info::device::global_mem_size>() / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
    std::cout << "\tLocal Memory Size: " << device.get_info<sycl::info::device::local_mem_size>() / 1024.0 << " KB" << std::endl;
    std::cout << "\tMax Memory Allocation Size: " << device.get_info<sycl::info::device::max_mem_alloc_size>() / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;

    std::cout << "\tMax Work Group Size: " << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
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

    std::cout << "\tMax Clock Frequency: " << device.get_info<sycl::info::device::max_clock_frequency>() / 1000.0 << " GHz" << std::endl;
    std::cout << "\tDouble precision support: " << (device.has(sycl::aspect::fp64) ? "true" : "false") << std::endl;
    std::cout << "\tAvailable: " << (device.get_info<sycl::info::device::is_available>() ? "true" : "false") << std::endl;
    std::cout << std::endl;
  }
}

void print_device_info(const sycl::queue& queue) {
  print_device_info(queue.get_device());
}

struct events {
  std::vector<sycl::event> events;

  void push_back(const sycl::event& event) { events.push_back(event); }
  void wait() {
    while (events.size() > 0) {
      auto& event = events.back();
      event.wait();
      events.pop_back();
    }
  }
};

}  // namespace sycl_utils
}  // namespace sycl_points
