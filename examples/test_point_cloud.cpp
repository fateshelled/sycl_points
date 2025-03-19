#include "sycl_points/point_cloud_reader.hpp"
#include "sycl_points/sycl_utils.hpp"

#include <iostream>
#include <chrono>

int main() {
  std::string source_filename = "../data/source.ply";

  sycl_points::PointCloudCPU source_points = sycl_points::PointCloudReader::readFile(source_filename);

  /* Specity device */
  sycl::device dev;  // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
  sycl::queue queue(dev);

  sycl_points::sycl_utils::print_device_info(queue);

  auto s = std::chrono::high_resolution_clock::now();
  sycl_points::PointCloudShared shared_points(queue, source_points);
  // sycl_points::PointCloudShared pts(queue);
  auto dt_to_pointcloud_shared = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();


  double dt_transform_cpu = 0.0;
  for (size_t i = 0; i < 11; ++i) {
    s = std::chrono::high_resolution_clock::now();
    shared_points.transform_cpu(Eigen::Matrix4f::Identity());
    if (i > 0) {
      dt_transform_cpu += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();
    }
  }
  dt_transform_cpu /= 10;

  double dt_transform_copy = 0.0;
  for (size_t i = 0; i < 11; ++i) {
    s = std::chrono::high_resolution_clock::now();
    auto ret = shared_points.transform_sycl_copy(Eigen::Matrix4f::Identity() * 2);
    if (i == 0) {
      // for (size_t j = 0; j < 10; ++j) {
      //   std::cout << "source: " << source_points.points[j].transpose() << std::endl;
      //   std::cout << "shared: " << (*shared_points.points)[j].transpose() << std::endl;
      //   std::cout << "ret: " << (*ret.points)[j].transpose() << std::endl;
      // }
    } else {
      dt_transform_copy += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();
    }
  }
  dt_transform_copy /= 10;

  double dt_transform_zerocopy = 0.0;
  for (size_t i = 0; i < 11; ++i) {
    s = std::chrono::high_resolution_clock::now();
    shared_points.transform_sycl(Eigen::Matrix4f::Identity());
    if (i > 0) {
      dt_transform_zerocopy += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();
    }
  }
  dt_transform_zerocopy /= 10;

  {

    // for (size_t i = 0; i < 10; ++i) {
    //   std::cout << "[" << i << "] " << source_points.points[i].transpose() << std::endl;
    // }


    // // validate
    // for (size_t i = 0; i < transformed_points.size(); ++i) {
    //   const auto delta = transformed_points.points[i] - shared_results[i];
    //   if (delta.norm() > 1e-5) {
    //     std::cout << "ERROR: [" << i << "]" << std::endl;
    //     std::cout << transformed_points.points[i].transpose() << std::endl;
    //     std::cout << shared_results[i].transpose() << std::endl;
    //     break;
    //   }
    // }

    std::cout << "to PointCloudShared: " << dt_to_pointcloud_shared << " us" << std::endl;
    std::cout << "transform on cpu (shared_ptr, zero copy): " << dt_transform_cpu << " us" << std::endl;
    std::cout << "transform on device (shared_ptr, copy): " << dt_transform_copy << " us" << std::endl;
    std::cout << "transform on device (shared_ptr, zero copy): " << dt_transform_zerocopy << " us" << std::endl;
  }

  return 0;
}
