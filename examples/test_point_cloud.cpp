#include "sycl_points/point_cloud_reader.hpp"
#include "sycl_points/sycl_utils.hpp"
#include "sycl_points/covariance.hpp"
#include "sycl_points/downsampling.hpp"

#include <iostream>
#include <chrono>

int main() {
  std::string source_filename = "../data/source.ply";

  const sycl_points::PointCloudCPU source_points = sycl_points::PointCloudReader::readFile(source_filename);

  /* Specity device */
  sycl::device dev;  // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
  sycl::queue queue(dev);

  sycl_points::sycl_utils::print_device_info(queue);

  auto s = std::chrono::high_resolution_clock::now();
  sycl_points::PointCloudShared shared_points(queue, source_points);
  // sycl_points::PointCloudShared pts(queue);
  const double dt_to_pointcloud_shared = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();

  // KDTree
  double dt_build_kdtree = 0.0;
  for (size_t i = 0; i < 10; ++i) {
    s = std::chrono::high_resolution_clock::now();
    auto kdtree_cpu = sycl_points::KNNSearch::buildKDTree(source_points);
    dt_build_kdtree += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();
  }
  dt_build_kdtree /= 10;

  auto kdtree_sycl = sycl_points::KNNSearchSYCL::buildKDTree(queue, shared_points);
  double dt_build_kdtree_sycl = 0.0;
  for (size_t i = 0; i < 10; ++i) {
    s = std::chrono::high_resolution_clock::now();
    auto tmp = sycl_points::KNNSearchSYCL::buildKDTree(queue, shared_points);
    dt_build_kdtree_sycl += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();
  }
  dt_build_kdtree_sycl /= 10;

  // Covariance
  double dt_covariances_sycl = 0.0;
  const size_t k_correspondence_covariance = 10;
  for (size_t i = 0; i < 11; ++i) {
    s = std::chrono::high_resolution_clock::now();
    sycl_points::compute_covariances_sycl(kdtree_sycl, shared_points, k_correspondence_covariance);
    if (i > 0) {
      dt_covariances_sycl += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();
    }
  }
  dt_covariances_sycl /= 10;

  // Downsampling
  double dt_voxel_downsampling = 0.0;
  const float voxel_size = 1.0;
  for (size_t i = 0; i < 11; ++i) {
    s = std::chrono::high_resolution_clock::now();
    auto downsampled = sycl_points::voxel_downsampling_sycl(queue, shared_points, voxel_size);
    if (i > 0) {
      dt_voxel_downsampling += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();
    }
  }
  dt_voxel_downsampling /= 10;

  // Transform
  double dt_transform_cpu_copy = 0.0;
  for (size_t i = 0; i < 11; ++i) {
    s = std::chrono::high_resolution_clock::now();
    auto tmp = shared_points.transform_cpu_copy(Eigen::Matrix4f::Identity());
    if (i > 0) {
      dt_transform_cpu_copy += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();
    }
  }
  dt_transform_cpu_copy /= 10;

  double dt_transform_cpu_zerocopy = 0.0;
  for (size_t i = 0; i < 11; ++i) {
    s = std::chrono::high_resolution_clock::now();
    shared_points.transform_cpu(Eigen::Matrix4f::Identity());
    if (i > 0) {
      dt_transform_cpu_zerocopy += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();
    }
  }
  dt_transform_cpu_zerocopy /= 10;

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

    std::cout << "Source: " << source_points.size() << " points" << std::endl;
    std::cout << "to PointCloudShared: " << dt_to_pointcloud_shared << " us" << std::endl;
    std::cout << "Build KDTree (CPU): " << dt_build_kdtree << " us" << std::endl;
    std::cout << "Build KDTree (shared_ptr): " << dt_build_kdtree_sycl << " us" << std::endl;
    std::cout << "Compute covariances on device (shared_ptr): " << dt_covariances_sycl << " us" << std::endl;
    std::cout << "Voxel downsampling (shared_ptr): " << dt_voxel_downsampling << " us" << std::endl;
    std::cout << "transform on cpu (shared_ptr, copy): " << dt_transform_cpu_copy << " us" << std::endl;
    std::cout << "transform on cpu (shared_ptr, zero copy): " << dt_transform_cpu_zerocopy << " us" << std::endl;
    std::cout << "transform on device (shared_ptr, copy): " << dt_transform_copy << " us" << std::endl;
    std::cout << "transform on device (shared_ptr, zero copy): " << dt_transform_zerocopy << " us" << std::endl;
  }

  return 0;
}
