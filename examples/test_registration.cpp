#include "sycl_points/point_cloud_reader.hpp"
#include "sycl_points/sycl_utils.hpp"
#include "sycl_points/covariance.hpp"
#include "sycl_points/downsampling.hpp"
#include "sycl_points/registration.hpp"

#include <iostream>
#include <chrono>

int main() {
  std::string source_filename = "../data/source.ply";
  std::string target_filename = "../data/target.ply";

  const sycl_points::PointCloudCPU source_points = sycl_points::PointCloudReader::readFile(source_filename);
  const sycl_points::PointCloudCPU target_points = sycl_points::PointCloudReader::readFile(target_filename);

  /* Specity device */
  sycl::device dev;  // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
  sycl::queue queue(dev);

  sycl_points::sycl_utils::print_device_info(queue);

  sycl_points::PointCloudShared source_shared(queue, source_points);
  sycl_points::PointCloudShared target_shared(queue, target_points);

  const float voxel_size = 0.25f;
  const size_t num_neighbors = 10;

  source_shared = sycl_points::voxel_downsampling_sycl(queue, source_shared, voxel_size);
  target_shared = sycl_points::voxel_downsampling_sycl(queue, target_shared, voxel_size);

  const auto source_tree = sycl_points::KNNSearchSYCL::buildKDTree(queue, source_shared);
  const auto target_tree = sycl_points::KNNSearchSYCL::buildKDTree(queue, target_shared);

  sycl_points::compute_covariances_sycl(source_tree, source_shared, num_neighbors);
  sycl_points::compute_covariances_sycl(target_tree, target_shared, num_neighbors);

  sycl_points::covariance_update_plane(source_shared);
  sycl_points::covariance_update_plane(target_shared);

  auto reg = sycl_points::Registration();
  const auto ret = reg.optimize(queue, source_shared, target_shared, target_tree);

  std::cout << ret.T.matrix() << std::endl;

}
