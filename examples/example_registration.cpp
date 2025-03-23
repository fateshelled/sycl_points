#include "sycl_points/point_cloud_reader.hpp"
#include "sycl_points/sycl_utils.hpp"
#include "sycl_points/covariance.hpp"
#include "sycl_points/downsampling.hpp"
#include "sycl_points/registration.hpp"

#include <iostream>
#include <chrono>
#include <map>

int main() {
  std::string source_filename = "../data/source.ply";
  std::string target_filename = "../data/target.ply";

  const sycl_points::PointCloudCPU source_points = sycl_points::PointCloudReader::readFile(source_filename);
  const sycl_points::PointCloudCPU target_points = sycl_points::PointCloudReader::readFile(target_filename);

  /* Specity device */
  sycl::device dev;  // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
  sycl::queue queue(dev);

  sycl_points::sycl_utils::print_device_info(queue);

  const float voxel_size = 0.25f;
  const size_t num_neighbors = 10;

  sycl_points::RegistrationParams param;
  param.max_iterations = 10;
  param.max_correspondence_distance = 1.0f;
  param.verbose = false;
  const auto reg = sycl_points::Registration(param);

  const size_t LOOP = 10;
  std::map<std::string, double> elapsed;
  for (size_t i = 0; i < LOOP + 1; ++ i) {
    auto t0 = std::chrono::high_resolution_clock::now();

    const sycl_points::PointCloudShared source_shared(queue, source_points);
    const sycl_points::PointCloudShared target_shared(queue, target_points);
    auto dt_to_shared = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    const auto source_downsampled = sycl_points::voxel_downsampling_sycl(queue, source_shared, voxel_size);
    const auto target_downsampled = sycl_points::voxel_downsampling_sycl(queue, target_shared, voxel_size);
    auto dt_downsampled = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    const auto source_tree = sycl_points::KNNSearchSYCL::buildKDTree(queue, source_downsampled);
    const auto target_tree = sycl_points::KNNSearchSYCL::buildKDTree(queue, target_downsampled);
    auto dt_build_kdtree = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    const auto source_neighbors = source_tree.searchKDTree_sycl(source_downsampled, num_neighbors);
    const auto target_neighbors = target_tree.searchKDTree_sycl(target_downsampled, num_neighbors);
    auto dt_knn_search_for_covs = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    *source_downsampled.covs = sycl_points::compute_covariances_sycl(queue, source_neighbors, *source_downsampled.points);
    *target_downsampled.covs = sycl_points::compute_covariances_sycl(queue, target_neighbors, *target_downsampled.points);
    auto dt_covariance = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    sycl_points::covariance_update_plane(source_downsampled);
    sycl_points::covariance_update_plane(target_downsampled);
    auto dt_to_plane = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    sycl_points::TransformMatrix init_T = sycl_points::TransformMatrix::Identity();
    const auto ret = reg.optimize(queue, source_downsampled, target_downsampled, target_tree, init_T);
    auto dt_registration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count();

    if (i > 0) {
      if (elapsed.count("to PointCloudShared") == 0) elapsed["to PointCloudShared"] = 0.0;
      elapsed["to PointCloudShared"] += dt_to_shared;

      if (elapsed.count("Downsampling") == 0) elapsed["Downsampling"] = 0.0;
      elapsed["Downsampling"] += dt_downsampled;

      if (elapsed.count("build KDTree") == 0) elapsed["build KDTree"] = 0.0;
      elapsed["build KDTree"] += dt_build_kdtree;

      if (elapsed.count("KNN Search") == 0) elapsed["KNN Search"] = 0.0;
      elapsed["KNN Search"] += dt_knn_search_for_covs;

      if (elapsed.count("compute Covariances") == 0) elapsed["compute Covariances"] = 0.0;
      elapsed["compute Covariances"] += dt_covariance;

      if (elapsed.count("update Covariance to plane") == 0) elapsed["update Covariance to plane"] = 0.0;
      elapsed["update Covariance to plane"] += dt_to_plane;

      if (elapsed.count("Registration") == 0) elapsed["Registration"] = 0.0;
      elapsed["Registration"] += dt_registration;
    }
    if (i == LOOP) {
      std::cout << ret.T.matrix() << std::endl;
    }
  }

  double total_elapsed = 0.0;
  for (auto [key, dt]: elapsed) {
    std::cout << key << ": " << dt / LOOP << " us" << std::endl;
    total_elapsed += dt / LOOP;
  }
  std::cout << "TOTAL: " << total_elapsed << " us" << std::endl;

}
