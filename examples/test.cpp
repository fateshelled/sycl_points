#include "sycl_points/point_cloud.hpp"
#include "sycl_points/point_cloud_reader.hpp"
#include "sycl_points/knn_search.hpp"
#include "sycl_points/downsampling.hpp"
#include "sycl_points/covariance.hpp"

#include <chrono>

using PointCloudCPU = sycl_points::PointCloudCPU<float>;
using PointCloudDevice = sycl_points::PointCloudDevice<float>;

int main() {

  std::string source_filename = "../data/source.ply";
  std::string target_filename = "../data/target.ply";

  using PCReader = sycl_points::PointCloudReader<float>;
  PointCloudCPU source_points = PCReader::readFile(source_filename);
  PointCloudCPU target_points = PCReader::readFile(target_filename);

  /* Specity device */
  sycl::device dev; // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
  sycl::queue queue(dev);

  // downsampled
  auto downsampled = sycl_points::voxel_downsampling_sycl(queue, source_points, 0.5f); // 3.5ms

  // 点群データの処理例（最初の10点を表示）
  int count = std::min(10, static_cast<int>(source_points.size()));
  for (int i = 0; i < count; i++) {
    const auto& point = source_points.points[i];
    std::cout << "Source " << i << ": " << point.transpose() << std::endl;
  }
  count = std::min(10, static_cast<int>(target_points.size()));
  for (int i = 0; i < count; i++) {
    const auto& point = target_points.points[i];
    std::cout << "Targets " << i << ": " << point.transpose() << std::endl;
  }

  double elapsed_downsample = 0.0;
  for (size_t i = 0; i < 10; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto tmp = sycl_points::voxel_downsampling_sycl(queue, source_points.points, 0.5f);
    auto end = std::chrono::high_resolution_clock::now();
    elapsed_downsample += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  std::cout << "SYCL voxel_downsampled: " << elapsed_downsample / 10 << " us." << std::endl;

  // Transform
  Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
  trans.block(0, 0, 3, 3) = Eigen::AngleAxisf(0.5 * M_PI, Eigen::Vector3f(0, 1, 0)).matrix(); // rotate 90 deg, y axis
  trans.block(0, 3, 3, 1) << 1.0, 2.0, 3.0;
  const auto transformed_points = source_points.transform_copy(trans);

  {
    double elapsed_transform_cpu = 0.0;
    for (size_t i = 0; i < 10; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      const auto tmp = source_points.transform_copy(trans);
      auto end = std::chrono::high_resolution_clock::now();
      elapsed_transform_cpu += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "CPU transform Points: " << elapsed_transform_cpu / 10 << " us." << std::endl;
  }

  {
    double elapsed_copy_to_device = 0.0;
    double elapsed_transform_sycl = 0.0;
    double elapsed_copy_to_host = 0.0;
    for (size_t i = 0; i < 10; ++i) {
      auto s0 = std::chrono::high_resolution_clock::now();
      PointCloudDevice points_dev(queue, source_points);
      auto e0 = std::chrono::high_resolution_clock::now();

      auto s1 = std::chrono::high_resolution_clock::now();
      PointCloudDevice transformed;
      points_dev.transform_copy(trans, transformed);
      auto e1 = std::chrono::high_resolution_clock::now();

      auto s2 = std::chrono::high_resolution_clock::now();
      PointCloudCPU points_host;
      auto events = transformed.copyToHost(points_host);
      queue.wait();
      auto e2 = std::chrono::high_resolution_clock::now();

      elapsed_copy_to_device += std::chrono::duration_cast<std::chrono::microseconds>(e0 - s0).count();
      elapsed_transform_sycl += std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count();
      elapsed_copy_to_host += std::chrono::duration_cast<std::chrono::microseconds>(e2 - s2).count();
    }
    std::cout << "SYCL Copy to Device: " << elapsed_copy_to_device / 10 << " us." << std::endl;
    std::cout << "SYCL transform Points: " << elapsed_transform_sycl / 10 << " us." << std::endl;
    std::cout << "SYCL Copy to Host: " << elapsed_copy_to_host / 10 << " us." << std::endl;
  }

  // Covariances
  auto kdtree = sycl_points::KNNSearch<float>::buildKDTree(queue, downsampled); // 1ms
  // const auto neighbors = kdtree.searchKDTree_sycl(source_points, 20);
  sycl_points::computeCovariances<float>(kdtree, downsampled, 20); // 3ms
  {
    double elapsed_covariances = 0.0;
    for (size_t i = 0; i < 10; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      const auto tmp = sycl_points::computeCovariances<float>(queue, downsampled.points, 20);
      auto end = std::chrono::high_resolution_clock::now();
      elapsed_covariances += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "Covariances : " << elapsed_covariances / 10 << " us." << std::endl;
  }


  {
    double elapsed_transform_cpu = 0.0;
    for (size_t i = 0; i < 10; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      const auto tmp = downsampled.transform_copy(trans);
      auto end = std::chrono::high_resolution_clock::now();
      elapsed_transform_cpu += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "CPU transform Points and Covs: " << elapsed_transform_cpu / 10 << " us." << std::endl;
  }
  {
    auto trans_cpu = downsampled.transform_copy(trans);
    auto pts_cpu = trans_cpu.points;
    auto cov_cpu = trans_cpu.covs;

    PointCloudDevice points_dev(queue, downsampled);
    PointCloudDevice trans_dev;
    points_dev.transform_copy(trans, trans_dev);
    PointCloudCPU trans_dev_cpu;
    auto events = trans_dev.copyToHost(trans_dev_cpu);
    queue.wait();
    auto pts_sycl = trans_dev_cpu.points;
    auto cov_sycl = trans_dev_cpu.covs;
    bool success = true;
    if (pts_cpu.size() != pts_sycl.size()) {
      std::cout << "Points size is not same" << std::endl;
      success = false;
    } else {
      for (size_t i = 0; i < pts_cpu.size(); ++i) {
        sycl_points::PointType<float> delta = pts_cpu[i] - pts_sycl[i];
        if (delta.norm() > 1e-5f) {
          std::cout << "incorrect @ [ " << i << " ]" << std::endl
                    << pts_cpu[i] << std::endl
                    << pts_sycl[i] << std::endl;
          success = false;
          break;
        }
      }
    }

    if (cov_cpu.size() != cov_sycl.size()) {
      std::cout << "Covs size is not same" << std::endl;
      success = false;
    } else {
      for (size_t i = 0; i < cov_cpu.size(); ++i) {
        sycl_points::Covariance<float> delta = cov_cpu[i] - cov_sycl[i];
        if (delta.norm() > 1e-5f) {
          std::cout << "incorrect @ [ " << i << " ]" << std::endl;
          std::cout << cov_cpu[i] << std::endl << std::endl;
          std::cout << cov_sycl[i] << std::endl << std::endl;
          std::cout << delta << std::endl;
          success = false;
          break;
        }
      }
    }
    if (success) {
      std::cout << "TRANSFORM SUCCESS" << std::endl;
    } else {
      std::cout << "TRANSFORM FAILED" << std::endl;
    }
  }

  {
    double elapsed_copy_to_device = 0.0;
    double elapsed_transform_sycl = 0.0;
    double elapsed_copy_to_host = 0.0;
    for (size_t i = 0; i < 10; ++i) {
      auto s0 = std::chrono::high_resolution_clock::now();
      PointCloudDevice points_dev(queue, downsampled);
      auto e0 = std::chrono::high_resolution_clock::now();

      auto s1 = std::chrono::high_resolution_clock::now();
      PointCloudDevice transformed;
      points_dev.transform_copy(trans, transformed);
      auto e1 = std::chrono::high_resolution_clock::now();

      auto s2 = std::chrono::high_resolution_clock::now();
      PointCloudCPU points_host;
      auto events = transformed.copyToHost(points_host);
      queue.wait();
      auto e2 = std::chrono::high_resolution_clock::now();
      elapsed_copy_to_device += std::chrono::duration_cast<std::chrono::microseconds>(e0 - s0).count();
      elapsed_transform_sycl += std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count();
      elapsed_copy_to_host += std::chrono::duration_cast<std::chrono::microseconds>(e2 - s2).count();
    }
    std::cout << "SYCL Copy to Device: " << elapsed_copy_to_device / 10 << " us." << std::endl;
    std::cout << "SYCL transform Points and Covs: " << elapsed_transform_sycl / 10 << " us." << std::endl;
    std::cout << "SYCL Copy to Host: " << elapsed_copy_to_host / 10 << " us." << std::endl;
  }

  return 0;
}
