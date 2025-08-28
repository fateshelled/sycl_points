#include <chrono>
#include <iostream>
#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/algorithms/voxel_downsampling.hpp>
#include <sycl_points/algorithms/transform.hpp>
#include <sycl_points/io/point_cloud_reader.hpp>

int main() {
    std::string source_filename = "../data/source.ply";

    const sycl_points::PointCloudCPU source_points = sycl_points::PointCloudReader::readFile(source_filename);

    /* Specity device */
    const auto device_selector = sycl_points::sycl_utils::device_selector::default_selector_v;
    sycl::device dev(device_selector);  // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
    sycl_points::sycl_utils::DeviceQueue queue(dev);

    queue.print_device_info();

    auto s = std::chrono::high_resolution_clock::now();
    sycl_points::PointCloudShared shared_points(queue, source_points);
    // sycl_points::PointCloudShared pts(queue);
    const double dt_to_pointcloud_shared =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();

    // KDTree
    double dt_build_kdtree = 0.0;
    for (size_t i = 0; i < 10; ++i) {
        s = std::chrono::high_resolution_clock::now();
        auto tmp = sycl_points::algorithms::knn_search::KDTree::build(queue, shared_points);
        dt_build_kdtree +=
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                .count();
    }
    dt_build_kdtree /= 10;

    // Covariance
    double dt_covariances = 0.0;
    const size_t k_correspondence_covariance = 10;
    auto kdtree = sycl_points::algorithms::knn_search::KDTree::build(queue, shared_points);
    for (size_t i = 0; i < 11; ++i) {
        s = std::chrono::high_resolution_clock::now();
        sycl_points::algorithms::covariance::compute_covariances_async(*kdtree, shared_points, k_correspondence_covariance).wait();
        if (i > 0) {
            dt_covariances +=
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                    .count();
        }
    }
    dt_covariances /= 10;

    // Downsampling
    double dt_voxel_downsampling = 0.0;
    const float voxel_size = 1.0;
    sycl_points::algorithms::filter::VoxelGrid voxel_grid(queue, voxel_size);
    for (size_t i = 0; i < 11; ++i) {
        s = std::chrono::high_resolution_clock::now();
        sycl_points::PointCloudShared downsampled(queue);
        voxel_grid.downsampling(shared_points, downsampled);
        if (i > 0) {
            dt_voxel_downsampling +=
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                    .count();
        }
    }
    dt_voxel_downsampling /= 10;

    // Transform
    double dt_transform_cpu_copy = 0.0;
    for (size_t i = 0; i < 11; ++i) {
        s = std::chrono::high_resolution_clock::now();
        auto tmp = sycl_points::algorithms::transform::transform_cpu_copy(shared_points, Eigen::Matrix4f::Identity());
        if (i > 0) {
            dt_transform_cpu_copy +=
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                    .count();
        }
    }
    dt_transform_cpu_copy /= 10;

    double dt_transform_cpu_inplace = 0.0;
    for (size_t i = 0; i < 11; ++i) {
        s = std::chrono::high_resolution_clock::now();
        sycl_points::algorithms::transform::transform_cpu(shared_points, Eigen::Matrix4f::Identity());
        if (i > 0) {
            dt_transform_cpu_inplace +=
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                    .count();
        }
    }
    dt_transform_cpu_inplace /= 10;

    double dt_transform_copy = 0.0;
    for (size_t i = 0; i < 11; ++i) {
        s = std::chrono::high_resolution_clock::now();
        auto ret = sycl_points::algorithms::transform::transform_cpu_copy(shared_points, Eigen::Matrix4f::Identity() * 2);
        if (i == 0) {
            // for (size_t j = 0; j < 10; ++j) {
            //   std::cout << "source: " << source_points.points[j].transpose() << std::endl;
            //   std::cout << "shared: " << (*shared_points.points)[j].transpose() << std::endl;
            //   std::cout << "ret: " << (*ret.points)[j].transpose() << std::endl;
            // }
        } else {
            dt_transform_copy +=
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                    .count();
        }
    }
    dt_transform_copy /= 10;

    double dt_transform_inplace = 0.0;
    for (size_t i = 0; i < 11; ++i) {
        s = std::chrono::high_resolution_clock::now();
        sycl_points::algorithms::transform::transform(shared_points, Eigen::Matrix4f::Identity());
        if (i > 0) {
            dt_transform_inplace +=
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                    .count();
        }
    }
    dt_transform_inplace /= 10;

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
        std::cout << "Build KDTree on cpu (shared_ptr): " << dt_build_kdtree << " us" << std::endl;
        std::cout << "Compute covariances on device (shared_ptr): " << dt_covariances << " us" << std::endl;
        std::cout << "Voxel downsampling (shared_ptr): " << dt_voxel_downsampling << " us" << std::endl;
        std::cout << "transform on cpu (shared_ptr, copy): " << dt_transform_cpu_copy << " us" << std::endl;
        std::cout << "transform on cpu (shared_ptr, zero copy): " << dt_transform_cpu_inplace << " us" << std::endl;
        std::cout << "transform on device (shared_ptr, copy): " << dt_transform_copy << " us" << std::endl;
        std::cout << "transform on device (shared_ptr, zero copy): " << dt_transform_inplace << " us" << std::endl;
    }

    return 0;
}
