#include "sycl_points/point_cloud_reader.hpp"
#include "sycl_points/sycl_utils.hpp"

#include <iostream>
#include <chrono>

int main() {
  std::string source_filename = "../data/source.ply";

  using PCReader = sycl_points::PointCloudReader<float>;
  sycl_points::PointCloudCPU<float> source_points = PCReader::readFile(source_filename);

  /* Specity device */
  sycl::device dev;  // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
  sycl::queue queue(dev);

  sycl_points::sycl_utils::print_device_info(queue);

  {
    // make allocator
    auto s = std::chrono::high_resolution_clock::now();
    sycl_points::host_allocator<float> host_alloc(queue);
    sycl_points::shared_allocator<float> shared_alloc(queue);
    const auto dt_make_allocate = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();

    // copy to host container
    s = std::chrono::high_resolution_clock::now();
    sycl_points::PointContainerHost<float> host_points(source_points.size(), host_alloc);
    for (size_t i = 0; i < source_points.size(); ++i) {
      host_points[i] = source_points.points[i];
    }
    const auto dt_copy_to_host = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();

    // copy to shared container
    s = std::chrono::high_resolution_clock::now();
    sycl_points::PointContainerShared<float> shared_points(source_points.size(), shared_alloc);
    for (size_t i = 0; i < source_points.size(); ++i) {
      shared_points[i] = source_points.points[i];
    }
    const auto dt_copy_to_shared = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();

    // Transform
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.block(0, 0, 3, 3) = Eigen::AngleAxisf(0.5 * M_PI, Eigen::Vector3f(0, 1, 0)).matrix();  // rotate 90 deg, y axis
    trans.block(0, 3, 3, 1) << 1.0, 2.0, 3.0;
    s = std::chrono::high_resolution_clock::now();
    const auto transformed_points = source_points.transform_copy(trans);  // GT
    const auto dt_transform_cpu = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();

    // transform on device (host to host)
    double dt_transform_on_device_host_to_host = 0.0;
    sycl_points::PointContainerHost<float> host_results(0, host_alloc);
    for (size_t _ = 0; _ < 11; ++_) {
      host_results.clear();
      s = std::chrono::high_resolution_clock::now();
      host_results.resize(source_points.size());
      auto source_ptr = host_points.data();
      auto result_ptr = host_results.data();
      auto trans_ptr = sycl::malloc_shared<Eigen::Matrix4f>(1, queue);
      trans_ptr[0] = trans;

      queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(source_points.size()), [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          auto t = trans_ptr[0];
          result_ptr[i][0] = t(0, 0) * source_ptr[i][0] + t(0, 1) * source_ptr[i][1] + t(0, 2) * source_ptr[i][2] + t(0, 3);
          result_ptr[i][1] = t(1, 0) * source_ptr[i][0] + t(1, 1) * source_ptr[i][1] + t(1, 2) * source_ptr[i][2] + t(1, 3);
          result_ptr[i][2] = t(2, 0) * source_ptr[i][0] + t(2, 1) * source_ptr[i][1] + t(2, 2) * source_ptr[i][2] + t(2, 3);
          result_ptr[i][3] = 1.0f;
        });
      });
      queue.wait();
      sycl::free(trans_ptr, queue);

      if (_ > 0) {
        dt_transform_on_device_host_to_host += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();
      }
    }
    dt_transform_on_device_host_to_host /= 10;

    // validate
    for (size_t i = 0; i < transformed_points.size(); ++i) {
      const auto delta = transformed_points.points[i] - host_results[i];
      if (delta.norm() > 1e-5) {
        std::cout << "ERROR: [" << i << "]" << std::endl;
        std::cout << transformed_points.points[i].transpose() << std::endl;
        std::cout << host_results[i].transpose() << std::endl;
        break;
      }
    }


    // transform on device (shared to shared)
    double dt_transform_on_device_shared_to_shared = 0.0;
    sycl_points::PointContainerShared<float> shared_results(0, shared_alloc);;
    for (size_t _ = 0; _ < 11; ++_) {
      shared_results.clear();
      s = std::chrono::high_resolution_clock::now();
      shared_results.resize(source_points.size());
      auto source_ptr = shared_points.data();
      auto result_ptr = shared_results.data();
      auto trans_ptr = sycl::malloc_shared<Eigen::Matrix4f>(1, queue);
      trans_ptr[0] = trans;

      queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(source_points.size()), [=](sycl::id<1> idx) {
          const size_t i = idx[0];
          auto t = trans_ptr[0];
          result_ptr[i][0] = t(0, 0) * source_ptr[i][0] + t(0, 1) * source_ptr[i][1] + t(0, 2) * source_ptr[i][2] + t(0, 3);
          result_ptr[i][1] = t(1, 0) * source_ptr[i][0] + t(1, 1) * source_ptr[i][1] + t(1, 2) * source_ptr[i][2] + t(1, 3);
          result_ptr[i][2] = t(2, 0) * source_ptr[i][0] + t(2, 1) * source_ptr[i][1] + t(2, 2) * source_ptr[i][2] + t(2, 3);
          result_ptr[i][3] = 1.0f;
        });
      });
      queue.wait();
      sycl::free(trans_ptr, queue);

      if (_ > 0) {
        dt_transform_on_device_shared_to_shared += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s).count();
      }
    }
    dt_transform_on_device_shared_to_shared /= 10;

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

    std::cout << "make allocator: " << dt_make_allocate << " us" << std::endl;
    std::cout << "copy to host container: " << dt_copy_to_host << " us" << std::endl;
    std::cout << "copy to shared container: " << dt_copy_to_shared << " us" << std::endl;
    std::cout << "transform on cpu: " << dt_transform_cpu << " us" << std::endl;
    std::cout << "transform on device(host_ptr): " << dt_transform_on_device_host_to_host << " us" << std::endl;
    std::cout << "transform on device(shared_ptr): " << dt_transform_on_device_shared_to_shared << " us" << std::endl;
  }

  return 0;
}
