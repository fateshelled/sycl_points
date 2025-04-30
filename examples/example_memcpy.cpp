#include <chrono>
#include <iostream>
#include <sycl_points/io/point_cloud_reader.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

int main() {
    std::string source_filename = "../data/source.ply";

    sycl_points::PointCloudCPU source_points = sycl_points::PointCloudReader::readFile(source_filename);

    /* Specity device */
    sycl::device dev;  // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
    sycl::queue queue(dev);

    sycl_points::sycl_utils::print_device_info(queue);

    {
        // make allocator
        auto s = std::chrono::high_resolution_clock::now();
        sycl_points::PointAllocatorShared shared_alloc(queue);
        const auto dt_make_allocate =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                .count();

        // copy from cpu to shared container
        s = std::chrono::high_resolution_clock::now();
        sycl_points::PointContainerShared shared_points(source_points.size(), shared_alloc);
        for (size_t i = 0; i < source_points.size(); ++i) {
            shared_points[i] = (*source_points.points)[i];
        }
        const auto dt_copy_to_shared =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                .count();

        // memcpy from cpu to shared container
        shared_points.clear();
        s = std::chrono::high_resolution_clock::now();
        shared_points.resize(source_points.size());
        queue
            .memcpy(shared_points.data(), source_points.points->data(),
                    source_points.size() * sizeof(sycl_points::PointType))
            .wait();
        const auto dt_memcpy_to_shared =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                .count();

        // copy from cpu to device ptr
        sycl_points::PointType* device_ptr;
        s = std::chrono::high_resolution_clock::now();
        device_ptr = sycl::malloc_device<sycl_points::PointType>(source_points.size(), queue);
        queue.memcpy(device_ptr, source_points.points->data(), source_points.size() * sizeof(sycl_points::PointType))
            .wait();
        const auto dt_memcpy_to_device =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                .count();
        sycl::free(device_ptr, queue);

        // copy from device ptr to shared container
        s = std::chrono::high_resolution_clock::now();
        queue.memcpy(shared_points.data(), device_ptr, shared_points.size() * sizeof(sycl_points::PointType)).wait();
        const auto dt_memcpy_from_device_to_shared =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                .count();

        sycl::free(device_ptr, queue);

        // Transform
        Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
        trans.block(0, 0, 3, 3) =
            Eigen::AngleAxisf(0.5 * M_PI, Eigen::Vector3f(0, 1, 0)).matrix();  // rotate 90 deg, y axis
        trans.block(0, 3, 3, 1) << 1.0, 2.0, 3.0;
        s = std::chrono::high_resolution_clock::now();
        const auto transformed_points = source_points.transform_copy(trans);  // GT
        const auto dt_transform_cpu =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                .count();

        auto tmp = source_points;
        s = std::chrono::high_resolution_clock::now();
        tmp.transform(trans);
        const auto dt_transform_cpu_zerocopy =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                .count();

        // transform on device (shared to shared)
        double dt_transform_on_device_shared_to_shared = 0.0;
        sycl_points::PointContainerShared shared_results(0, shared_alloc);
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
                    result_ptr[i][0] =
                        t(0, 0) * source_ptr[i][0] + t(0, 1) * source_ptr[i][1] + t(0, 2) * source_ptr[i][2] + t(0, 3);
                    result_ptr[i][1] =
                        t(1, 0) * source_ptr[i][0] + t(1, 1) * source_ptr[i][1] + t(1, 2) * source_ptr[i][2] + t(1, 3);
                    result_ptr[i][2] =
                        t(2, 0) * source_ptr[i][0] + t(2, 1) * source_ptr[i][1] + t(2, 2) * source_ptr[i][2] + t(2, 3);
                    result_ptr[i][3] = 1.0f;
                });
            });
            queue.wait();
            sycl::free(trans_ptr, queue);

            if (_ > 0) {
                dt_transform_on_device_shared_to_shared +=
                    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - s)
                        .count();
            }
        }
        dt_transform_on_device_shared_to_shared /= 10;

        std::cout << "make allocator: " << dt_make_allocate << " us" << std::endl;
        std::cout << "copy from cpu to shared container: " << dt_copy_to_shared << " us" << std::endl;
        std::cout << "memcpy from cpu to device ptr: " << dt_memcpy_to_device << " us" << std::endl;
        std::cout << "memcpy from device ptr to shared container: " << dt_memcpy_from_device_to_shared << " us"
                  << std::endl;

        std::cout << "memcpy cpu to shared: " << dt_memcpy_to_shared << " us" << std::endl;

        std::cout << "transform on cpu (copy): " << dt_transform_cpu << " us" << std::endl;
        std::cout << "transform on cpu (zero copy): " << dt_transform_cpu_zerocopy << " us" << std::endl;
        std::cout << "transform on device(shared_ptr): " << dt_transform_on_device_shared_to_shared << " us"
                  << std::endl;
        // std::cout << "transform on device(device_ptr): " << dt_transform_on_device_device_ptr << " us" << std::endl;
    }

    return 0;
}
