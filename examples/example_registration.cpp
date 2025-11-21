#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/algorithms/knn/kdtree.hpp>
#include <sycl_points/algorithms/preprocess_filter.hpp>
#include <sycl_points/algorithms/registration.hpp>
#include <sycl_points/algorithms/voxel_downsampling.hpp>
#include <sycl_points/io/point_cloud_reader.hpp>

int main() {
    std::string source_filename = "../data/source.ply";
    std::string target_filename = "../data/target.ply";

    const sycl_points::PointCloudCPU source_points =
        sycl_points::PointCloudReader::readFile(source_filename, false, false);
    const sycl_points::PointCloudCPU target_points =
        sycl_points::PointCloudReader::readFile(target_filename, false, false);

    /* Specity device */
    const auto device_selector = sycl_points::sycl_utils::device_selector::default_selector_v;
    sycl::device dev(device_selector);  // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
    sycl_points::sycl_utils::DeviceQueue queue(dev);

    queue.print_device_info();

    const float voxel_size = 0.25f;
    const size_t num_neighbors = 10;

    sycl_points::algorithms::registration::RegistrationParams param;
    param.max_iterations = 10;
    param.max_correspondence_distance = 1.0f;
    param.optimization_method = sycl_points::algorithms::registration::OptimizationMethod::POWELL_DOGLEG;
    param.robust.type = sycl_points::algorithms::registration::RobustLossType::GEMAN_MCCLURE;
    param.robust.init_scale = 10.0f;
    param.robust.auto_scale = true;
    param.robust.scaling_factor = 0.5f;
    param.robust.scaling_iter = 3;

    const auto registration = std::make_shared<sycl_points::algorithms::registration::RegistrationGICP>(queue, param);
    const auto voxel_grid = std::make_shared<sycl_points::algorithms::filter::VoxelGrid>(queue, voxel_size);
    const auto preprocess_filter = std::make_shared<sycl_points::algorithms::filter::PreprocessFilter>(queue);

    const float BOX_FILTER_MIN_DISTANCE = 0.5f;
    const float BOX_FILTER_MAX_DISTANCE = 50.0f;
    const size_t LOOP = 100;
    const size_t WARM_UP = 10;
    std::map<std::string, double> elapsed;
    for (size_t i = 0; i < LOOP + WARM_UP; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();

        sycl_points::PointCloudShared source_shared(queue, source_points);
        sycl_points::PointCloudShared target_shared(queue, target_points);
        const auto dt_to_shared =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0)
                .count();

        t0 = std::chrono::high_resolution_clock::now();
        preprocess_filter->box_filter(source_shared, BOX_FILTER_MIN_DISTANCE, BOX_FILTER_MAX_DISTANCE);
        sycl_points::PointCloudShared source_downsampled(queue);
        voxel_grid->downsampling(source_shared, source_downsampled);

        preprocess_filter->box_filter(target_shared, BOX_FILTER_MIN_DISTANCE, BOX_FILTER_MAX_DISTANCE);
        sycl_points::PointCloudShared target_downsampled(queue);
        voxel_grid->downsampling(target_shared, target_downsampled);

        source_downsampled.reserve_covs(source_downsampled.size());
        target_downsampled.reserve_covs(target_downsampled.size());
        source_downsampled.reserve_normals(source_downsampled.size());
        target_downsampled.reserve_normals(target_downsampled.size());

        const auto dt_downsampled =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0)
                .count();

        t0 = std::chrono::high_resolution_clock::now();
        const auto source_tree = sycl_points::algorithms::knn::KDTree::build(queue, source_downsampled);
        const auto target_tree = sycl_points::algorithms::knn::KDTree::build(queue, target_downsampled);
        const auto dt_build_kdtree =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0)
                .count();

        t0 = std::chrono::high_resolution_clock::now();
        const auto source_neighbors = source_tree->knn_search(source_downsampled, num_neighbors);
        const auto target_neighbors = target_tree->knn_search(target_downsampled, num_neighbors);
        const auto dt_knn_search_for_covs =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0)
                .count();

        t0 = std::chrono::high_resolution_clock::now();
        sycl_points::algorithms::covariance::compute_covariances_async(source_neighbors, source_downsampled).wait();
        sycl_points::algorithms::covariance::compute_covariances_async(target_neighbors, target_downsampled).wait();
        const auto dt_covariance =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0)
                .count();

        t0 = std::chrono::high_resolution_clock::now();
        sycl_points::algorithms::covariance::compute_normals_async(source_neighbors, source_downsampled).wait();
        sycl_points::algorithms::covariance::compute_normals_async(target_neighbors, target_downsampled).wait();
        const auto dt_normal =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0)
                .count();

        /* GICP matching requires updating to planar covariance or normalized covariance */
        t0 = std::chrono::high_resolution_clock::now();
        // sycl_points::algorithms::covariance::covariance_update_plane(source_downsampled);
        // sycl_points::algorithms::covariance::covariance_update_plane(target_downsampled);
        sycl_points::algorithms::covariance::covariance_normalize(source_downsampled);
        sycl_points::algorithms::covariance::covariance_normalize(target_downsampled);
        const auto dt_udpate_covs =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0)
                .count();

        t0 = std::chrono::high_resolution_clock::now();
        /* NOTE:
            Random downsampling after covariance computation maintains GICP accuracy
            while reducing processing time */
        // preprocess_filter->random_sampling(source_downsampled, 1000);

        sycl_points::TransformMatrix init_T = sycl_points::TransformMatrix::Identity();
        const auto ret = registration->align(source_downsampled, target_downsampled, *target_tree, init_T);
        const auto dt_registration =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0)
                .count();

        if (i > WARM_UP) {
            if (elapsed.count("1. to PointCloudShared") == 0) elapsed["1. to PointCloudShared"] = 0.0;
            elapsed["1. to PointCloudShared"] += dt_to_shared;

            if (elapsed.count("2. Downsampling") == 0) elapsed["2. Downsampling"] = 0.0;
            elapsed["2. Downsampling"] += dt_downsampled;

            if (elapsed.count("3. KDTree build") == 0) elapsed["3. KDTree build"] = 0.0;
            elapsed["3. KDTree build"] += dt_build_kdtree;

            if (elapsed.count("4. KDTree kNN Search") == 0) elapsed["4. KDTree kNN Search"] = 0.0;
            elapsed["4. KDTree kNN Search"] += dt_knn_search_for_covs;

            if (elapsed.count("5. compute Covariances") == 0) elapsed["5. compute Covariances"] = 0.0;
            elapsed["5. compute Covariances"] += dt_covariance;

            if (elapsed.count("6. compute Normals") == 0) elapsed["6. compute Normals"] = 0.0;
            elapsed["6. compute Normals"] += dt_normal;

            if (elapsed.count("7. update Covariances") == 0) elapsed["7. update Covariances"] = 0.0;
            elapsed["7. update Covariances"] += dt_udpate_covs;

            if (elapsed.count("8. Registration") == 0) elapsed["8. Registration"] = 0.0;
            elapsed["8. Registration"] += dt_registration;
        }
        if (i == LOOP + WARM_UP - 1) {
            std::cout << ret.T.matrix() << std::endl;
        }
    }

    std::cout << std::endl;
    double total_elapsed = 0.0;
    for (auto [key, dt] : elapsed) {
        std::cout << std::setw(24) << std::fixed << key + ": ";
        std::cout << std::setw(7) << std::fixed << std::setprecision(2) << dt / LOOP << " us" << std::endl;
        total_elapsed += dt / LOOP;
    }
    std::cout << std::setw(24) << std::fixed << "TOTAL: ";
    std::cout << std::setw(7) << std::fixed << std::setprecision(2) << total_elapsed << " us" << std::endl << std::endl;
}
