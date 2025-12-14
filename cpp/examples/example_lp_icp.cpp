/**
 * @file example_lp_icp.cpp
 * @brief Example demonstrating LP-ICP (Localizability-aware Point-to-Plane ICP)
 *
 * This example shows how to use the LP-ICP registration algorithm that
 * detects and handles degenerate cases (tunnels, corridors, etc.) through
 * localizability analysis.
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <sycl_points/algorithms/covariance.hpp>
#include <sycl_points/algorithms/knn/kdtree.hpp>
#include <sycl_points/algorithms/registration_lp_icp.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sp = sycl_points;
namespace reg = sp::algorithms::registration;
namespace loc = sp::algorithms::localizability;

/// @brief Generate a tunnel-like point cloud (degenerate case)
sp::PointCloudShared::Ptr generateTunnelPointCloud(const sp::sycl_utils::DeviceQueue& queue, size_t num_points,
                                                    float radius, float length) {
    auto cloud = std::make_shared<sp::PointCloudShared>(queue);
    cloud->resize(num_points);
    cloud->allocate_normals();

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> length_dist(0.0f, length);

    auto* points = cloud->points_ptr();
    auto* normals = cloud->normals_ptr();

    for (size_t i = 0; i < num_points; ++i) {
        float theta = angle_dist(gen);
        float z = length_dist(gen);

        // Tunnel wall points
        float x = radius * std::cos(theta);
        float y = radius * std::sin(theta);

        points[i] = sp::PointType(x, y, z, 1.0f);

        // Normal points inward
        normals[i] = sp::Normal(-std::cos(theta), -std::sin(theta), 0.0f, 0.0f);
    }

    return cloud;
}

/// @brief Generate a room-like point cloud (well-constrained case)
sp::PointCloudShared::Ptr generateRoomPointCloud(const sp::sycl_utils::DeviceQueue& queue, size_t num_points,
                                                   float size) {
    auto cloud = std::make_shared<sp::PointCloudShared>(queue);
    cloud->resize(num_points);
    cloud->allocate_normals();

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> pos_dist(-size / 2.0f, size / 2.0f);
    std::uniform_int_distribution<int> wall_dist(0, 5);

    auto* points = cloud->points_ptr();
    auto* normals = cloud->normals_ptr();

    for (size_t i = 0; i < num_points; ++i) {
        int wall = wall_dist(gen);
        float u = pos_dist(gen);
        float v = pos_dist(gen);

        float x, y, z;
        float nx, ny, nz;

        switch (wall) {
            case 0:  // +X wall
                x = size / 2.0f;
                y = u;
                z = v;
                nx = -1.0f;
                ny = 0.0f;
                nz = 0.0f;
                break;
            case 1:  // -X wall
                x = -size / 2.0f;
                y = u;
                z = v;
                nx = 1.0f;
                ny = 0.0f;
                nz = 0.0f;
                break;
            case 2:  // +Y wall
                x = u;
                y = size / 2.0f;
                z = v;
                nx = 0.0f;
                ny = -1.0f;
                nz = 0.0f;
                break;
            case 3:  // -Y wall
                x = u;
                y = -size / 2.0f;
                z = v;
                nx = 0.0f;
                ny = 1.0f;
                nz = 0.0f;
                break;
            case 4:  // +Z wall (ceiling)
                x = u;
                y = v;
                z = size / 2.0f;
                nx = 0.0f;
                ny = 0.0f;
                nz = -1.0f;
                break;
            case 5:  // -Z wall (floor)
                x = u;
                y = v;
                z = -size / 2.0f;
                nx = 0.0f;
                ny = 0.0f;
                nz = 1.0f;
                break;
        }

        points[i] = sp::PointType(x, y, z, 1.0f);
        normals[i] = sp::Normal(nx, ny, nz, 0.0f);
    }

    return cloud;
}

/// @brief Apply transformation to point cloud
sp::PointCloudShared::Ptr transformCloud(const sp::PointCloudShared& cloud, const Eigen::Isometry3f& T) {
    auto result = std::make_shared<sp::PointCloudShared>(cloud);

    auto* points = result->points_ptr();
    auto* normals = result->has_normal() ? result->normals_ptr() : nullptr;

    const Eigen::Matrix3f R = T.rotation();
    const Eigen::Vector3f t = T.translation();

    for (size_t i = 0; i < result->size(); ++i) {
        Eigen::Vector3f p = points[i].head<3>();
        p = R * p + t;
        points[i] = sp::PointType(p.x(), p.y(), p.z(), 1.0f);

        if (normals) {
            Eigen::Vector3f n = normals[i].head<3>();
            n = R * n;
            normals[i] = sp::Normal(n.x(), n.y(), n.z(), 0.0f);
        }
    }

    return result;
}

void runRegistration(const std::string& scenario_name, sp::PointCloudShared::Ptr source,
                      sp::PointCloudShared::Ptr target, const Eigen::Isometry3f& ground_truth,
                      const sp::sycl_utils::DeviceQueue& queue) {
    std::cout << "\n=== " << scenario_name << " ===" << std::endl;

    // Build KD-tree for target
    sp::algorithms::knn::KDTree kdtree(queue, *target);

    // Create initial guess (identity or with some perturbation)
    Eigen::Isometry3f initial_guess = Eigen::Isometry3f::Identity();

    // Setup LP-ICP parameters
    reg::LPRegistrationParams params;
    params.max_iterations = 30;
    params.max_correspondence_distance = 1.0f;
    params.enable_localizability = true;
    params.verbose = true;
    params.localizability_params.verbose = true;

    // Create LP-ICP registration
    reg::LPRegistration lp_icp(queue, params);

    // Perform registration
    auto start = std::chrono::high_resolution_clock::now();
    auto result = lp_icp.align(*source, *target, kdtree, initial_guess.matrix());
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Compute error
    Eigen::Isometry3f estimated = Eigen::Isometry3f(result.T);
    Eigen::Isometry3f error = ground_truth.inverse() * estimated;
    float translation_error = error.translation().norm();
    Eigen::AngleAxisf aa(error.rotation());
    float rotation_error = std::abs(aa.angle()) * 180.0f / M_PI;

    // Print results
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Converged: " << (result.converged ? "Yes" : "No") << std::endl;
    std::cout << "  Iterations: " << result.iterations << std::endl;
    std::cout << "  Time: " << time_ms << " ms" << std::endl;
    std::cout << "  Translation error: " << translation_error << " m" << std::endl;
    std::cout << "  Rotation error: " << rotation_error << " deg" << std::endl;

    if (result.has_localizability) {
        std::cout << "\nLocalizability Analysis:" << std::endl;
        std::cout << "  Directions: [rx, ry, rz, tx, ty, tz]" << std::endl;
        std::cout << "  L_f: " << result.localizability.aggregate.L_f.transpose() << std::endl;
        std::cout << "  L_u: " << result.localizability.aggregate.L_u.transpose() << std::endl;
        std::cout << "  Categories: ";
        const char* dir_names[] = {"rx", "ry", "rz", "tx", "ty", "tz"};
        for (size_t j = 0; j < 6; ++j) {
            std::cout << dir_names[j] << "=";
            switch (result.localizability.aggregate.categories[j]) {
                case loc::LocalizabilityCategory::FULL:
                    std::cout << "Full";
                    break;
                case loc::LocalizabilityCategory::PARTIAL:
                    std::cout << "Partial";
                    break;
                case loc::LocalizabilityCategory::NONE:
                    std::cout << "None";
                    break;
            }
            if (j < 5) std::cout << ", ";
        }
        std::cout << std::endl;

        std::cout << "  Soft constraints: " << result.localizability.soft_constraints.size() << std::endl;
        std::cout << "  Hard constraints: " << result.localizability.hard_constraint.num_constraints << std::endl;
    }
}

int main() {
    std::cout << "LP-ICP Registration Example" << std::endl;
    std::cout << "==========================" << std::endl;

    // Create SYCL queue
    auto queue = sp::sycl_utils::create_device_queue_auto();
    sp::sycl_utils::print_device_info(queue);

    const size_t num_points = 5000;

    // Scenario 1: Room (well-constrained)
    {
        auto target = generateRoomPointCloud(queue, num_points, 10.0f);

        // Create source with known transformation
        Eigen::Isometry3f ground_truth = Eigen::Isometry3f::Identity();
        ground_truth.translate(Eigen::Vector3f(0.5f, 0.3f, 0.2f));
        ground_truth.rotate(Eigen::AngleAxisf(0.1f, Eigen::Vector3f::UnitZ()));

        auto source = transformCloud(*target, ground_truth.inverse());

        runRegistration("Room (Well-Constrained)", source, target, ground_truth, queue);
    }

    // Scenario 2: Tunnel (degenerate along Z-axis)
    {
        auto target = generateTunnelPointCloud(queue, num_points, 2.0f, 20.0f);

        // Create source with translation along tunnel axis (degenerate direction)
        Eigen::Isometry3f ground_truth = Eigen::Isometry3f::Identity();
        ground_truth.translate(Eigen::Vector3f(0.1f, 0.1f, 1.0f));  // Large Z translation

        auto source = transformCloud(*target, ground_truth.inverse());

        runRegistration("Tunnel (Degenerate along Z)", source, target, ground_truth, queue);
    }

    std::cout << "\nDone!" << std::endl;
    return 0;
}
