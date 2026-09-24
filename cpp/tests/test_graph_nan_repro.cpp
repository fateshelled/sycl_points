#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <random>

#ifndef EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_VECTORIZE
#endif

#include <sycl/sycl.hpp>

#include "sycl_points/algorithms/graph/graph_optimization.hpp"
#include "sycl_points/algorithms/graph/graph_solver.hpp"
#include "sycl_points/algorithms/graph/sliding_window.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/pipeline/lidar_odometry_params.hpp"
#include "sycl_points/pipeline/submapping.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/eigen_utils.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace {

using namespace sycl_points;
using namespace sycl_points::algorithms;

sycl_utils::DeviceQueue make_queue() {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    return sycl_utils::DeviceQueue(device);
}

PointCloudShared::Ptr make_cube_cloud(const sycl_utils::DeviceQueue& queue, size_t n, float half,
                                      std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-half, half);
    PointCloudCPU cpu;
    cpu.points->resize(n);
    for (size_t i = 0; i < n; ++i) {
        (*cpu.points)[i] = PointType(dist(gen), dist(gen), dist(gen), 1.0f);
    }
    return std::make_shared<PointCloudShared>(queue, cpu);
}

PointCloudShared::Ptr transform_cloud(const sycl_utils::DeviceQueue& queue, const PointCloudShared& src,
                                      const Eigen::Isometry3f& T, std::mt19937& gen) {
    std::normal_distribution<float> noise(0.0f, 0.002f);
    PointCloudCPU cpu;
    cpu.points->resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        Eigen::Vector4f p = T.matrix() * src.points->at(i);
        (*cpu.points)[i] = PointType(p.x() + noise(gen), p.y() + noise(gen), p.z() + noise(gen), 1.0f);
    }
    return std::make_shared<PointCloudShared>(queue, cpu);
}

// Mirrors the pipeline's reg_params_ construction (graph/factor/* + graph/robust/*).
registration::RegistrationParams make_graph_reg_params() {
    registration::RegistrationFactorParams fp;
    fp.reg_type = registration::RegType::GICP;
    fp.max_correspondence_distance = 2.0f;
    fp.verbose = false;
    fp.rotation_constraint.enable = true;   // graph_odometry.yaml: true
    fp.rotation_constraint.weight = 1.0f;
    fp.rotation_constraint.robust.default_scale = 5.0f;
    registration::RegistrationParams params(fp);

    // pipeline: reg_params_.robust.type = graph.robust_type (HUBER), default_scale = 10
    params.robust.type = robust::RobustLossType::HUBER;
    params.robust.default_scale = 10.0f;
    return params;
}

PointCloudShared::Ptr make_planar_world(const sycl_utils::DeviceQueue& queue, std::mt19937& gen) {
    // Ground plane + two walls: real LiDAR-like planar structure (rank-1 covs).
    std::uniform_real_distribution<float> ux(-10.0f, 10.0f);
    std::uniform_real_distribution<float> uz(0.0f, 3.0f);
    std::normal_distribution<float> noise(0.0f, 0.005f);
    const size_t n_ground = 1500, n_wall = 750;
    PointCloudCPU cpu;
    cpu.points->resize(n_ground + 2 * n_wall);
    size_t i = 0;
    for (size_t j = 0; j < n_ground; ++j, ++i) {
        (*cpu.points)[i] = PointType(ux(gen), ux(gen), noise(gen), 1.0f);
    }
    for (size_t j = 0; j < n_wall; ++j, ++i) {  // wall at x = +5
        (*cpu.points)[i] = PointType(5.0f + noise(gen), ux(gen), uz(gen), 1.0f);
    }
    for (size_t j = 0; j < n_wall; ++j, ++i) {  // wall at y = +5
        (*cpu.points)[i] = PointType(ux(gen), 5.0f + noise(gen), uz(gen), 1.0f);
    }
    return std::make_shared<PointCloudShared>(queue, cpu);
}

Eigen::Isometry3f advance(const Eigen::Isometry3f& base, const Eigen::Matrix<float, 6, 1>& d) {
    Eigen::Isometry3f out(base);
    out.matrix() = base.matrix() * eigen_utils::lie::se3_exp(d);
    return out;
}

TEST(GraphNaNRepro, PipelineLikeConfig) {
    auto queue = make_queue();
    std::mt19937 gen(123);

    // Solver params mirroring graph_odometry.yaml (robust ladder enabled).
    graph::GraphSolverParams solver_params;
    solver_params.max_iterations = 8;
    solver_params.convergence_translation = 1e-4f;
    solver_params.convergence_rotation = 1e-4f;
    solver_params.relinearize_translation_thresh = 0.05f;
    solver_params.relinearize_rotation_thresh = 0.02f;
    solver_params.marginalization_lambda = 1e-3f;
    solver_params.robust.enable = true;
    solver_params.robust.init_scale = 10.0f;
    solver_params.robust.min_scale = 1.25f;
    solver_params.robust.levels = 3;
    solver_params.robust.iters_per_level = 2;
    solver_params.robust.relinearize_per_rung = false;

    graph::GraphOptimization::Options gopts;
    gopts.gate.enabled = true;
    gopts.gate.min_translation = 0.05f;  // small => frequent promotion (stress submap updates)
    gopts.gate.min_rotation = 0.05f;
    gopts.gate.min_time_seconds = 0.0f;
    gopts.relative_pose.sigma_rotation = 5e-3f;
    gopts.relative_pose.sigma_translation = 2e-2f;

    graph::GraphOptimization opt(queue, solver_params, 5, gopts);
    const auto reg_params = make_graph_reg_params();

    // Submap mirroring the pipeline (VOXEL_HASH_MAP).
    pipeline::lidar_odometry::Parameters submap_params;
    submap_params.submap.map_type = pipeline::odometry::SubmapMapType::VOXEL_HASH_MAP;
    submap_params.submap.voxel_size = 0.4f;
    submap_params.submap.max_distance_range = 50.0f;
    submap_params.submap.point_random_sampling_num = 512;
    submap_params.submap.weighted_sampling_ratio = 0.8f;
    submap_params.submap.keyframe.inlier_ratio_threshold = 0.0f;
    submap_params.submap.keyframe.distance_threshold = 0.05f;
    submap_params.submap.keyframe.angle_threshold_degrees = 3.0f;
    submap_params.submap.keyframe.time_threshold_seconds = 0.0f;
    submap_params.covariance_estimation.neighbor_num = 15;
    submap_params.registration.factor.reg_type = registration::RegType::GICP;
    submap_params.registration.factor.rotation_constraint.enable = true;
    pipeline::submapping::Submap submap(queue, submap_params);

    const size_t n_points = 3000;
    auto world = make_planar_world(queue, gen);

    Eigen::Matrix<float, 6, 1> delta = Eigen::Matrix<float, 6, 1>::Zero();
    delta.head<3>() = Eigen::Vector3f(0.0f, 0.0f, 0.01f);
    delta.tail<3>() = Eigen::Vector3f(0.05f, 0.0f, 0.0f);

    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    bool first = true;
    std::shared_ptr<PointCloudShared> submap_cloud;
    std::shared_ptr<const knn::KNNBase> submap_knn;

    for (int k = 0; k < 24; ++k) {
        const Eigen::Isometry3f T_prev = T;
        T = advance(T, delta);

        auto scan = transform_cloud(queue, *world, T.inverse(), gen);
        auto scan_knn = knn::KDTree::build(queue, *scan);
        covariance::estimate_async(*scan_knn, *scan, 15).wait_and_throw();

        if (first) {
            submap.add_first_frame(*scan, 0.1 * k, T);
            submap_cloud = std::make_shared<PointCloudShared>(submap.get_submap_point_cloud());
            submap_knn = knn::KDTree::build(queue, *submap_cloud);
            first = false;
            continue;
        }

        const Eigen::Isometry3f init_T = advance(T_prev, delta);
        auto fr = opt.process_frame(scan, submap_cloud, submap_knn, scan_knn, init_T, 0.1 * k, reg_params);

        // Submap update (pipeline analog: add_frame with the solved pose).
        algorithms::registration::RegistrationResult reg_result;
        reg_result.T = fr.current_pose;
        reg_result.converged = fr.converged;
        reg_result.iterations = fr.iterations;
        reg_result.error = fr.error;
        const bool changed = submap.add_frame(*scan, reg_result, 1.0f, 0.1 * k, nullptr);
        if (changed) {
            submap_cloud = std::make_shared<PointCloudShared>(submap.get_submap_point_cloud());
            submap_knn = knn::KDTree::build(queue, *submap_cloud);
        }

        const bool err_bad = !std::isfinite(fr.error);
        const auto& prior = opt.window().prior();
        std::printf(
            "frame %2d: error=%12.4g iters=%2zu conv=%d kf=%d win=%2zu SUBMAP_UPDATED=%d prior_valid=%d "
            "prior_Htr=%g prior_bnorm=%g prior_finite=%d%s\n",  //
            k, fr.error, fr.iterations, fr.converged ? 1 : 0, fr.keyframe ? 1 : 0, opt.window().window_size(),
            changed ? 1 : 0, prior.is_valid() ? 1 : 0,
            prior.is_valid() ? static_cast<double>(prior.H_prior.trace()) : 0.0,
            prior.is_valid() ? static_cast<double>(prior.b_prior.norm()) : 0.0,
            prior.is_valid() ? (prior.H_prior.allFinite() && prior.b_prior.allFinite() ? 1 : 0) : 1,
            err_bad ? "   <-- NON-FINITE" : "");

        if (err_bad) {
            // Per-factor breakdown: which factor carries the NaN?
            for (const auto& f : opt.window().factors()) {
                auto lin = f->get_linearization(queue, solver_params.relinearize_rotation_thresh,
                                                solver_params.relinearize_translation_thresh, 0.0f);
                const auto [sid, tid] = f->node_ids();
                std::printf("    factor (%llu, %llu): error=%.6g inlier=%u H00trace=%.6g b0norm=%.6g\n",  //
                            (unsigned long long)sid, (unsigned long long)tid, lin.error, lin.inlier,
                            lin.H00.trace(), lin.b0.norm());
            }
            // Node pose sanity.
            for (const auto& n : opt.window().active_nodes()) {
                const Eigen::Vector3f t = n->pose.translation();
                std::printf("    node %llu pose=(%.4g, %.4g, %.4g) finite=%d\n",  //
                            (unsigned long long)n->id, t.x(), t.y(), t.z(),
                            n->pose.matrix().allFinite() ? 1 : 0);
            }
            break;
        }
    }
}

}  // namespace
