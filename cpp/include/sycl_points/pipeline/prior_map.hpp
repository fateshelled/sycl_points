#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "sycl_points/algorithms/feature/covariance.hpp"
#include "sycl_points/algorithms/filter/voxel_downsampling.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/io/point_cloud_reader.hpp"

namespace sycl_points::pipeline {

struct PriorMapOptions {
    bool downsample = true;
    float voxel_size = 0.4f;
    float eigenvalue_floor = 1e-4f;
    size_t covariance_neighbors = 20;
    std::string covariance_mode = "AUTO";
    Eigen::Isometry3f file_pose = Eigen::Isometry3f::Identity();
    float local_radius = 60.0f;
    float prediction_margin = 3.0f;
    size_t local_min_points = 50;
};

// The cloud and its KD-tree form one immutable target generation.
struct LocalMapTarget {
    PointCloudShared::Ptr cloud;
    algorithms::knn::KDTree::Ptr tree;
    Eigen::Vector3f center;
    float radius = 0.0f;
};

class PriorMap {
public:
    PriorMap(const sycl_utils::DeviceQueue& queue, const std::string& path, const PriorMapOptions& options)
        : queue_(queue), options_(options) {
        if (path.empty() || (options.covariance_mode != "AUTO" && options.covariance_mode != "REESTIMATE") ||
            !std::isfinite(options.local_radius) || options.local_radius <= 0 ||
            !std::isfinite(options.prediction_margin) || options.prediction_margin < 0 ||
            !std::isfinite(options.voxel_size) || options.voxel_size <= 0 ||
            options.covariance_neighbors < 4 || options.covariance_neighbors > 100 || options.local_min_points < 1 ||
            !options.file_pose.matrix().allFinite()) {
            throw std::invalid_argument("Invalid prior map configuration");
        }
        bool preserve = false;
        {
            auto cpu = PointCloudReader::readFile(path, false, false);
            if (cpu.size() == 0) throw std::runtime_error("Prior map is empty");
            for (const auto& p : *cpu.points) {
                if (!p.head<3>().allFinite()) throw std::runtime_error("Non-finite prior map point");
            }
            preserve = options.covariance_mode == "AUTO" && cpu.has_cov();
            if (!preserve) cpu.covs->clear();
            map_ = std::make_shared<PointCloudShared>(queue, cpu);
        }
        auto& input = *map_;
        if (options.downsample) {
            algorithms::filter::VoxelGrid voxel(queue, options.voxel_size);
            voxel.set_covariance_eigenvalue_floor(options.eigenvalue_floor);
            voxel.downsampling(input, input);
            covariance_clamp_count_ += voxel.covariance_clamp_count();
        }
        if (input.size() == 0) throw std::runtime_error("Prior map is empty after downsampling");
        if (preserve && !options.downsample) {
            for (auto& cov : *input.covs) {
                cov = algorithms::filter::regularize_map_covariance(cov, options.eigenvalue_floor,
                                                                      &covariance_clamp_count_);
            }
        }
        if (!preserve) {
            if (input.size() < options.covariance_neighbors) {
                throw std::runtime_error("Too few map points for covariance estimation");
            }
            const auto tree = algorithms::knn::KDTree::build(queue, input);
            algorithms::covariance::estimate_async(*tree, input, options.covariance_neighbors).wait_and_throw();
            if (!input.has_cov()) throw std::runtime_error("Map covariance estimation failed");
            for (auto& cov : *input.covs) {
                cov = algorithms::filter::regularize_map_covariance(cov, options.eigenvalue_floor,
                                                                      &covariance_clamp_count_);
            }
        }
        const Eigen::Matrix3f rotation = options.file_pose.rotation();
        for (size_t i = 0; i < input.size(); ++i) {
            (*input.points)[i].head<3>() = options.file_pose * (*input.points)[i].head<3>();
            const Eigen::Matrix3f rotated = rotation * (*input.covs)[i].topLeftCorner<3, 3>() * rotation.transpose();
            (*input.covs)[i].setZero();
            (*input.covs)[i].topLeftCorner<3, 3>() = 0.5f * (rotated + rotated.transpose()).eval();
        }
        cell_size_ = options.local_radius;
        for (size_t i = 0; i < map_->size(); ++i) cells_[cell((*map_->points)[i].head<3>())].push_back(i);
    }

    size_t size() const { return map_->size(); }
    size_t covariance_clamp_count() const { return covariance_clamp_count_; }

    // required_radius contains the full predicted submap, correspondence distance and drift margin.
    std::shared_ptr<const LocalMapTarget> target(const Eigen::Vector3f& center, float required_radius) {
        if (!center.allFinite() || !std::isfinite(required_radius) || required_radius < 0 ||
            required_radius + options_.prediction_margin > options_.local_radius) return nullptr;
        if (active_ && (center - active_->center).norm() + required_radius + options_.prediction_margin <=
                           active_->radius) return active_;

        auto next = std::make_shared<LocalMapTarget>();
        next->center = center;
        next->radius = options_.local_radius;
        next->cloud = std::make_shared<PointCloudShared>(queue_);
        const auto min_cell = cell(center - Eigen::Vector3f::Constant(next->radius));
        const auto max_cell = cell(center + Eigen::Vector3f::Constant(next->radius));
        for (int64_t x = min_cell.x; x <= max_cell.x; ++x) {
            for (int64_t y = min_cell.y; y <= max_cell.y; ++y) {
                for (int64_t z = min_cell.z; z <= max_cell.z; ++z) {
                    const auto it = cells_.find({x, y, z});
                    if (it == cells_.end()) continue;
                    for (const size_t index : it->second) {
                        if (((*map_->points)[index].head<3>() - center).squaredNorm() > next->radius * next->radius)
                            continue;
                        next->cloud->points->push_back((*map_->points)[index]);
                        next->cloud->covs->push_back((*map_->covs)[index]);
                    }
                }
            }
        }
        if (next->cloud->size() < options_.local_min_points) return nullptr;
        next->tree = algorithms::knn::KDTree::build(queue_, *next->cloud);
        active_ = next;
        return active_;
    }

private:
    struct Cell {
        int64_t x, y, z;
        bool operator==(const Cell&) const = default;
    };
    struct CellHash {
        size_t operator()(const Cell& c) const {
            size_t h = std::hash<int64_t>{}(c.x);
            h ^= std::hash<int64_t>{}(c.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int64_t>{}(c.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    Cell cell(const Eigen::Vector3f& p) const {
        return {static_cast<int64_t>(std::floor(p.x() / cell_size_)),
                static_cast<int64_t>(std::floor(p.y() / cell_size_)),
                static_cast<int64_t>(std::floor(p.z() / cell_size_))};
    }
    sycl_utils::DeviceQueue queue_;
    PriorMapOptions options_;
    float cell_size_;
    size_t covariance_clamp_count_ = 0;
    PointCloudShared::Ptr map_;
    std::unordered_map<Cell, std::vector<size_t>, CellHash> cells_;
    std::shared_ptr<const LocalMapTarget> active_;
};

}  // namespace sycl_points::pipeline
