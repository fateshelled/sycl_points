#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <string>
#include <sycl_points/io/point_cloud_writer.hpp>

namespace sycl_points {
namespace ros2 {

/// @brief Shared shutdown export settings for LO, LIO, and graph LIO nodes.
struct SubmapExport {
    bool save_on_exit = false;
    std::string output_path = "submap.pcd";
    bool binary = true;

    void declare_parameters(rclcpp::Node& node) {
        this->save_on_exit = node.declare_parameter<bool>("submap/save_on_exit", false);
        this->output_path = node.declare_parameter<std::string>("submap/output_path", "submap.pcd");
        this->binary = node.declare_parameter<bool>("submap/output_binary", true);
        if (this->save_on_exit) {
            auto extension = std::filesystem::path(this->output_path).extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension != ".pcd" && extension != ".ply") {
                throw std::invalid_argument("submap/output_path must have a .pcd or .ply extension");
            }
        }
    }

    /// @brief Save after processing callbacks have stopped, while the pipeline is still alive.
    template <typename Pipeline>
    void save(rclcpp::Node& node, const Pipeline& pipeline, const std::string& frame_id) const noexcept {
        if (!this->save_on_exit) return;
        try {
            PointCloudShared cloud(*pipeline.get_device_queue());
            pipeline.extract_full_submap_point_cloud(cloud);
            if (cloud.size() == 0) {
                RCLCPP_WARN(node.get_logger(), "Skipping submap save: internal map has no points to export");
                return;
            }
            const std::filesystem::path path(this->output_path);
            if (path.has_parent_path()) {
                std::filesystem::create_directories(path.parent_path());
            }
            PointCloudWriter::writeFile(path.string(), cloud, this->binary);
            RCLCPP_INFO(node.get_logger(), "Saved full retained submap: points=%zu frame=%s path=%s", cloud.size(),
                        frame_id.c_str(), this->output_path.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(node.get_logger(), "Failed to save submap: %s", e.what());
        } catch (...) {
            RCLCPP_ERROR(node.get_logger(), "Failed to save submap: unknown error");
        }
    }
};

}  // namespace ros2
}  // namespace sycl_points
