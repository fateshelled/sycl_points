#pragma once

#include <cstdint>
#include <cmath>
#include <string>
#include <utility>

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/ros2/convert.hpp>
#include <sycl_points_msgs/msg/mapping_scan.hpp>

namespace sycl_points::ros2 {

/// Publish scan and LiDAR pose atomically; the mapper never has to synchronize
/// separate odometry and point cloud topics (whose odometry pose is for base_link).
class MappingScanPublisher {
public:
    void initialize(rclcpp::Node& node, const std::string& topic) {
        if (node.declare_parameter<bool>("mapping/output/enable", true)) {
            publisher_ = node.create_publisher<sycl_points_msgs::msg::MappingScan>(
                topic, rclcpp::QoS(10).reliable());
        }
    }

    bool has_subscribers() const {
        return publisher_ && publisher_->get_subscription_count() > 0;
    }

    void publish(const std_msgs::msg::Header& scan_header, const std::string& odom_frame,
                 const Eigen::Isometry3f& lidar_pose, const PointCloudShared& cloud, bool deskewed) {
        const uint64_t sequence = sequence_++;
        if (!has_subscribers()) return;

        auto converted = toROS2msg(cloud, scan_header);
        if (!converted) return;
        // The mapping path deliberately bypasses the frontend's filtering.
        // Do not advertise an unfiltered scan as dense if it contains bad points.
        for (const auto& point : *cloud.points) {
            if (!std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z())) {
                converted->is_dense = false;
                break;
            }
        }

        sycl_points_msgs::msg::MappingScan message;
        message.sequence = sequence;
        message.deskewed = deskewed;
        message.cloud = std::move(*converted);
        message.lidar_pose.header.stamp = scan_header.stamp;
        message.lidar_pose.header.frame_id = odom_frame;
        message.lidar_pose.pose.position.x = lidar_pose.translation().x();
        message.lidar_pose.pose.position.y = lidar_pose.translation().y();
        message.lidar_pose.pose.position.z = lidar_pose.translation().z();
        const Eigen::Quaternionf q(lidar_pose.rotation());
        message.lidar_pose.pose.orientation.x = q.x();
        message.lidar_pose.pose.orientation.y = q.y();
        message.lidar_pose.pose.orientation.z = q.z();
        message.lidar_pose.pose.orientation.w = q.w();
        publisher_->publish(std::move(message));
    }

private:
    rclcpp::Publisher<sycl_points_msgs::msg::MappingScan>::SharedPtr publisher_;
    uint64_t sequence_ = 0;
};

}  // namespace sycl_points::ros2
