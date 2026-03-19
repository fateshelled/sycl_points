#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>

#include "sycl_points_ros2/lidar_odometry_base_node.hpp"

namespace sycl_points {
namespace ros2 {
class LiDAROdometryNode : public LiDAROdometryBaseNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LiDAROdometryNode(const rclcpp::NodeOptions& options);
    ~LiDAROdometryNode();

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_ = nullptr;

    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::UniquePtr msg);
};
}  // namespace ros2
}  // namespace sycl_points
