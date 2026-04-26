#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "sycl_points_ros2/lidar_inertial_odometry_base_node.hpp"

namespace sycl_points {
namespace ros2 {

class LidarInertialOdometryNode : public LidarInertialOdometryBaseNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit LidarInertialOdometryNode(const rclcpp::NodeOptions& options);
    ~LidarInertialOdometryNode() override = default;

private:
    // -----------------------------------------------------------------------
    // Subscriptions (separate callback groups for concurrent IMU feed)
    // -----------------------------------------------------------------------
    rclcpp::CallbackGroup::SharedPtr cb_group_lidar_;
    rclcpp::CallbackGroup::SharedPtr cb_group_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;

    // -----------------------------------------------------------------------
    // Callbacks
    // -----------------------------------------------------------------------
    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::UniquePtr msg);
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
};

}  // namespace ros2
}  // namespace sycl_points
