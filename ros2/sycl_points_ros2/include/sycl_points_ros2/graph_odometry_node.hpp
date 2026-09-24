#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "sycl_points_ros2/graph_odometry_base_node.hpp"

namespace sycl_points {
namespace ros2 {

class GraphOdometryNode : public GraphOdometryBaseNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit GraphOdometryNode(const rclcpp::NodeOptions& options);
    ~GraphOdometryNode() override = default;

private:
    rclcpp::CallbackGroup::SharedPtr cb_group_lidar_ = nullptr;
    rclcpp::CallbackGroup::SharedPtr cb_group_imu_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_ = nullptr;

    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::UniquePtr msg);
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
};

}  // namespace ros2
}  // namespace sycl_points
