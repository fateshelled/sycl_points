#pragma once

#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
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
    rclcpp::CallbackGroup::SharedPtr cb_group_processing_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_ = nullptr;
    rclcpp::TimerBase::SharedPtr processing_timer_ = nullptr;

    std::mutex pending_mutex_;
    std::deque<sensor_msgs::msg::PointCloud2::UniquePtr> pending_point_clouds_;
    std::size_t max_pending_point_clouds_ = 3;
    std::size_t dropped_pending_point_clouds_ = 0;
    sensor_msgs::msg::PointCloud2::UniquePtr active_point_cloud_;
    ProcessedFrame active_frame_;

    void point_cloud_callback(sensor_msgs::msg::PointCloud2::UniquePtr msg);
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void processing_timer_callback();
};

}  // namespace ros2
}  // namespace sycl_points
