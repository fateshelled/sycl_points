#include "sycl_points_ros2/lidar_odometry_node.hpp"

#include <rclcpp_components/register_node_macro.hpp>

namespace sycl_points {
namespace ros2 {

/// @brief constructor
/// @param options node option
LiDAROdometryNode::LiDAROdometryNode(const rclcpp::NodeOptions& options)
    : LiDAROdometryBaseNode("lidar_odometry", options) {
    this->initialize_processing();
    this->initialize_publishers({});

    this->sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        this->points_topic_, rclcpp::QoS(10), std::bind(&LiDAROdometryNode::point_cloud_callback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribe PointCloud: %s", this->sub_pc_->get_topic_name());
}

LiDAROdometryNode::~LiDAROdometryNode() = default;

void LiDAROdometryNode::point_cloud_callback(const sensor_msgs::msg::PointCloud2::UniquePtr msg) {
    const auto frame = this->process_point_cloud_message(*msg);
    if (frame.result == ResultType::success) {
        this->publish_processed_frame(msg->header, frame);
    }
}
}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LiDAROdometryNode)
