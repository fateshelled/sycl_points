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

    // Create separate MutuallyExclusive callback groups for LiDAR and IMU so that
    // IMU callbacks can run concurrently with point cloud processing when using
    // a MultiThreadedExecutor
    this->cb_group_lidar_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    this->cb_group_imu_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions lidar_sub_options;
    lidar_sub_options.callback_group = this->cb_group_lidar_;
    this->sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        this->points_topic_, rclcpp::QoS(10),
        std::bind(&LiDAROdometryNode::point_cloud_callback, this, std::placeholders::_1),
        lidar_sub_options);
    RCLCPP_INFO(this->get_logger(), "Subscribe PointCloud: %s", this->sub_pc_->get_topic_name());

    rclcpp::SubscriptionOptions imu_sub_options;
    imu_sub_options.callback_group = this->cb_group_imu_;
    this->sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
        this->imu_topic_, rclcpp::QoS(100),
        std::bind(&LiDAROdometryNode::imu_callback, this, std::placeholders::_1),
        imu_sub_options);
    RCLCPP_INFO(this->get_logger(), "Subscribe IMU: %s", this->sub_imu_->get_topic_name());
}

LiDAROdometryNode::~LiDAROdometryNode() = default;

void LiDAROdometryNode::point_cloud_callback(const sensor_msgs::msg::PointCloud2::UniquePtr msg) {
    const auto frame = this->process_point_cloud_message(*msg);
    if (frame.result == ResultType::success || frame.result == ResultType::first_frame) {
        this->publish_processed_frame(msg->header, frame);
    }
}

void LiDAROdometryNode::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(this->imu_buffer_mutex_);
    this->imu_buffer_.push_back(*msg);

    // Discard old entries, keeping only the last imu_buffer_duration_sec_ seconds
    const double latest_sec = rclcpp::Time(this->imu_buffer_.back().header.stamp).seconds();
    while (!this->imu_buffer_.empty()) {
        const double oldest_sec = rclcpp::Time(this->imu_buffer_.front().header.stamp).seconds();
        if (latest_sec - oldest_sec > imu_buffer_duration_sec_) {
            this->imu_buffer_.pop_front();
        } else {
            break;
        }
    }
}
}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LiDAROdometryNode)
