#include "sycl_points_ros2/lidar_inertial_odometry_node.hpp"

#include <rclcpp_components/register_node_macro.hpp>

namespace sycl_points {
namespace ros2 {

LidarInertialOdometryNode::LidarInertialOdometryNode(const rclcpp::NodeOptions& options)
    : LidarInertialOdometryBaseNode("lidar_inertial_odometry", options) {
    this->initialize_processing();
    this->initialize_publishers({});

    // -----------------------------------------------------------------------
    // Subscriptions
    // -----------------------------------------------------------------------
    // Create separate MutuallyExclusive callback groups for LiDAR and IMU so that
    // IMU callbacks can run concurrently with point cloud processing when using
    // a MultiThreadedExecutor
    cb_group_lidar_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cb_group_imu_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions lidar_opts;
    lidar_opts.callback_group = cb_group_lidar_;
    sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        points_topic_, points_qos_params_.to_qos(),
        std::bind(&LidarInertialOdometryNode::point_cloud_callback, this, std::placeholders::_1), lidar_opts);
    RCLCPP_INFO(this->get_logger(), "Subscribe PointCloud: %s (history=%s, depth=%ld, reliability=%s)",
                sub_pc_->get_topic_name(), points_qos_params_.history.c_str(), points_qos_params_.depth,
                points_qos_params_.reliability.c_str());

    rclcpp::SubscriptionOptions imu_opts;
    imu_opts.callback_group = cb_group_imu_;
    sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, imu_qos_params_.to_qos(),
        std::bind(&LidarInertialOdometryNode::imu_callback, this, std::placeholders::_1), imu_opts);
    RCLCPP_INFO(this->get_logger(), "Subscribe IMU: %s (history=%s, depth=%ld, reliability=%s)",
                sub_imu_->get_topic_name(), imu_qos_params_.history.c_str(), imu_qos_params_.depth,
                imu_qos_params_.reliability.c_str());
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void LidarInertialOdometryNode::point_cloud_callback(const sensor_msgs::msg::PointCloud2::UniquePtr msg) {
    auto frame = this->process_point_cloud_message(*msg);
    if (frame.result == ResultType::success || frame.result == ResultType::first_frame) {
        this->publish_processed_frame(msg->header, frame);
    }
    this->record_processing_times(frame);
}

void LidarInertialOdometryNode::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    imu::IMUMeasurement meas;
    meas.timestamp = rclcpp::Time(msg->header.stamp).seconds();
    meas.gyro =
        Eigen::Vector3f(static_cast<float>(msg->angular_velocity.x), static_cast<float>(msg->angular_velocity.y),
                        static_cast<float>(msg->angular_velocity.z));
    meas.accel =
        Eigen::Vector3f(static_cast<float>(msg->linear_acceleration.x), static_cast<float>(msg->linear_acceleration.y),
                        static_cast<float>(msg->linear_acceleration.z));
    pipeline_->add_imu_measurement(meas);
}

}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LidarInertialOdometryNode)
