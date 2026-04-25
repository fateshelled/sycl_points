#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <map>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <sycl_points/pipeline/lidar_inertial_odometry.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <vector>

#include "sycl_points_ros2/lidar_odometry_base_node.hpp"

namespace sycl_points {
namespace ros2 {

/// @brief ROS 2 node for the LiDAR-Inertial Odometry pipeline.
///
/// IMU subscription is always enabled (LIO requires IMU).
/// Publishes the same topics as LiDAROdometryNode (odom, pose, TF,
/// preprocessed cloud, submap) so it can be used as a drop-in replacement.
class LidarInertialOdometryNode : public rclcpp::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit LidarInertialOdometryNode(const rclcpp::NodeOptions& options);
    ~LidarInertialOdometryNode() override;

private:
    // -----------------------------------------------------------------------
    // Pipeline
    // -----------------------------------------------------------------------
    std::unique_ptr<pipeline::lidar_inertial_odometry::LidarInertialOdometryPipeline> pipeline_;
    pipeline::lidar_inertial_odometry::Parameters params_;
    sycl_points::shared_vector_ptr<uint8_t> msg_data_buffer_;
    PointCloudShared::Ptr scan_pc_;

    // -----------------------------------------------------------------------
    // Subscriptions (separate callback groups for concurrent IMU feed)
    // -----------------------------------------------------------------------
    rclcpp::CallbackGroup::SharedPtr cb_group_lidar_;
    rclcpp::CallbackGroup::SharedPtr cb_group_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;

    // -----------------------------------------------------------------------
    // Publishers
    // -----------------------------------------------------------------------
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_preprocessed_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_submap_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // -----------------------------------------------------------------------
    // Node parameters (ROS2 / TF)
    // -----------------------------------------------------------------------
    std::string points_topic_ = "points";
    std::string imu_topic_ = "imu/data";
    std::string odom_frame_id_ = "odom";
    std::string base_link_id_ = "base_link";
    Eigen::Isometry3f T_base_link_to_lidar_ = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f T_lidar_to_base_link_ = Eigen::Isometry3f::Identity();
    bool input_convert_rgb_ = true;
    bool input_convert_intensity_ = true;

    LiDAROdometryBaseNode::SubscriptionQoSParams points_qos_params_{"keep_last", 1, "best_effort"};
    LiDAROdometryBaseNode::SubscriptionQoSParams imu_qos_params_{"keep_all", 100, "best_effort"};

    // -----------------------------------------------------------------------
    // Processing-time bookkeeping
    // -----------------------------------------------------------------------
    std::map<std::string, std::vector<double>> processing_times_;

    // -----------------------------------------------------------------------
    // Callbacks
    // -----------------------------------------------------------------------
    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::UniquePtr msg);
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

    // -----------------------------------------------------------------------
    // Helpers (same logic as LiDAROdometryBaseNode)
    // -----------------------------------------------------------------------
    nav_msgs::msg::Odometry make_odom_message(const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom) const;
    geometry_msgs::msg::PoseStamped make_pose_message(const std_msgs::msg::Header& header,
                                                      const Eigen::Isometry3f& odom) const;
    nav_msgs::msg::Odometry make_keyframe_pose_message(const std_msgs::msg::Header& header,
                                                       const Eigen::Isometry3f& odom) const;
    geometry_msgs::msg::TransformStamped make_transform_message(const std_msgs::msg::Header& header,
                                                                const Eigen::Isometry3f& odom) const;
    void record_processing_times(double dt_from_msg, double dt_publish);
    void print_processing_times(const std::string& name, double dt);
    void add_delta_time(const std::string& name, double dt);
    void log_processing_times();
};

}  // namespace ros2
}  // namespace sycl_points
