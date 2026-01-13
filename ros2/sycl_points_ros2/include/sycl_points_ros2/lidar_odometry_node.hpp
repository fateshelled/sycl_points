#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sycl_points/pipeline/lidar_odometry.hpp>
#include <sycl_points/pipeline/lidar_odometry_params.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

#include "sycl_points_ros2/covariance_marker_publisher.hpp"

namespace sycl_points {
namespace ros2 {
class LiDAROdometryNode : public rclcpp::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LiDAROdometryNode(const rclcpp::NodeOptions& options);
    ~LiDAROdometryNode();

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_ = nullptr;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_preprocessed_ = nullptr;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_submap_ = nullptr;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_ = nullptr;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose_ = nullptr;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_ = nullptr;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ = nullptr;

    std::unique_ptr<CovarianceMarkerPublisher> covariance_marker_publisher_ = nullptr;

    std::unique_ptr<sycl_points::pipeline::lidar_odometry::LiDAROdometryPipeline> pipeline_ = nullptr;

    sycl_points::shared_vector_ptr<uint8_t> msg_data_buffer_ = nullptr;
    PointCloudShared::Ptr scan_pc_ = nullptr;

    pipeline::lidar_odometry::Parameters params_;

    std::map<std::string, std::vector<double>> processing_times_;
    void add_delta_time(const std::string& name, double dt) {
        if (this->processing_times_.count(name) > 0) {
            this->processing_times_[name].push_back(dt);
        } else {
            this->processing_times_[name] = {dt};
        }
    }
    void print_processing_times(const std::string& name, double dt) {
        constexpr size_t LENGTH = 24;
        std::string log = name + ": ";
        for (size_t i = 0; i < LENGTH - name.length(); ++i) {
            log += " ";
        };
        log += "%9.2f us";
        RCLCPP_INFO(this->get_logger(), log.c_str(), dt);
    }

    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::UniquePtr msg);
    void publish_odom(const std_msgs::msg::Header& header,
                      const algorithms::registration::RegistrationResult& reg_result);
    void publish_keyframe_pose(const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom);
};
}  // namespace ros2
}  // namespace sycl_points
