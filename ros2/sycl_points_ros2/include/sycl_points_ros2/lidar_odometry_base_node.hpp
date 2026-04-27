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
#include <sycl_points/pipeline/lidar_odometry.hpp>
#include <sycl_points/pipeline/lidar_odometry_params.hpp>
#include <sycl_points/ros2/covariance_marker_publisher.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <vector>

namespace sycl_points {
namespace ros2 {

class LiDAROdometryBaseNode : public rclcpp::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct PublishOptions {
        bool publish_odom = true;
        bool publish_tf = true;
        bool publish_debug_clouds = true;
    };

    /// @brief QoS settings for a single subscription (history, depth, reliability)
    struct SubscriptionQoSParams {
        std::string history = "keep_last";        ///< "keep_last" or "keep_all"
        int64_t depth = 10;                       ///< queue depth (used only when history == "keep_last")
        std::string reliability = "best_effort";  ///< "best_effort" or "reliable"

        rclcpp::QoS to_qos() const {
            if (history == "keep_all") {
                rclcpp::QoS qos = rclcpp::QoS(rclcpp::KeepAll());
                if (reliability == "reliable") {
                    qos.reliable();
                } else {
                    qos.best_effort();
                }
                return qos;
            }
            rclcpp::QoS qos(static_cast<size_t>(depth));
            if (reliability == "reliable") {
                qos.reliable();
            } else {
                qos.best_effort();
            }
            return qos;
        }
    };

    using ResultType = pipeline::lidar_odometry::LiDAROdometryPipeline::ResultType;

    struct ProcessedFrame {
        ResultType result = ResultType::error;
        Eigen::Isometry3f odom = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f keyframe_pose = Eigen::Isometry3f::Identity();
        const algorithms::registration::RegistrationResult* registration_result = nullptr;
        double dt_from_ros2_msg = 0.0;
        std::map<std::string, double> pipeline_processing_times;
        double processing_subtotal = 0.0;
        double publish_time = 0.0;
    };

    LiDAROdometryBaseNode(const std::string& node_name, const rclcpp::NodeOptions& options);
    ~LiDAROdometryBaseNode() override;

protected:
    void initialize_processing();
    void initialize_publishers(const PublishOptions& options);
    ProcessedFrame process_point_cloud_message(const sensor_msgs::msg::PointCloud2& msg);
    void publish_processed_frame(const std_msgs::msg::Header& header, ProcessedFrame& frame);
    void record_processing_times(const ProcessedFrame& frame);

    nav_msgs::msg::Odometry make_odom_message(
        const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom,
        const algorithms::registration::RegistrationResult* reg_result = nullptr) const;
    geometry_msgs::msg::PoseStamped make_pose_message(const std_msgs::msg::Header& header,
                                                      const Eigen::Isometry3f& odom) const;
    nav_msgs::msg::Odometry make_keyframe_pose_message(const std_msgs::msg::Header& header,
                                                       const Eigen::Isometry3f& odom) const;
    geometry_msgs::msg::TransformStamped make_transform_message(const std_msgs::msg::Header& header,
                                                                const Eigen::Isometry3f& odom) const;

    bool publish_debug_clouds_enabled() const { return this->publish_options_.publish_debug_clouds; }
    bool publish_odom_enabled() const { return this->publish_options_.publish_odom; }
    bool publish_tf_enabled() const { return this->publish_options_.publish_tf; }

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

    std::string points_topic_ = "points";
    bool input_convert_rgb_ = true;
    bool input_convert_intensity_ = false;
    bool input_use_reflectivity_as_intensity = true;

    // ROS2/TF frame parameters
    std::string odom_frame_id_ = "odom";
    std::string base_link_id_ = "base_link";
    Eigen::Isometry3f T_base_link_to_lidar_ = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f T_lidar_to_base_link_ = Eigen::Isometry3f::Identity();

    // Visualization parameters
    ros2::CovarianceMarkerConfig scan_covariance_marker_config_;

    // IMU subscription topic name
    std::string imu_topic_ = "imu/data";

    // QoS settings for subscriptions
    // Points: keep_last(1) — only the latest scan is needed for real-time odometry
    // IMU: keep_all — all measurements between frames are required for preintegration
    SubscriptionQoSParams points_qos_params_{"keep_last", 1, "best_effort"};
    SubscriptionQoSParams imu_qos_params_{"keep_all", 100, "best_effort"};

private:
    void add_delta_time(const std::string& name, double dt);
    void print_processing_times(const std::string& name, double dt);
    void log_processing_times();

    PublishOptions publish_options_;
    bool processing_initialized_ = false;
    std::map<std::string, std::vector<double>> processing_times_;
};

}  // namespace ros2
}  // namespace sycl_points
