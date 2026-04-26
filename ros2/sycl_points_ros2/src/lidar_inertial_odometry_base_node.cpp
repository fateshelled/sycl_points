#include "sycl_points_ros2/lidar_inertial_odometry_base_node.hpp"

#include <sycl_points/ros2/convert.hpp>
#include <sycl_points/utils/time_utils.hpp>

#include "sycl_points_ros2/declare_lidar_inertial_odometry_params.hpp"

namespace sycl_points {
namespace ros2 {

LidarInertialOdometryBaseNode::LidarInertialOdometryBaseNode(const std::string& node_name,
                                                             const rclcpp::NodeOptions& options)
    : rclcpp::Node(node_name, options) {}

LidarInertialOdometryBaseNode::~LidarInertialOdometryBaseNode() {
    if (!this->processing_initialized_) {
        return;
    }
    this->log_processing_times();
}

void LidarInertialOdometryBaseNode::initialize_processing() {
    this->params_ = ros2::declare_lidar_inertial_odometry_parameters(this);

    this->points_topic_ = this->declare_parameter<std::string>("points_topic", this->points_topic_);
    this->imu_topic_ = this->declare_parameter<std::string>("imu_topic", this->imu_topic_);
    this->input_convert_rgb_ = this->declare_parameter<bool>("input/convert_rgb", true);
    this->input_convert_intensity_ = this->declare_parameter<bool>("input/convert_intensity", true);

    // QoS settings
    this->points_qos_params_.history =
        this->declare_parameter<std::string>("points_qos/history", this->points_qos_params_.history);
    this->points_qos_params_.depth =
        this->declare_parameter<int64_t>("points_qos/depth", this->points_qos_params_.depth);
    this->points_qos_params_.reliability =
        this->declare_parameter<std::string>("points_qos/reliability", this->points_qos_params_.reliability);
    this->imu_qos_params_.history =
        this->declare_parameter<std::string>("imu_qos/history", this->imu_qos_params_.history);
    this->imu_qos_params_.depth = this->declare_parameter<int64_t>("imu_qos/depth", this->imu_qos_params_.depth);
    this->imu_qos_params_.reliability =
        this->declare_parameter<std::string>("imu_qos/reliability", this->imu_qos_params_.reliability);

    this->odom_frame_id_ = this->declare_parameter<std::string>("odom_frame_id", this->odom_frame_id_);
    this->base_link_id_ = this->declare_parameter<std::string>("base_link_id", this->base_link_id_);
    {
        const auto x = this->declare_parameter<double>("T_base_link_to_lidar/x", 0.0);
        const auto y = this->declare_parameter<double>("T_base_link_to_lidar/y", 0.0);
        const auto z = this->declare_parameter<double>("T_base_link_to_lidar/z", 0.0);
        const auto qx = this->declare_parameter<double>("T_base_link_to_lidar/qx", 0.0);
        const auto qy = this->declare_parameter<double>("T_base_link_to_lidar/qy", 0.0);
        const auto qz = this->declare_parameter<double>("T_base_link_to_lidar/qz", 0.0);
        const auto qw = this->declare_parameter<double>("T_base_link_to_lidar/qw", 1.0);
        this->T_base_link_to_lidar_.setIdentity();
        this->T_base_link_to_lidar_.translation() << static_cast<float>(x), static_cast<float>(y),
            static_cast<float>(z);
        const Eigen::Quaternionf q(static_cast<float>(qw), static_cast<float>(qx), static_cast<float>(qy),
                                   static_cast<float>(qz));
        this->T_base_link_to_lidar_.matrix().block<3, 3>(0, 0) = q.normalized().matrix();
        this->T_lidar_to_base_link_ = this->T_base_link_to_lidar_.inverse();
    }
    {
        const auto x = this->declare_parameter<double>("initial_base_link_pose/x", 0.0);
        const auto y = this->declare_parameter<double>("initial_base_link_pose/y", 0.0);
        const auto z = this->declare_parameter<double>("initial_base_link_pose/z", 0.0);
        const auto qx = this->declare_parameter<double>("initial_base_link_pose/qx", 0.0);
        const auto qy = this->declare_parameter<double>("initial_base_link_pose/qy", 0.0);
        const auto qz = this->declare_parameter<double>("initial_base_link_pose/qz", 0.0);
        const auto qw = this->declare_parameter<double>("initial_base_link_pose/qw", 1.0);
        Eigen::Isometry3f init = Eigen::Isometry3f::Identity();
        init.translation() << static_cast<float>(x), static_cast<float>(y), static_cast<float>(z);
        const Eigen::Quaternionf q(static_cast<float>(qw), static_cast<float>(qx), static_cast<float>(qy),
                                   static_cast<float>(qz));
        init.matrix().block<3, 3>(0, 0) = q.normalized().matrix();
        this->params_.pose.initial = init * this->T_base_link_to_lidar_;
    }

    this->pipeline_ = std::make_unique<pipeline::lidar_inertial_odometry::LidarInertialOdometryPipeline>(this->params_);
    this->pipeline_->get_device_queue()->print_device_info();

    this->msg_data_buffer_.reset(new shared_vector<uint8_t>(*this->pipeline_->get_device_queue()->ptr));
    this->scan_pc_.reset(new PointCloudShared(*this->pipeline_->get_device_queue()));
    this->processing_initialized_ = true;

    RCLCPP_INFO(this->get_logger(), "Input conversion - RGB: %s, intensity: %s",
                this->input_convert_rgb_ ? "enabled" : "disabled",
                this->input_convert_intensity_ ? "enabled" : "disabled");
}

void LidarInertialOdometryBaseNode::initialize_publishers(const PublishOptions& options) {
    this->publish_options_ = options;

    if (options.publish_debug_clouds) {
        this->pub_preprocessed_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/preprocessed", rclcpp::QoS(5));
        this->pub_submap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/submap", rclcpp::QoS(5));
    }

    if (options.publish_odom) {
        this->pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/odom", rclcpp::QoS(5));
        this->pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("sycl_lo/pose", rclcpp::QoS(5));
        this->pub_keyframe_pose_ =
            this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/keyframe/pose", rclcpp::QoS(5));
    }

    if (options.publish_tf) {
        this->tf_broadcaster_ =
            std::make_unique<tf2_ros::TransformBroadcaster>(*this, tf2_ros::DynamicBroadcasterQoS(1000));
    }

    if (this->pub_preprocessed_ != nullptr) {
        RCLCPP_INFO(this->get_logger(), "Publish Preprocessed PointCloud: %s",
                    this->pub_preprocessed_->get_topic_name());
    }
    if (this->pub_submap_ != nullptr) {
        RCLCPP_INFO(this->get_logger(), "Publish Submap PointCloud: %s", this->pub_submap_->get_topic_name());
    }
    if (this->pub_odom_ != nullptr) {
        RCLCPP_INFO(this->get_logger(), "Publish Odometry: %s", this->pub_odom_->get_topic_name());
    }
    if (this->pub_pose_ != nullptr) {
        RCLCPP_INFO(this->get_logger(), "Publish Pose: %s", this->pub_pose_->get_topic_name());
    }
    if (this->pub_keyframe_pose_ != nullptr) {
        RCLCPP_INFO(this->get_logger(), "Publish Keyframe Pose: %s", this->pub_keyframe_pose_->get_topic_name());
    }
}

LidarInertialOdometryBaseNode::ProcessedFrame LidarInertialOdometryBaseNode::process_point_cloud_message(
    const sensor_msgs::msg::PointCloud2& msg) {
    ProcessedFrame frame;
    const double timestamp = rclcpp::Time(msg.header.stamp).seconds();
    bool converted = false;

    double dt_from_ros2_msg = 0.0;
    time_utils::measure_execution(
        [&]() {
            converted = fromROS2msg(*this->pipeline_->get_device_queue(), msg, this->scan_pc_, this->msg_data_buffer_,
                                    this->input_convert_rgb_, this->input_convert_intensity_);
        },
        dt_from_ros2_msg);

    if (!converted || this->scan_pc_ == nullptr) {
        RCLCPP_WARN(this->get_logger(), "failed to convert input point cloud");
        frame.result = ResultType::error;
        return frame;
    }

    if (this->scan_pc_->size() == 0) {
        RCLCPP_WARN(this->get_logger(), "input point cloud is empty");
        frame.result = ResultType::error;
        return frame;
    }

    frame.result = this->pipeline_->process(this->scan_pc_, timestamp);
    if (frame.result >= ResultType::error) {
        RCLCPP_WARN(this->get_logger(), "LIO failed: %s", this->pipeline_->get_error_message().c_str());
        return frame;
    }

    frame.odom = this->pipeline_->get_odom();
    frame.keyframe_pose = this->pipeline_->get_last_keyframe_pose();

    frame.dt_from_ros2_msg = dt_from_ros2_msg;
    frame.pipeline_processing_times = this->pipeline_->get_current_processing_time();
    frame.processing_subtotal = dt_from_ros2_msg;
    for (const auto& item : frame.pipeline_processing_times) {
        frame.processing_subtotal += item.second;
    }

    return frame;
}

void LidarInertialOdometryBaseNode::publish_processed_frame(const std_msgs::msg::Header& header,
                                                            ProcessedFrame& frame) {
    frame.publish_time = 0.0;
    time_utils::measure_execution(
        [&]() {
            if (this->publish_tf_enabled() && this->tf_broadcaster_ != nullptr) {
                this->tf_broadcaster_->sendTransform(this->make_transform_message(header, frame.odom));
            }

            if (this->publish_odom_enabled()) {
                if (this->pub_odom_ != nullptr) {
                    this->pub_odom_->publish(this->make_odom_message(header, frame.odom));
                }
                if (this->pub_pose_ != nullptr) {
                    this->pub_pose_->publish(this->make_pose_message(header, frame.odom));
                }
                if (this->pub_keyframe_pose_ != nullptr) {
                    this->pub_keyframe_pose_->publish(this->make_keyframe_pose_message(header, frame.keyframe_pose));
                }
            }

            if (!this->publish_debug_clouds_enabled()) {
                return;
            }

            if (this->pub_preprocessed_ != nullptr && this->pub_preprocessed_->get_subscription_count() > 0) {
                const auto pc_msg = toROS2msg(this->pipeline_->get_preprocessed_point_cloud(), header);
                if (pc_msg != nullptr) {
                    this->pub_preprocessed_->publish(*pc_msg);
                }
            }

            if (this->pub_submap_ != nullptr && this->pub_submap_->get_subscription_count() > 0) {
                auto submap_msg = toROS2msg(this->pipeline_->get_submap_point_cloud(), header);
                if (submap_msg != nullptr) {
                    submap_msg->header.frame_id = this->odom_frame_id_;
                    this->pub_submap_->publish(*submap_msg);
                }
            }
        },
        frame.publish_time);
}

// ---------------------------------------------------------------------------
// Message helpers (same logic as LidarInertialOdometryNode)
// ---------------------------------------------------------------------------

nav_msgs::msg::Odometry LidarInertialOdometryBaseNode::make_odom_message(const std_msgs::msg::Header& header,
                                                                         const Eigen::Isometry3f& odom) const {
    const Eigen::Isometry3f T = odom * this->T_lidar_to_base_link_;
    const Eigen::Quaternionf q(T.rotation());

    nav_msgs::msg::Odometry msg;
    msg.header.stamp = header.stamp;
    msg.header.frame_id = this->odom_frame_id_;
    msg.child_frame_id = this->base_link_id_;
    msg.pose.pose.position.x = T.translation().x();
    msg.pose.pose.position.y = T.translation().y();
    msg.pose.pose.position.z = T.translation().z();
    msg.pose.pose.orientation.x = q.x();
    msg.pose.pose.orientation.y = q.y();
    msg.pose.pose.orientation.z = q.z();
    msg.pose.pose.orientation.w = q.w();
    return msg;
}

geometry_msgs::msg::PoseStamped LidarInertialOdometryBaseNode::make_pose_message(const std_msgs::msg::Header& header,
                                                                                 const Eigen::Isometry3f& odom) const {
    const auto odom_msg = this->make_odom_message(header, odom);
    geometry_msgs::msg::PoseStamped pose;
    pose.header = odom_msg.header;
    pose.pose = odom_msg.pose.pose;
    return pose;
}

nav_msgs::msg::Odometry LidarInertialOdometryBaseNode::make_keyframe_pose_message(const std_msgs::msg::Header& header,
                                                                                  const Eigen::Isometry3f& odom) const {
    return this->make_odom_message(header, odom);
}

geometry_msgs::msg::TransformStamped LidarInertialOdometryBaseNode::make_transform_message(
    const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom) const {
    const Eigen::Isometry3f T = odom * this->T_lidar_to_base_link_;
    const Eigen::Quaternionf q(T.rotation());

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = header.stamp;
    tf.header.frame_id = this->odom_frame_id_;
    tf.child_frame_id = this->base_link_id_;
    tf.transform.translation.x = T.translation().x();
    tf.transform.translation.y = T.translation().y();
    tf.transform.translation.z = T.translation().z();
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();
    return tf;
}

void LidarInertialOdometryBaseNode::record_processing_times(const ProcessedFrame& frame) {
    const double total_time = frame.processing_subtotal + frame.publish_time;

    this->add_delta_time("0. from ROS 2 msg", frame.dt_from_ros2_msg);
    for (const auto& [process_name, time] : frame.pipeline_processing_times) {
        this->add_delta_time(process_name, time);
    }
    this->add_delta_time("5. publish ROS 2 msg", frame.publish_time);
    this->add_delta_time("6. total", total_time);

    this->print_processing_times("0. from ROS 2 msg", frame.dt_from_ros2_msg);
    for (const auto& [process_name, time] : frame.pipeline_processing_times) {
        this->print_processing_times(process_name, time);
    }
    this->print_processing_times("5. publish ROS 2 msg", frame.publish_time);
    this->print_processing_times("6. total", total_time);
    RCLCPP_INFO(this->get_logger(), "");
}

void LidarInertialOdometryBaseNode::add_delta_time(const std::string& name, double dt) {
    if (this->processing_times_.count(name) > 0) {
        this->processing_times_[name].push_back(dt);
    } else {
        this->processing_times_[name] = {dt};
    }
}

void LidarInertialOdometryBaseNode::print_processing_times(const std::string& name, double dt) {
    constexpr size_t LENGTH = 24;
    std::string log = name + ": ";
    if (name.length() < LENGTH) {
        log += std::string(LENGTH - name.length(), ' ');
    }
    log += "%9.2f us";
    RCLCPP_INFO(this->get_logger(), log.c_str(), dt);
}

void LidarInertialOdometryBaseNode::log_processing_times() {
    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MAX processing time");

    this->processing_times_.insert(this->pipeline_->get_total_processing_times().begin(),
                                   this->pipeline_->get_total_processing_times().end());

    for (auto& item : this->processing_times_) {
        if (item.second.empty()) continue;
        const double max = *std::max_element(item.second.begin(), item.second.end());
        this->print_processing_times(item.first, max);
    }

    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MEAN processing time");
    for (auto& item : this->processing_times_) {
        if (item.second.empty()) continue;
        const double avg =
            std::accumulate(item.second.begin(), item.second.end(), 0.0) / static_cast<double>(item.second.size());
        this->print_processing_times(item.first, avg);
    }

    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MEDIAN processing time");
    for (auto& item : this->processing_times_) {
        if (item.second.empty()) continue;
        std::sort(item.second.begin(), item.second.end());
        const double median = item.second[item.second.size() / 2];
        this->print_processing_times(item.first, median);
    }
    RCLCPP_INFO(this->get_logger(), "");
}

}  // namespace ros2
}  // namespace sycl_points
