#include "sycl_points_ros2/lidar_odometry_base_node.hpp"

#include <algorithm>
#include <numeric>
#include <sycl_points/ros2/convert.hpp>
#include <sycl_points/utils/time_utils.hpp>

#include "sycl_points_ros2/declare_lidar_odometry_params.hpp"

namespace sycl_points {
namespace ros2 {

LiDAROdometryBaseNode::LiDAROdometryBaseNode(const std::string& node_name, const rclcpp::NodeOptions& options)
    : rclcpp::Node(node_name, options) {}

LiDAROdometryBaseNode::~LiDAROdometryBaseNode() {
    if (!this->processing_initialized_) {
        return;
    }
    this->log_processing_times();
}

void LiDAROdometryBaseNode::initialize_processing() {
    this->params_ = ros2::declare_lidar_odometry_parameters(this);

    this->points_topic_ = this->declare_parameter<std::string>("points_topic", this->points_topic_);
    this->imu_topic_ = this->declare_parameter<std::string>("imu_topic", this->imu_topic_);
    this->input_convert_rgb_ = this->declare_parameter<bool>("input/convert_rgb", true);
    this->input_convert_intensity_ = this->declare_parameter<bool>("input/convert_intensity", true);

    // tf and pose (ROS2/TF specific)
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
        const Eigen::Quaternionf quat(static_cast<float>(qw), static_cast<float>(qx), static_cast<float>(qy),
                                      static_cast<float>(qz));
        this->T_base_link_to_lidar_.matrix().block<3, 3>(0, 0) = quat.normalized().matrix();
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
        Eigen::Isometry3f initial_base_link = Eigen::Isometry3f::Identity();
        initial_base_link.translation() << static_cast<float>(x), static_cast<float>(y), static_cast<float>(z);
        const Eigen::Quaternionf quat(static_cast<float>(qw), static_cast<float>(qx), static_cast<float>(qy),
                                      static_cast<float>(qz));
        initial_base_link.matrix().block<3, 3>(0, 0) = quat.normalized().matrix();
        this->params_.pose.initial = initial_base_link * this->T_base_link_to_lidar_;
    }

    // Visualization (ROS2 specific)
    {
        auto& c = this->scan_covariance_marker_config_;
        c.topic_name = this->declare_parameter<std::string>("vis/covariance_markers/scan/topic_name", c.topic_name);
        c.marker_ns = this->declare_parameter<std::string>("vis/covariance_markers/scan/marker_ns", c.marker_ns);
        c.scale_factor = static_cast<float>(
            this->declare_parameter<double>("vis/covariance_markers/scan/scale_factor", c.scale_factor));
        c.min_scale =
            static_cast<float>(this->declare_parameter<double>("vis/covariance_markers/scan/min_scale", c.min_scale));
        c.max_scale =
            static_cast<float>(this->declare_parameter<double>("vis/covariance_markers/scan/max_scale", c.max_scale));
        c.alpha = static_cast<float>(this->declare_parameter<double>("vis/covariance_markers/scan/alpha", c.alpha));
        c.color_by_planarity =
            this->declare_parameter<bool>("vis/covariance_markers/scan/color_by_planarity", c.color_by_planarity);
        c.default_r =
            static_cast<float>(this->declare_parameter<double>("vis/covariance_markers/scan/default_r", c.default_r));
        c.default_g =
            static_cast<float>(this->declare_parameter<double>("vis/covariance_markers/scan/default_g", c.default_g));
        c.default_b =
            static_cast<float>(this->declare_parameter<double>("vis/covariance_markers/scan/default_b", c.default_b));
    }

    this->pipeline_ = std::make_unique<pipeline::lidar_odometry::LiDAROdometryPipeline>(this->params_);
    this->pipeline_->get_device_queue()->print_device_info();

    this->msg_data_buffer_.reset(new shared_vector<uint8_t>(*this->pipeline_->get_device_queue()->ptr));
    this->scan_pc_.reset(new PointCloudShared(*this->pipeline_->get_device_queue()));
    this->processing_initialized_ = true;

    RCLCPP_INFO(this->get_logger(), "Input conversion - RGB: %s, intensity: %s",
                this->input_convert_rgb_ ? "enabled" : "disabled",
                this->input_convert_intensity_ ? "enabled" : "disabled");
}

void LiDAROdometryBaseNode::initialize_publishers(const PublishOptions& options) {
    this->publish_options_ = options;

    if (options.publish_debug_clouds) {
        this->pub_preprocessed_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/preprocessed", rclcpp::QoS(5));
        this->pub_submap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/submap", rclcpp::QoS(5));
        this->covariance_marker_publisher_ =
            std::make_unique<CovarianceMarkerPublisher>(*this, this->scan_covariance_marker_config_);
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
    if (this->covariance_marker_publisher_ != nullptr) {
        RCLCPP_INFO(this->get_logger(), "Publish Covariance Markers: %s",
                    this->covariance_marker_publisher_->get_topic_name());
    }
}

LiDAROdometryBaseNode::ProcessedFrame LiDAROdometryBaseNode::process_point_cloud_message(
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
        RCLCPP_WARN(this->get_logger(), "lidar odometry failed: %s", this->pipeline_->get_error_message().c_str());
        return frame;
    }

    frame.odom = this->pipeline_->get_odom();
    frame.keyframe_pose = this->pipeline_->get_last_keyframe_pose();
    if (frame.result == ResultType::success) {
        frame.registration_result = &this->pipeline_->get_registration_result();
    }

    frame.dt_from_ros2_msg = dt_from_ros2_msg;
    frame.pipeline_processing_times = this->pipeline_->get_current_processing_time();
    frame.processing_subtotal = dt_from_ros2_msg;
    for (const auto& item : frame.pipeline_processing_times) {
        frame.processing_subtotal += item.second;
    }

    return frame;
}

void LiDAROdometryBaseNode::publish_processed_frame(const std_msgs::msg::Header& header, ProcessedFrame& frame) {
    frame.publish_time = 0.0;
    time_utils::measure_execution(
        [&]() {
            if (this->publish_tf_enabled() && this->tf_broadcaster_ != nullptr) {
                auto tf_msg = this->make_transform_message(header, frame.odom);
                this->tf_broadcaster_->sendTransform(tf_msg);
            }

            if (this->publish_odom_enabled()) {
                auto odom_msg = this->make_odom_message(header, frame.odom, frame.registration_result);
                auto pose_msg = this->make_pose_message(header, frame.odom);
                auto keyframe_msg = this->make_keyframe_pose_message(header, frame.keyframe_pose);

                if (this->pub_odom_ != nullptr) {
                    this->pub_odom_->publish(odom_msg);
                }
                if (this->pub_pose_ != nullptr) {
                    this->pub_pose_->publish(pose_msg);
                }
                if (this->pub_keyframe_pose_ != nullptr) {
                    this->pub_keyframe_pose_->publish(keyframe_msg);
                }
            }

            if (!this->publish_debug_clouds_enabled()) {
                return;
            }

            if (this->pub_preprocessed_ != nullptr && this->pub_preprocessed_->get_subscription_count() > 0) {
                const auto preprocessed_msg = toROS2msg(this->pipeline_->get_preprocessed_point_cloud(), header);
                if (preprocessed_msg != nullptr) {
                    this->pub_preprocessed_->publish(*preprocessed_msg);
                }
            }

            if (this->covariance_marker_publisher_ != nullptr) {
                if (const auto* registration_input_pc = this->pipeline_->get_registration_input_point_cloud();
                    registration_input_pc != nullptr) {
                    this->covariance_marker_publisher_->publish_if_subscribed(header, *registration_input_pc);
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

nav_msgs::msg::Odometry LiDAROdometryBaseNode::make_odom_message(
    const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom,
    const algorithms::registration::RegistrationResult* reg_result) const {
    const Eigen::Isometry3f odom_to_base_link = odom * this->T_lidar_to_base_link_;
    const auto odom_trans = odom_to_base_link.translation();
    const Eigen::Quaternionf odom_quat(odom_to_base_link.rotation());

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = header.stamp;
    odom_msg.header.frame_id = this->odom_frame_id_;
    odom_msg.child_frame_id = this->base_link_id_;
    odom_msg.pose.pose.position.x = odom_trans.x();
    odom_msg.pose.pose.position.y = odom_trans.y();
    odom_msg.pose.pose.position.z = odom_trans.z();
    odom_msg.pose.pose.orientation.x = odom_quat.x();
    odom_msg.pose.pose.orientation.y = odom_quat.y();
    odom_msg.pose.pose.orientation.z = odom_quat.z();
    odom_msg.pose.pose.orientation.w = odom_quat.w();

    if (reg_result != nullptr) {
        const Eigen::Matrix<float, 6, 6> cov_reg = reg_result->H.inverse();
        Eigen::Matrix<float, 6, 6> cov_odom;
        cov_odom.block<3, 3>(0, 0) = cov_reg.block<3, 3>(3, 3);
        cov_odom.block<3, 3>(3, 3) = cov_reg.block<3, 3>(0, 0);
        cov_odom.block<3, 3>(0, 3) = cov_reg.block<3, 3>(3, 0);
        cov_odom.block<3, 3>(3, 0) = cov_reg.block<3, 3>(0, 3);

        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(odom_msg.pose.covariance.data()) =
            cov_odom.cast<double>();
    }

    return odom_msg;
}

geometry_msgs::msg::PoseStamped LiDAROdometryBaseNode::make_pose_message(const std_msgs::msg::Header& header,
                                                                         const Eigen::Isometry3f& odom) const {
    const auto odom_msg = this->make_odom_message(header, odom);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = odom_msg.header;
    pose.pose = odom_msg.pose.pose;
    return pose;
}

nav_msgs::msg::Odometry LiDAROdometryBaseNode::make_keyframe_pose_message(const std_msgs::msg::Header& header,
                                                                          const Eigen::Isometry3f& odom) const {
    const Eigen::Isometry3f odom_to_base_link = odom * this->T_lidar_to_base_link_;
    const auto odom_trans = odom_to_base_link.translation();
    const Eigen::Quaternionf odom_quat(odom_to_base_link.rotation());

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header = header;
    odom_msg.header.frame_id = this->odom_frame_id_;
    odom_msg.child_frame_id = this->base_link_id_;
    odom_msg.pose.pose.position.x = odom_trans.x();
    odom_msg.pose.pose.position.y = odom_trans.y();
    odom_msg.pose.pose.position.z = odom_trans.z();
    odom_msg.pose.pose.orientation.x = odom_quat.x();
    odom_msg.pose.pose.orientation.y = odom_quat.y();
    odom_msg.pose.pose.orientation.z = odom_quat.z();
    odom_msg.pose.pose.orientation.w = odom_quat.w();
    return odom_msg;
}

geometry_msgs::msg::TransformStamped LiDAROdometryBaseNode::make_transform_message(
    const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom) const {
    const Eigen::Isometry3f odom_to_base_link = odom * this->T_lidar_to_base_link_;
    const auto odom_trans = odom_to_base_link.translation();
    const Eigen::Quaternionf odom_quat(odom_to_base_link.rotation());

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = header.stamp;
    tf.header.frame_id = this->odom_frame_id_;
    tf.child_frame_id = this->base_link_id_;
    tf.transform.translation.x = odom_trans.x();
    tf.transform.translation.y = odom_trans.y();
    tf.transform.translation.z = odom_trans.z();
    tf.transform.rotation.x = odom_quat.x();
    tf.transform.rotation.y = odom_quat.y();
    tf.transform.rotation.z = odom_quat.z();
    tf.transform.rotation.w = odom_quat.w();
    return tf;
}

void LiDAROdometryBaseNode::record_processing_times(const ProcessedFrame& frame) {
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

void LiDAROdometryBaseNode::add_delta_time(const std::string& name, double dt) {
    if (this->processing_times_.count(name) > 0) {
        this->processing_times_[name].push_back(dt);
    } else {
        this->processing_times_[name] = {dt};
    }
}

void LiDAROdometryBaseNode::print_processing_times(const std::string& name, double dt) {
    constexpr size_t LENGTH = 24;
    std::string log = name + ": ";
    for (size_t i = 0; i < LENGTH - name.length(); ++i) {
        log += " ";
    }
    log += "%9.2f us";
    RCLCPP_INFO(this->get_logger(), log.c_str(), dt);
}

void LiDAROdometryBaseNode::log_processing_times() {
    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MAX processing time");

    this->processing_times_.insert(this->pipeline_->get_total_processing_times().begin(),
                                   this->pipeline_->get_total_processing_times().end());

    for (auto& item : this->processing_times_) {
        if (item.second.empty()) {
            continue;
        }
        const double max = *std::max_element(item.second.begin(), item.second.end());
        this->print_processing_times(item.first, max);
    }

    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MEAN processing time");
    for (auto& item : this->processing_times_) {
        if (item.second.empty()) {
            continue;
        }
        const double avg =
            std::accumulate(item.second.begin(), item.second.end(), 0.0) / static_cast<double>(item.second.size());
        this->print_processing_times(item.first, avg);
    }

    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MEDIAN processing time");
    for (auto& item : this->processing_times_) {
        if (item.second.empty()) {
            continue;
        }
        std::sort(item.second.begin(), item.second.end());
        const double median = item.second[item.second.size() / 2];
        this->print_processing_times(item.first, median);
    }
    RCLCPP_INFO(this->get_logger(), "");
}

}  // namespace ros2
}  // namespace sycl_points
