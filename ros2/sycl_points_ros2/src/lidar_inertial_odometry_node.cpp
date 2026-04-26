#include "sycl_points_ros2/lidar_inertial_odometry_node.hpp"

#include <algorithm>
#include <numeric>
#include <rclcpp_components/register_node_macro.hpp>
#include <sycl_points/ros2/convert.hpp>
#include <sycl_points/utils/time_utils.hpp>

#include "sycl_points_ros2/declare_lidar_inertial_odometry_params.hpp"

namespace sycl_points {
namespace ros2 {

LidarInertialOdometryNode::LidarInertialOdometryNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("lidar_inertial_odometry", options) {
    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    params_ = declare_lidar_inertial_odometry_parameters(this);

    points_topic_             = this->declare_parameter<std::string>("points_topic", points_topic_);
    imu_topic_                = this->declare_parameter<std::string>("imu_topic", imu_topic_);
    input_convert_rgb_        = this->declare_parameter<bool>("input/convert_rgb", true);
    input_convert_intensity_  = this->declare_parameter<bool>("input/convert_intensity", true);

    points_qos_params_.history =
        this->declare_parameter<std::string>("points_qos/history", points_qos_params_.history);
    points_qos_params_.depth =
        this->declare_parameter<int64_t>("points_qos/depth", points_qos_params_.depth);
    points_qos_params_.reliability =
        this->declare_parameter<std::string>("points_qos/reliability", points_qos_params_.reliability);
    imu_qos_params_.history =
        this->declare_parameter<std::string>("imu_qos/history", imu_qos_params_.history);
    imu_qos_params_.depth =
        this->declare_parameter<int64_t>("imu_qos/depth", imu_qos_params_.depth);
    imu_qos_params_.reliability =
        this->declare_parameter<std::string>("imu_qos/reliability", imu_qos_params_.reliability);

    odom_frame_id_ = this->declare_parameter<std::string>("odom_frame_id", odom_frame_id_);
    base_link_id_  = this->declare_parameter<std::string>("base_link_id", base_link_id_);

    {
        const auto x  = this->declare_parameter<double>("T_base_link_to_lidar/x", 0.0);
        const auto y  = this->declare_parameter<double>("T_base_link_to_lidar/y", 0.0);
        const auto z  = this->declare_parameter<double>("T_base_link_to_lidar/z", 0.0);
        const auto qx = this->declare_parameter<double>("T_base_link_to_lidar/qx", 0.0);
        const auto qy = this->declare_parameter<double>("T_base_link_to_lidar/qy", 0.0);
        const auto qz = this->declare_parameter<double>("T_base_link_to_lidar/qz", 0.0);
        const auto qw = this->declare_parameter<double>("T_base_link_to_lidar/qw", 1.0);
        T_base_link_to_lidar_.setIdentity();
        T_base_link_to_lidar_.translation() << static_cast<float>(x), static_cast<float>(y), static_cast<float>(z);
        const Eigen::Quaternionf q(static_cast<float>(qw), static_cast<float>(qx),
                                   static_cast<float>(qy), static_cast<float>(qz));
        T_base_link_to_lidar_.matrix().block<3, 3>(0, 0) = q.normalized().matrix();
        T_lidar_to_base_link_ = T_base_link_to_lidar_.inverse();
    }
    {
        const auto x  = this->declare_parameter<double>("initial_base_link_pose/x", 0.0);
        const auto y  = this->declare_parameter<double>("initial_base_link_pose/y", 0.0);
        const auto z  = this->declare_parameter<double>("initial_base_link_pose/z", 0.0);
        const auto qx = this->declare_parameter<double>("initial_base_link_pose/qx", 0.0);
        const auto qy = this->declare_parameter<double>("initial_base_link_pose/qy", 0.0);
        const auto qz = this->declare_parameter<double>("initial_base_link_pose/qz", 0.0);
        const auto qw = this->declare_parameter<double>("initial_base_link_pose/qw", 1.0);
        Eigen::Isometry3f init = Eigen::Isometry3f::Identity();
        init.translation() << static_cast<float>(x), static_cast<float>(y), static_cast<float>(z);
        const Eigen::Quaternionf q(static_cast<float>(qw), static_cast<float>(qx),
                                   static_cast<float>(qy), static_cast<float>(qz));
        init.matrix().block<3, 3>(0, 0) = q.normalized().matrix();
        params_.pose.initial = init * T_base_link_to_lidar_;
    }

    // -----------------------------------------------------------------------
    // Pipeline
    // -----------------------------------------------------------------------
    pipeline_ = std::make_unique<pipeline::lidar_inertial_odometry::LidarInertialOdometryPipeline>(params_);
    pipeline_->get_device_queue()->print_device_info();

    msg_data_buffer_.reset(new shared_vector<uint8_t>(*pipeline_->get_device_queue()->ptr));
    scan_pc_.reset(new PointCloudShared(*pipeline_->get_device_queue()));

    // -----------------------------------------------------------------------
    // Publishers
    // -----------------------------------------------------------------------
    pub_preprocessed_  = this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/preprocessed", rclcpp::QoS(5));
    pub_submap_        = this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/submap", rclcpp::QoS(5));
    pub_pose_          = this->create_publisher<geometry_msgs::msg::PoseStamped>("sycl_lo/pose", rclcpp::QoS(5));
    pub_odom_          = this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/odom", rclcpp::QoS(5));
    pub_keyframe_pose_ = this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/keyframe/pose", rclcpp::QoS(5));
    tf_broadcaster_    = std::make_unique<tf2_ros::TransformBroadcaster>(*this, tf2_ros::DynamicBroadcasterQoS(1000));

    // -----------------------------------------------------------------------
    // Subscriptions
    // -----------------------------------------------------------------------
    cb_group_lidar_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cb_group_imu_   = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions lidar_opts;
    lidar_opts.callback_group = cb_group_lidar_;
    sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        points_topic_, points_qos_params_.to_qos(),
        std::bind(&LidarInertialOdometryNode::point_cloud_callback, this, std::placeholders::_1), lidar_opts);
    RCLCPP_INFO(this->get_logger(), "Subscribe PointCloud: %s", sub_pc_->get_topic_name());

    rclcpp::SubscriptionOptions imu_opts;
    imu_opts.callback_group = cb_group_imu_;
    sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, imu_qos_params_.to_qos(),
        std::bind(&LidarInertialOdometryNode::imu_callback, this, std::placeholders::_1), imu_opts);
    RCLCPP_INFO(this->get_logger(), "Subscribe IMU: %s", sub_imu_->get_topic_name());
}

LidarInertialOdometryNode::~LidarInertialOdometryNode() {
    log_processing_times();
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void LidarInertialOdometryNode::point_cloud_callback(const sensor_msgs::msg::PointCloud2::UniquePtr msg) {
    const double timestamp = rclcpp::Time(msg->header.stamp).seconds();

    double dt_convert = 0.0;
    bool converted = false;
    time_utils::measure_execution(
        [&]() {
            converted = fromROS2msg(*pipeline_->get_device_queue(), *msg, scan_pc_, msg_data_buffer_,
                                    input_convert_rgb_, input_convert_intensity_);
        },
        dt_convert);

    if (!converted || scan_pc_ == nullptr || scan_pc_->size() == 0) {
        RCLCPP_WARN(this->get_logger(), "failed to convert input point cloud");
        return;
    }

    const auto result = pipeline_->process(scan_pc_, timestamp);
    if (result >= pipeline::lidar_inertial_odometry::LidarInertialOdometryPipeline::ResultType::error) {
        RCLCPP_WARN(this->get_logger(), "LIO failed: %s", pipeline_->get_error_message().c_str());
        return;
    }

    if (result != pipeline::lidar_inertial_odometry::LidarInertialOdometryPipeline::ResultType::success &&
        result != pipeline::lidar_inertial_odometry::LidarInertialOdometryPipeline::ResultType::first_frame) {
        return;
    }

    const auto& header  = msg->header;
    const auto& odom    = pipeline_->get_odom();
    const auto& kf_pose = pipeline_->get_last_keyframe_pose();

    // Publish (timed, matching "5. publish ROS 2 msg" in the base node)
    double dt_publish = 0.0;
    time_utils::measure_execution(
        [&]() {
            tf_broadcaster_->sendTransform(make_transform_message(header, odom));
            pub_odom_->publish(make_odom_message(header, odom));
            pub_pose_->publish(make_pose_message(header, odom));
            pub_keyframe_pose_->publish(make_keyframe_pose_message(header, kf_pose));

            if (pub_preprocessed_->get_subscription_count() > 0) {
                const auto pc_msg = toROS2msg(pipeline_->get_preprocessed_point_cloud(), header);
                if (pc_msg) pub_preprocessed_->publish(*pc_msg);
            }
            if (pub_submap_->get_subscription_count() > 0) {
                auto submap_msg = toROS2msg(pipeline_->get_submap_point_cloud(), header);
                if (submap_msg) {
                    submap_msg->header.frame_id = odom_frame_id_;
                    pub_submap_->publish(*submap_msg);
                }
            }
        },
        dt_publish);

    record_processing_times(dt_convert, dt_publish);
}

void LidarInertialOdometryNode::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    imu::IMUMeasurement meas;
    meas.timestamp = rclcpp::Time(msg->header.stamp).seconds();
    meas.gyro  = Eigen::Vector3f(static_cast<float>(msg->angular_velocity.x),
                                  static_cast<float>(msg->angular_velocity.y),
                                  static_cast<float>(msg->angular_velocity.z));
    meas.accel = Eigen::Vector3f(static_cast<float>(msg->linear_acceleration.x),
                                  static_cast<float>(msg->linear_acceleration.y),
                                  static_cast<float>(msg->linear_acceleration.z));
    pipeline_->add_imu_measurement(meas);
}

// ---------------------------------------------------------------------------
// Message helpers (same logic as LiDAROdometryBaseNode)
// ---------------------------------------------------------------------------

nav_msgs::msg::Odometry LidarInertialOdometryNode::make_odom_message(const std_msgs::msg::Header& header,
                                                                      const Eigen::Isometry3f& odom) const {
    const Eigen::Isometry3f T = odom * T_lidar_to_base_link_;
    const Eigen::Quaternionf q(T.rotation());

    nav_msgs::msg::Odometry msg;
    msg.header.stamp    = header.stamp;
    msg.header.frame_id = odom_frame_id_;
    msg.child_frame_id  = base_link_id_;
    msg.pose.pose.position.x    = T.translation().x();
    msg.pose.pose.position.y    = T.translation().y();
    msg.pose.pose.position.z    = T.translation().z();
    msg.pose.pose.orientation.x = q.x();
    msg.pose.pose.orientation.y = q.y();
    msg.pose.pose.orientation.z = q.z();
    msg.pose.pose.orientation.w = q.w();
    return msg;
}

geometry_msgs::msg::PoseStamped LidarInertialOdometryNode::make_pose_message(const std_msgs::msg::Header& header,
                                                                              const Eigen::Isometry3f& odom) const {
    const auto odom_msg = make_odom_message(header, odom);
    geometry_msgs::msg::PoseStamped pose;
    pose.header = odom_msg.header;
    pose.pose   = odom_msg.pose.pose;
    return pose;
}

nav_msgs::msg::Odometry LidarInertialOdometryNode::make_keyframe_pose_message(const std_msgs::msg::Header& header,
                                                                               const Eigen::Isometry3f& odom) const {
    return make_odom_message(header, odom);
}

geometry_msgs::msg::TransformStamped LidarInertialOdometryNode::make_transform_message(
    const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom) const {
    const Eigen::Isometry3f T = odom * T_lidar_to_base_link_;
    const Eigen::Quaternionf q(T.rotation());

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp    = header.stamp;
    tf.header.frame_id = odom_frame_id_;
    tf.child_frame_id  = base_link_id_;
    tf.transform.translation.x = T.translation().x();
    tf.transform.translation.y = T.translation().y();
    tf.transform.translation.z = T.translation().z();
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();
    return tf;
}

void LidarInertialOdometryNode::record_processing_times(double dt_from_msg, double dt_publish) {
    double total = dt_from_msg + dt_publish;

    add_delta_time("0. from ROS 2 msg", dt_from_msg);
    print_processing_times("0. from ROS 2 msg", dt_from_msg);

    for (const auto& [name, dt] : pipeline_->get_current_processing_time()) {
        add_delta_time(name, dt);
        print_processing_times(name, dt);
        total += dt;
    }

    add_delta_time("5. publish ROS 2 msg", dt_publish);
    print_processing_times("5. publish ROS 2 msg", dt_publish);

    add_delta_time("6. total", total);
    print_processing_times("6. total", total);

    RCLCPP_INFO(this->get_logger(), "");
}

void LidarInertialOdometryNode::print_processing_times(const std::string& name, double dt) {
    constexpr size_t LENGTH = 24;
    std::string log = name + ": ";
    if (name.length() < LENGTH) {
        log += std::string(LENGTH - name.length(), ' ');
    }
    log += "%9.2f us";
    RCLCPP_INFO(this->get_logger(), log.c_str(), dt);
}

void LidarInertialOdometryNode::add_delta_time(const std::string& name, double dt) {
    processing_times_[name].push_back(dt);
}

void LidarInertialOdometryNode::log_processing_times() {
    RCLCPP_INFO(this->get_logger(), "");
    processing_times_.insert(pipeline_->get_total_processing_times().begin(),
                              pipeline_->get_total_processing_times().end());

    RCLCPP_INFO(this->get_logger(), "MAX processing time");
    for (auto& [name, times] : processing_times_) {
        if (times.empty()) continue;
        print_processing_times(name, *std::max_element(times.begin(), times.end()));
    }

    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MEAN processing time");
    for (auto& [name, times] : processing_times_) {
        if (times.empty()) continue;
        print_processing_times(name, std::accumulate(times.begin(), times.end(), 0.0) /
                                         static_cast<double>(times.size()));
    }

    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MEDIAN processing time");
    for (auto& [name, times] : processing_times_) {
        if (times.empty()) continue;
        std::sort(times.begin(), times.end());
        print_processing_times(name, times[times.size() / 2]);
    }
    RCLCPP_INFO(this->get_logger(), "");
}

}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LidarInertialOdometryNode)
