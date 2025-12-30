#include "sycl_points_ros2/lidar_odometry_node.hpp"

#include <rclcpp_components/register_node_macro.hpp>
#include <sycl_points/ros2/convert.hpp>
#include <sycl_points/utils/time_utils.hpp>

#include "sycl_points_ros2/declare_lidar_odometry_params.hpp"

namespace sycl_points {
namespace ros2 {

/// @brief constructor
/// @param options node option
LiDAROdometryNode::LiDAROdometryNode(const rclcpp::NodeOptions& options) : rclcpp::Node("lidar_odometry", options) {
    // get parameters
    this->params_ = ros2::declare_lidar_odometry_parameters(this);

    this->pipeline_ = std::make_unique<pipeline::lidar_odometry::LiDAROdometryPipeline>(this->params_);
    this->pipeline_->get_device_queue()->print_device_info();

    // initialize buffer
    {
        this->msg_data_buffer_.reset(new shared_vector<uint8_t>(*this->pipeline_->get_device_queue()->ptr));
        this->scan_pc_.reset(new PointCloudShared(*this->pipeline_->get_device_queue()));
    }

    // Subscription
    {
        this->sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "points", rclcpp::QoS(10),
            std::bind(&LiDAROdometryNode::point_cloud_callback, this, std::placeholders::_1));
    }

    // Publisher
    {
        this->pub_preprocessed_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/preprocessed", rclcpp::QoS(5));

        this->pub_submap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("sycl_lo/submap", rclcpp::QoS(5));

        this->pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/odom", rclcpp::QoS(5));
        this->pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("sycl_lo/pose", rclcpp::QoS(5));
        this->pub_keyframe_pose_ =
            this->create_publisher<nav_msgs::msg::Odometry>("sycl_lo/keyframe/pose", rclcpp::QoS(5));

        this->tf_broadcaster_ =
            std::make_unique<tf2_ros::TransformBroadcaster>(*this, tf2_ros::DynamicBroadcasterQoS(1000));
    }

    RCLCPP_INFO(this->get_logger(), "Subscribe PointCloud: %s", this->sub_pc_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Preprocessed PointCloud: %s", this->pub_preprocessed_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Submap PointCloud: %s", this->pub_submap_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Odometry: %s", this->pub_odom_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Pose: %s", this->pub_pose_->get_topic_name());
    RCLCPP_INFO(this->get_logger(), "Publish Keyframe Pose: %s", this->pub_keyframe_pose_->get_topic_name());
}

/// @brief destructor
LiDAROdometryNode::~LiDAROdometryNode() {
    // processing time log
    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MAX processing time");

    this->processing_times_.insert(this->pipeline_->get_total_processing_times().begin(),
                                   this->pipeline_->get_total_processing_times().end());

    for (auto& item : this->processing_times_) {
        const double max = *std::max_element(item.second.begin(), item.second.end());
        this->print_processing_times(item.first, max);
    }

    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MEAN processing time");
    for (auto& item : this->processing_times_) {
        const double avg =
            std::accumulate(item.second.begin(), item.second.end(), 0.0) / static_cast<double>(item.second.size());
        this->print_processing_times(item.first, avg);
    }

    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "MEDIAN processing time");
    for (auto& item : this->processing_times_) {
        std::sort(item.second.begin(), item.second.end());
        const double median = item.second[item.second.size() / 2];
        this->print_processing_times(item.first, median);
    }
    RCLCPP_INFO(this->get_logger(), "");
}

void LiDAROdometryNode::point_cloud_callback(const sensor_msgs::msg::PointCloud2::UniquePtr msg) {
    const double timestamp = rclcpp::Time(msg->header.stamp).seconds();

    double dt_from_ros2_msg = 0.0;
    time_utils::measure_execution(
        [&]() {
            return fromROS2msg(*this->pipeline_->get_device_queue(), *msg, this->scan_pc_, this->msg_data_buffer_);
        },
        dt_from_ros2_msg);

    if (this->scan_pc_->size() == 0) {
        RCLCPP_WARN(this->get_logger(), "input point cloud is empty");
        return;
    }

    const auto ret = this->pipeline_->process(this->scan_pc_, timestamp);
    using ResultType = pipeline::lidar_odometry::LiDAROdometryPipeline::ResultType;
    if (ret >= ResultType::error) {
        RCLCPP_WARN(this->get_logger(), "lidar odometry failed: %s", this->pipeline_->get_error_message().c_str());
        return;
    }

    // publish ROS2 message
    double dt_ros2_msg = 0.0;
    time_utils::measure_execution(
        [&]() {
            const auto& reg_result = this->pipeline_->get_registration_result();
            const auto& last_keyframe_pose = this->pipeline_->get_last_keyframe_pose();

            this->publish_odom(msg->header, reg_result);
            this->publish_keyframe_pose(msg->header, last_keyframe_pose);

            if (this->pub_preprocessed_->get_subscription_count() > 0) {
                const auto preprocessed_msg = toROS2msg(this->pipeline_->get_preprocessed_point_cloud(), msg->header);
                if (preprocessed_msg != nullptr) {
                    this->pub_preprocessed_->publish(*preprocessed_msg);
                }
            }

            if (this->pub_submap_->get_subscription_count() > 0) {
                auto submap_msg = toROS2msg(this->pipeline_->get_submap_point_cloud(), msg->header);
                if (submap_msg != nullptr) {
                    submap_msg->header.frame_id = this->params_.odom_frame_id;
                    this->pub_submap_->publish(*submap_msg);
                }
            }
        },
        dt_ros2_msg);

    // processing time log
    {
        const auto& current_processing_time = this->pipeline_->get_current_processing_time();
        double total_time = 0.0;
        total_time += dt_from_ros2_msg;
        total_time += dt_ros2_msg;
        for (auto& item : current_processing_time) {
            total_time += item.second;
        }

        this->add_delta_time("0. from ROS 2 msg", dt_from_ros2_msg);
        this->add_delta_time("5. publish ROS 2 msg", dt_ros2_msg);
        this->add_delta_time("6. total", total_time);

        this->print_processing_times("0. from ROS 2 msg", dt_from_ros2_msg);
        for (auto& [process_name, time] : current_processing_time) {
            this->print_processing_times(process_name, time);
        }
        this->print_processing_times("5. publish ROS 2 msg", dt_ros2_msg);
        this->print_processing_times("6. total", total_time);
        RCLCPP_INFO(this->get_logger(), "");
    }
}

void LiDAROdometryNode::publish_odom(const std_msgs::msg::Header& header,
                                     const algorithms::registration::RegistrationResult& reg_result) {
    const auto odom_trans = reg_result.T.translation();
    const Eigen::Quaternionf odom_quat(reg_result.T.rotation());
    {
        geometry_msgs::msg::TransformStamped::SharedPtr tf;
        tf.reset(new geometry_msgs::msg::TransformStamped);
        tf->header.stamp = header.stamp;
        tf->header.frame_id = this->params_.odom_frame_id;
        tf->child_frame_id = this->params_.base_link_id;

        tf->transform.translation.x = odom_trans.x();
        tf->transform.translation.y = odom_trans.y();
        tf->transform.translation.z = odom_trans.z();
        tf->transform.rotation.x = odom_quat.x();
        tf->transform.rotation.y = odom_quat.y();
        tf->transform.rotation.z = odom_quat.z();
        tf->transform.rotation.w = odom_quat.w();
        this->tf_broadcaster_->sendTransform(*std::move(tf));
    }

    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = header.stamp;
    pose.header.frame_id = this->params_.odom_frame_id;
    pose.pose.position.x = odom_trans.x();
    pose.pose.position.y = odom_trans.y();
    pose.pose.position.z = odom_trans.z();
    pose.pose.orientation.x = odom_quat.x();
    pose.pose.orientation.y = odom_quat.y();
    pose.pose.orientation.z = odom_quat.z();
    pose.pose.orientation.w = odom_quat.w();
    this->pub_pose_->publish(pose);

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = header.stamp;
    odom_msg.header.frame_id = this->params_.odom_frame_id;
    odom_msg.child_frame_id = this->params_.base_link_id;
    odom_msg.pose.pose.position = pose.pose.position;
    odom_msg.pose.pose.orientation = pose.pose.orientation;
    // convert Hessian to Covariance
    {
        // rotation first
        const Eigen::Matrix<float, 6, 6> cov_reg = reg_result.H.inverse();

        // ROS covariance is translation first
        Eigen::Matrix<float, 6, 6> cov_odom;
        cov_odom.block<3, 3>(0, 0) = cov_reg.block<3, 3>(3, 3);
        cov_odom.block<3, 3>(3, 3) = cov_reg.block<3, 3>(0, 0);
        cov_odom.block<3, 3>(0, 3) = cov_reg.block<3, 3>(3, 0);
        cov_odom.block<3, 3>(3, 0) = cov_reg.block<3, 3>(0, 3);

        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(odom_msg.pose.covariance.data()) =
            cov_odom.cast<double>();
    }
    this->pub_odom_->publish(odom_msg);
}

void LiDAROdometryNode::publish_keyframe_pose(const std_msgs::msg::Header& header, const Eigen::Isometry3f& odom) {
    const auto odom_trans = odom.translation();
    const Eigen::Quaternionf odom_quat(odom.rotation());

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header = header;
    odom_msg.header.frame_id = this->params_.odom_frame_id;
    odom_msg.child_frame_id = this->params_.base_link_id;
    odom_msg.pose.pose.position.x = odom_trans.x();
    odom_msg.pose.pose.position.y = odom_trans.y();
    odom_msg.pose.pose.position.z = odom_trans.z();
    odom_msg.pose.pose.orientation.x = odom_quat.x();
    odom_msg.pose.pose.orientation.y = odom_quat.y();
    odom_msg.pose.pose.orientation.z = odom_quat.z();
    odom_msg.pose.pose.orientation.w = odom_quat.w();
    this->pub_keyframe_pose_->publish(odom_msg);
}
}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LiDAROdometryNode)
