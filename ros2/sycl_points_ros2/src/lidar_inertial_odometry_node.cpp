#include "sycl_points_ros2/lidar_inertial_odometry_node.hpp"

#include <rclcpp_components/register_node_macro.hpp>
#include <stdexcept>

namespace sycl_points {
namespace ros2 {

LidarInertialOdometryNode::LidarInertialOdometryNode(const rclcpp::NodeOptions& options)
    : LidarInertialOdometryBaseNode("lidar_inertial_odometry", options) {
    this->initialize_processing();
    this->initialize_publishers({});

    const auto max_pending = this->declare_parameter<int64_t>("processing/max_pending_point_clouds", 3);
    if (max_pending <= 0) {
        throw std::invalid_argument("processing/max_pending_point_clouds must be positive");
    }
    max_pending_point_clouds_ = static_cast<std::size_t>(max_pending);

    // -----------------------------------------------------------------------
    // Subscriptions
    // -----------------------------------------------------------------------
    // Keep LiDAR enqueueing, IMU ingestion, and heavy processing independent.
    // The processing group is MutuallyExclusive so LIO frames remain serialized.
    cb_group_lidar_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cb_group_imu_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cb_group_processing_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

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

    processing_timer_ = this->create_wall_timer(std::chrono::milliseconds(1),
                                                std::bind(&LidarInertialOdometryNode::processing_timer_callback, this),
                                                cb_group_processing_, false);
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void LidarInertialOdometryNode::point_cloud_callback(sensor_msgs::msg::PointCloud2::UniquePtr msg) {
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        if (pending_point_clouds_.size() >= max_pending_point_clouds_) {
            pending_point_clouds_.pop_front();
            ++dropped_pending_point_clouds_;
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Point cloud processing backlog reached %zu frames; dropping the oldest pending "
                                 "frame (total dropped=%zu)",
                                 max_pending_point_clouds_, dropped_pending_point_clouds_);
        }
        pending_point_clouds_.push_back(std::move(msg));
        processing_timer_->reset();
    }
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

void LidarInertialOdometryNode::processing_timer_callback() {
    using IMUCoverage = pipeline::lidar_inertial_odometry::LidarInertialOdometryPipeline::IMUCoverage;

    if (active_point_cloud_ == nullptr) {
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            if (pending_point_clouds_.empty()) {
                processing_timer_->cancel();
                return;
            }
            active_point_cloud_ = std::move(pending_point_clouds_.front());
            pending_point_clouds_.pop_front();
        }

        active_frame_ = ProcessedFrame{};
        if (!this->prepare_point_cloud_message(*active_point_cloud_, active_frame_)) {
            this->record_processing_times(active_frame_);
            active_point_cloud_.reset();
            return;
        }
    }

    const double frame_timestamp = rclcpp::Time(active_point_cloud_->header.stamp).seconds();
    auto coverage = pipeline_->get_frame_imu_coverage(frame_timestamp);

    if (params_.imu.deskew.enable && scan_pc_->has_timestamps() && scan_pc_->end_time_ms > scan_pc_->start_time_ms) {
        const double scan_start = scan_pc_->start_time_ms * 1e-3;
        const double scan_end = scan_pc_->end_time_ms * 1e-3;
        coverage = pipeline_->combine_imu_coverage(coverage, pipeline_->get_imu_coverage(scan_start, scan_end));
    }

    if (coverage == IMUCoverage::waiting_for_future) return;

    if (coverage == IMUCoverage::start_expired) {
        RCLCPP_ERROR(this->get_logger(),
                     "Cannot process point cloud: required IMU samples have expired "
                     "(frame=%.9f). Increase imu/buffer_duration_sec or reduce processing backlog.",
                     frame_timestamp);
        active_frame_.result = ResultType::insufficient_imu_coverage;
        this->record_processing_times(active_frame_);
        active_point_cloud_.reset();
        return;
    }

    this->process_prepared_point_cloud_message(frame_timestamp, active_frame_);
    if (active_frame_.result == ResultType::success || active_frame_.result == ResultType::first_frame ||
        active_frame_.result == ResultType::imu_only) {
        this->publish_processed_frame(active_point_cloud_->header, active_frame_);
    }
    this->record_processing_times(active_frame_);
    active_point_cloud_.reset();
}

}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LidarInertialOdometryNode)
