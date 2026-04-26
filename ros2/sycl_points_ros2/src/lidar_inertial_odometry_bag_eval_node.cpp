#include "sycl_points_ros2/lidar_inertial_odometry_bag_eval_node.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_transport/reader_writer_factory.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sycl_points {
namespace ros2 {

LidarInertialOdometryBagEvalNode::LidarInertialOdometryBagEvalNode(const rclcpp::NodeOptions& options)
    : LidarInertialOdometryBaseNode("lidar_inertial_odometry_bag_eval", options) {
    this->bag_uri_ = this->declare_parameter<std::string>("rosbag/uri", "");
    this->start_offset_sec_ = this->declare_parameter<double>("rosbag/start_offset/sec", 0.0);
    this->output_tum_ = this->declare_parameter<std::string>("eval/output_tum", "sycl_lio_odom.tum");
    this->write_first_frame_ = this->declare_parameter<bool>("eval/write_first_frame", true);
    this->exit_on_end_ = this->declare_parameter<bool>("eval/exit_on_end", true);

    this->initialize_processing();

    this->start_timer_ = this->create_wall_timer(std::chrono::milliseconds(10), [this]() { this->run(); });
}

LidarInertialOdometryBagEvalNode::~LidarInertialOdometryBagEvalNode() {
    if (this->tum_stream_.is_open()) {
        this->tum_stream_.flush();
        this->tum_stream_.close();
    }
}

void LidarInertialOdometryBagEvalNode::run() {
    if (this->start_timer_ != nullptr) {
        this->start_timer_->cancel();
    }

    try {
        if (this->points_topic_.empty()) {
            throw std::runtime_error("`points_topic` must not be empty");
        }
        if (this->imu_topic_.empty()) {
            throw std::runtime_error("`imu_topic` must not be empty");
        }
        if (this->bag_uri_.empty()) {
            throw std::runtime_error("`rosbag/uri` must not be empty");
        }
        if (this->output_tum_.empty()) {
            throw std::runtime_error("`eval/output_tum` must not be empty");
        }

        const std::filesystem::path output_path(this->output_tum_);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }

        this->tum_stream_.open(output_path, std::ios::out | std::ios::trunc);
        if (!this->tum_stream_.is_open()) {
            throw std::runtime_error("failed to open output tum file: " + output_path.string());
        }

        rosbag2_storage::StorageOptions storage_options;
        storage_options.uri = this->bag_uri_;
        storage_options.storage_id = this->detect_storage_id(this->bag_uri_);

        auto reader = rosbag2_transport::ReaderWriterFactory::make_reader(storage_options);
        reader->open(storage_options);
        rclcpp::Serialization<sensor_msgs::msg::PointCloud2> pc_serializer;
        rclcpp::Serialization<sensor_msgs::msg::Imu> imu_serializer;

        const auto& metadata = reader->get_metadata();
        const auto bag_start_time_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(metadata.starting_time.time_since_epoch()).count();
        const auto start_time_ns = bag_start_time_ns + static_cast<int64_t>(this->start_offset_sec_ * 1.0e9);
        if (start_time_ns > bag_start_time_ns) {
            reader->seek(start_time_ns);
        }

        int64_t handled_frames = 0;
        int64_t written_frames = 0;

        while (rclcpp::ok() && reader->has_next()) {
            auto bag_message = reader->read_next();
            if (bag_message == nullptr) continue;

            // IMU is always required for LIO — feed all IMU measurements to the pipeline
            if (bag_message->topic_name == this->imu_topic_) {
                sensor_msgs::msg::Imu imu_msg;
                rclcpp::SerializedMessage serialized_imu(*bag_message->serialized_data);
                imu_serializer.deserialize_message(&serialized_imu, &imu_msg);

                imu::IMUMeasurement meas;
                meas.timestamp = rclcpp::Time(imu_msg.header.stamp).seconds();
                meas.gyro = Eigen::Vector3f(static_cast<float>(imu_msg.angular_velocity.x),
                                            static_cast<float>(imu_msg.angular_velocity.y),
                                            static_cast<float>(imu_msg.angular_velocity.z));
                meas.accel = Eigen::Vector3f(static_cast<float>(imu_msg.linear_acceleration.x),
                                             static_cast<float>(imu_msg.linear_acceleration.y),
                                             static_cast<float>(imu_msg.linear_acceleration.z));
                this->pipeline_->add_imu_measurement(meas);
                continue;
            }

            if (bag_message->topic_name != this->points_topic_) continue;

            sensor_msgs::msg::PointCloud2 msg;
            rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
            pc_serializer.deserialize_message(&serialized_msg, &msg);

            ++handled_frames;
            const auto frame = this->process_point_cloud_message(msg);
            this->record_processing_times(frame);

            const bool should_write_tum = frame.result == ResultType::success ||
                                          (frame.result == ResultType::first_frame && this->write_first_frame_);
            if (should_write_tum) {
                const auto pose_msg = this->make_pose_message(msg.header, frame.odom);
                this->write_tum_line(pose_msg);
                ++written_frames;
            }
        }

        this->tum_stream_.flush();
        this->tum_stream_.close();

        RCLCPP_INFO(this->get_logger(),
                    "Bag evaluation finished. topic=%s handled_frames=%ld written_frames=%ld tum=%s",
                    this->points_topic_.c_str(), handled_frames, written_frames, this->output_tum_.c_str());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "bag evaluation failed: %s", e.what());
    }

    if (this->exit_on_end_) {
        rclcpp::shutdown();
    }
}

std::string LidarInertialOdometryBagEvalNode::detect_storage_id(const std::string& uri) const {
    rosbag2_storage::MetadataIo io;
    try {
        const auto metadata = io.read_metadata(uri);
        return metadata.storage_identifier;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to detect storage id from metadata: " + std::string(e.what()));
    }
}

void LidarInertialOdometryBagEvalNode::write_tum_line(const geometry_msgs::msg::PoseStamped& pose_msg) {
    const double timestamp =
        static_cast<double>(pose_msg.header.stamp.sec) + static_cast<double>(pose_msg.header.stamp.nanosec) * 1.0e-9;

    this->tum_stream_ << std::fixed << std::setprecision(9) << timestamp << " " << pose_msg.pose.position.x << " "
                      << pose_msg.pose.position.y << " " << pose_msg.pose.position.z << " "
                      << pose_msg.pose.orientation.x << " " << pose_msg.pose.orientation.y << " "
                      << pose_msg.pose.orientation.z << " " << pose_msg.pose.orientation.w << "\n";
}

}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LidarInertialOdometryBagEvalNode)
