#include "sycl_points_ros2/lidar_odometry_bag_eval_node.hpp"

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

LiDAROdometryBagEvalNode::LiDAROdometryBagEvalNode(const rclcpp::NodeOptions& options)
    : LiDAROdometryBaseNode("lidar_odometry_bag_eval", options) {

    this->bag_uri_ = this->declare_parameter<std::string>("rosbag/uri", "");
    this->bag_topic_ = this->declare_parameter<std::string>("rosbag/topic", "/os_cloud_node/points");
    this->start_offset_sec_ = this->declare_parameter<double>("rosbag/start_offset_sec", 0.0);
    this->output_tum_ = this->declare_parameter<std::string>("eval/output_tum", "sycl_lo_odom.tum");
    this->write_first_frame_ = this->declare_parameter<bool>("eval/write_first_frame", true);
    this->exit_on_end_ = this->declare_parameter<bool>("eval/exit_on_end", true);

    this->initialize_processing();

    this->start_timer_ = this->create_wall_timer(std::chrono::milliseconds(10), [this]() { this->run(); });
}

LiDAROdometryBagEvalNode::~LiDAROdometryBagEvalNode() {
    if (this->tum_stream_.is_open()) {
        this->tum_stream_.flush();
        this->tum_stream_.close();
    }
}

void LiDAROdometryBagEvalNode::run() {
    if (this->start_timer_ != nullptr) {
        this->start_timer_->cancel();
    }

    try {
        if (this->bag_uri_.empty()) {
            throw std::runtime_error("`rosbag/uri` must not be empty");
        }
        if (this->bag_topic_.empty()) {
            throw std::runtime_error("`rosbag/topic` must not be empty");
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
        rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serializer;

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
            if (bag_message == nullptr || bag_message->topic_name != this->bag_topic_) {
                continue;
            }

            sensor_msgs::msg::PointCloud2 msg;
            rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
            serializer.deserialize_message(&serialized_msg, &msg);

            ++handled_frames;
            const auto frame = this->process_point_cloud_message(msg);

            if (frame.result == ResultType::success ||
                (frame.result == ResultType::first_frame && this->write_first_frame_)) {
                const auto pose_msg = this->make_pose_message(msg.header, frame.odom);
                this->write_tum_line(pose_msg);
                ++written_frames;
            }
        }

        this->tum_stream_.flush();
        this->tum_stream_.close();

        RCLCPP_INFO(this->get_logger(),
                    "Bag evaluation finished. topic=%s handled_frames=%ld written_frames=%ld tum=%s",
                    this->bag_topic_.c_str(), handled_frames, written_frames, this->output_tum_.c_str());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "bag evaluation failed: %s", e.what());
    }

    if (this->exit_on_end_) {
        rclcpp::shutdown();
    }
}

std::string LiDAROdometryBagEvalNode::detect_storage_id(const std::string& uri) const {
    namespace fs = std::filesystem;

    const fs::path path(uri);
    if (path.extension() == ".mcap") {
        return "mcap";
    }
    if (path.extension() == ".db3") {
        return "sqlite3";
    }

    const fs::path metadata_path = path / "metadata.yaml";
    if (!fs::exists(metadata_path)) {
        throw std::runtime_error("failed to detect storage id: metadata.yaml not found at " + metadata_path.string());
    }

    std::ifstream metadata_stream(metadata_path);
    if (!metadata_stream.is_open()) {
        throw std::runtime_error("failed to open metadata.yaml: " + metadata_path.string());
    }

    std::string line;
    while (std::getline(metadata_stream, line)) {
        constexpr std::string_view key = "  storage_identifier:";
        const auto pos = line.find(key);
        if (pos == std::string::npos) {
            continue;
        }

        std::string value = line.substr(pos + key.size());
        const auto first = value.find_first_not_of(" \t");
        if (first == std::string::npos) {
            break;
        }
        value = value.substr(first);
        const auto last = value.find_last_not_of(" \t\r\n");
        if (last != std::string::npos) {
            value = value.substr(0, last + 1);
        }
        if (!value.empty()) {
            return value;
        }
    }

    throw std::runtime_error("failed to detect storage identifier from metadata: " + metadata_path.string());
}

void LiDAROdometryBagEvalNode::write_tum_line(const geometry_msgs::msg::PoseStamped& pose_msg) {
    const double timestamp =
        static_cast<double>(pose_msg.header.stamp.sec) + static_cast<double>(pose_msg.header.stamp.nanosec) * 1.0e-9;

    this->tum_stream_ << std::fixed << std::setprecision(9) << timestamp << " " << pose_msg.pose.position.x << " "
                      << pose_msg.pose.position.y << " " << pose_msg.pose.position.z << " "
                      << pose_msg.pose.orientation.x << " " << pose_msg.pose.orientation.y << " "
                      << pose_msg.pose.orientation.z << " " << pose_msg.pose.orientation.w << "\n";
}

}  // namespace ros2
}  // namespace sycl_points

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::LiDAROdometryBagEvalNode)
