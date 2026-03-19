#pragma once

#include <fstream>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "sycl_points_ros2/lidar_odometry_base_node.hpp"

namespace sycl_points {
namespace ros2 {

class LiDAROdometryBagEvalNode : public LiDAROdometryBaseNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LiDAROdometryBagEvalNode(const rclcpp::NodeOptions& options);
    ~LiDAROdometryBagEvalNode() override;

private:
    void run();
    std::string detect_storage_id(const std::string& uri) const;
    void write_tum_line(const geometry_msgs::msg::PoseStamped& pose_msg);

    rclcpp::TimerBase::SharedPtr start_timer_ = nullptr;
    std::ofstream tum_stream_;

    std::string bag_uri_;
    std::string bag_topic_;
    std::string output_tum_;
    double start_offset_sec_ = 0.0;
    int64_t max_frames_ = 0;
    bool write_first_frame_ = true;
    bool exit_on_end_ = true;
};

}  // namespace ros2
}  // namespace sycl_points
