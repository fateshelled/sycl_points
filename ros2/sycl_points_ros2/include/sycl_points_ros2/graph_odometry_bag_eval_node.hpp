#pragma once

#include <fstream>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "sycl_points_ros2/graph_odometry_base_node.hpp"

namespace sycl_points {
namespace ros2 {

class GraphOdometryBagEvalNode : public GraphOdometryBaseNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GraphOdometryBagEvalNode(const rclcpp::NodeOptions& options);
    ~GraphOdometryBagEvalNode() override;

private:
    void run();
    std::string detect_storage_id(const std::string& uri) const;
    void write_tum_line(const geometry_msgs::msg::PoseStamped& pose_msg);

    rclcpp::TimerBase::SharedPtr start_timer_ = nullptr;
    std::ofstream tum_stream_;

    std::string bag_uri_;
    std::string output_tum_;
    double start_offset_sec_ = 0.0;
    bool write_first_frame_ = true;
    bool exit_on_end_ = true;
};

}  // namespace ros2
}  // namespace sycl_points
