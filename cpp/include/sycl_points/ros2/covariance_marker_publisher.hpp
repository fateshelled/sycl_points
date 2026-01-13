#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <visualization_msgs/msg/marker_array.hpp>

#include "sycl_points/points/point_cloud.hpp"

namespace sycl_points {
namespace ros2 {

/// @brief Configuration parameters for covariance visualization
struct CovarianceMarkerConfig {
    std::string topic_name = "covariance_markers";
    std::string marker_ns = "covariance_ellipsoids";
    float scale_factor = 1.0f;       // Sigma multiplier for ellipsoid size
    double min_scale = 0.001;        // Minimum ellipsoid scale
    double max_scale = 1.0;          // Maximum ellipsoid scale
    float alpha = 0.5f;              // Marker transparency
    bool color_by_planarity = true;  // Color based on planarity indicator
    float default_r = 0.0f;          // Default red color (if not using planarity)
    float default_g = 1.0f;          // Default green color
    float default_b = 0.2f;          // Default blue color
    size_t qos_depth = 10;           // QoS history depth
};

/// @brief Publisher class for visualizing point cloud covariance matrices as ellipsoid markers
class CovarianceMarkerPublisher {
public:
    using Ptr = std::shared_ptr<CovarianceMarkerPublisher>;
    using ConstPtr = std::shared_ptr<const CovarianceMarkerPublisher>;

    /// @brief Constructor
    /// @param node ROS2 node to create publisher on
    /// @param config Configuration parameters
    explicit CovarianceMarkerPublisher(rclcpp::Node& node,
                                       const CovarianceMarkerConfig& config = CovarianceMarkerConfig())
        : config_(config), node_(node) {
        init_publisher();
    }

    /// @brief Constructor with topic name override
    /// @param node ROS2 node to create publisher on
    /// @param topic_name Topic name for the MarkerArray publisher
    CovarianceMarkerPublisher(rclcpp::Node& node, const std::string& topic_name) : node_(node) {
        config_.topic_name = topic_name;
        init_publisher();
    }

    /// @brief Get the topic name
    /// @return Topic name string
    const char* get_topic_name() const { return publisher_->get_topic_name(); }

    /// @brief Get the number of subscribers
    /// @return Number of subscribers to the marker topic
    size_t get_subscription_count() const { return publisher_->get_subscription_count(); }

    /// @brief Check if there are any subscribers
    /// @return True if there are subscribers
    bool has_subscribers() const { return get_subscription_count() > 0; }

    /// @brief Get the configuration
    /// @return Current configuration
    const CovarianceMarkerConfig& get_config() const { return config_; }

    /// @brief Set the configuration
    /// @param config New configuration
    void set_config(const CovarianceMarkerConfig& config) { config_ = config; }

    /// @brief Publish covariance markers for a point cloud
    /// @param header ROS2 message header (timestamp and frame_id)
    /// @param cloud Point cloud with covariance data
    void publish(const std_msgs::msg::Header& header, const PointCloudShared& cloud) {
        if (!cloud.has_cov() || cloud.size() == 0) {
            return;
        }

        auto marker_array = create_marker_array(header, cloud);
        publisher_->publish(marker_array);
    }

    /// @brief Publish covariance markers only if there are subscribers
    /// @param header ROS2 message header (timestamp and frame_id)
    /// @param cloud Point cloud with covariance data
    /// @return True if markers were published
    bool publish_if_subscribed(const std_msgs::msg::Header& header, const PointCloudShared& cloud) {
        if (!has_subscribers()) {
            return false;
        }
        publish(header, cloud);
        return true;
    }

    /// @brief Create a MarkerArray from point cloud covariances
    /// @param header ROS2 message header
    /// @param cloud Point cloud with covariance data
    /// @return MarkerArray message containing ellipsoid markers
    visualization_msgs::msg::MarkerArray create_marker_array(const std_msgs::msg::Header& header,
                                                             const PointCloudShared& cloud) const {
        visualization_msgs::msg::MarkerArray marker_array;

        if (!cloud.has_cov() || cloud.size() == 0) {
            return marker_array;
        }

        const size_t num_points = cloud.size();
        const auto* points_ptr = cloud.points_ptr();
        const auto* covs_ptr = cloud.covs_ptr();

        // Delete all previous markers
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header = header;
        delete_marker.ns = config_.marker_ns;
        delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

        // Reserve capacity for all markers (delete marker + one per point)
        marker_array.markers.reserve(num_points + 1);
        marker_array.markers.push_back(delete_marker);

        for (size_t i = 0; i < num_points; ++i) {
            // Prevent marker ID overflow by wrapping around INT32_MAX
            const int32_t marker_id = static_cast<int32_t>(i % INT32_MAX);
            auto marker = create_ellipsoid_marker(header, points_ptr[i], covs_ptr[i], marker_id);
            if (marker.has_value()) {
                marker_array.markers.push_back(marker.value());
            }
        }

        return marker_array;
    }

private:
    /// @brief Initialize the publisher with current configuration
    void init_publisher() {
        publisher_ = node_.create_publisher<visualization_msgs::msg::MarkerArray>(
            config_.topic_name, rclcpp::QoS(static_cast<size_t>(config_.qos_depth)));
    }

    /// @brief Create an ellipsoid marker from a point and its covariance matrix
    /// @param header ROS2 message header
    /// @param point Point position
    /// @param cov 4x4 covariance matrix
    /// @param id Marker ID
    /// @return Optional marker (empty if eigendecomposition fails)
    std::optional<visualization_msgs::msg::Marker> create_ellipsoid_marker(const std_msgs::msg::Header& header,
                                                                           const PointType& point,
                                                                           const Covariance& cov, int32_t id) const {
        // Extract 3x3 covariance block
        const Eigen::Matrix3f cov3x3 = cov.block<3, 3>(0, 0);

        // Perform eigenvalue decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov3x3);
        if (solver.info() != Eigen::Success) {
            return std::nullopt;
        }

        const Eigen::Vector3f eigenvalues = solver.eigenvalues();
        const Eigen::Matrix3f eigenvectors = solver.eigenvectors();

        // Skip if any eigenvalue is invalid (NaN, negative, or too small)
        constexpr float MIN_EIGENVALUE = 1e-8f;
        if (!eigenvalues.allFinite() || eigenvalues.minCoeff() < MIN_EIGENVALUE) {
            return std::nullopt;
        }

        // Create ellipsoid marker
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = config_.marker_ns;
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Position
        marker.pose.position.x = static_cast<double>(point.x());
        marker.pose.position.y = static_cast<double>(point.y());
        marker.pose.position.z = static_cast<double>(point.z());

        // Orientation from eigenvectors
        // Ensure proper rotation matrix (det = 1)
        Eigen::Matrix3f rotation = eigenvectors;
        if (rotation.determinant() < 0) {
            rotation.col(0) *= -1.0f;
        }
        const Eigen::Quaternionf quat(rotation);
        marker.pose.orientation.x = static_cast<double>(quat.x());
        marker.pose.orientation.y = static_cast<double>(quat.y());
        marker.pose.orientation.z = static_cast<double>(quat.z());
        marker.pose.orientation.w = static_cast<double>(quat.w());

        // Scale from eigenvalues (sqrt for std dev, scaled for visualization)
        marker.scale.x = std::clamp(static_cast<double>(config_.scale_factor * std::sqrt(eigenvalues(0))),
                                    config_.min_scale, config_.max_scale);
        marker.scale.y = std::clamp(static_cast<double>(config_.scale_factor * std::sqrt(eigenvalues(1))),
                                    config_.min_scale, config_.max_scale);
        marker.scale.z = std::clamp(static_cast<double>(config_.scale_factor * std::sqrt(eigenvalues(2))),
                                    config_.min_scale, config_.max_scale);

        // Color
        if (config_.color_by_planarity) {
            // Color based on eigenvalue ratio (planarity indicator)
            const float planarity = std::clamp(1.0f - eigenvalues(0) / (eigenvalues(2) + 1e-6f), 0.0f, 1.0f);
            marker.color.r = static_cast<float>(1.0 - planarity);
            marker.color.g = static_cast<float>(planarity);
            marker.color.b = 0.2f;
        } else {
            marker.color.r = config_.default_r;
            marker.color.g = config_.default_g;
            marker.color.b = config_.default_b;
        }
        marker.color.a = config_.alpha;

        return marker;
    }

    CovarianceMarkerConfig config_;
    rclcpp::Node& node_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
};

}  // namespace ros2
}  // namespace sycl_points
