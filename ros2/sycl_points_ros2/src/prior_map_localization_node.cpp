#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <limits>
#include <map>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <numeric>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <sycl_points/algorithms/registration/degenerate_regularization.hpp>
#include <vector>

#include "sycl_points/algorithms/filter/preprocess_filter.hpp"
#include "sycl_points/algorithms/filter/voxel_downsampling.hpp"
#include "sycl_points/algorithms/localization/branch_and_bound.hpp"
#include "sycl_points/algorithms/registration/registration.hpp"
#include "sycl_points/pipeline/prior_map.hpp"
#include "sycl_points/ros2/convert.hpp"

namespace sycl_points::ros2 {
namespace {
using CloudMsg = sensor_msgs::msg::PointCloud2;
using OdomMsg = nav_msgs::msg::Odometry;
using Clock = std::chrono::steady_clock;

int64_t stamp(const builtin_interfaces::msg::Time& t) { return static_cast<int64_t>(t.sec) * 1000000000LL + t.nanosec; }

Eigen::Isometry3f pose(const geometry_msgs::msg::Pose& p) {
    Eigen::Quaternionf q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
    if (!q.coeffs().allFinite() || q.norm() < 1e-6f) {
        throw std::invalid_argument("Invalid pose quaternion");
    }
    Eigen::Isometry3f result = Eigen::Isometry3f::Identity();
    result.linear() = q.normalized().toRotationMatrix();
    result.translation() = Eigen::Vector3f(p.position.x, p.position.y, p.position.z);
    if (!result.matrix().allFinite()) {
        throw std::invalid_argument("Invalid pose translation");
    }
    return result;
}

geometry_msgs::msg::Pose to_msg(const Eigen::Isometry3f& t) {
    geometry_msgs::msg::Pose p;
    p.position.x = t.translation().x();
    p.position.y = t.translation().y();
    p.position.z = t.translation().z();
    Eigen::Quaternionf q(t.rotation());
    p.orientation.x = q.x();
    p.orientation.y = q.y();
    p.orientation.z = q.z();
    p.orientation.w = q.w();
    return p;
}
}  // namespace

class PriorMapLocalizationNode : public rclcpp::Node {
public:
    // Declares parameters, builds the prior map, and wires the ROS 2 interfaces.
    explicit PriorMapLocalizationNode(const rclcpp::NodeOptions& options)
        : Node("prior_map_localization", options), parameters_(declare_parameters()) {
        const auto device =
            sycl_utils::device_selector::select_device(this->parameters_.device_vendor, this->parameters_.device_type);
        this->queue_ = std::make_unique<sycl_utils::DeviceQueue>(device);
        this->queue_->print_device_info();

        this->source_pc_ = std::make_shared<PointCloudShared>(*this->queue_);
        this->registration_pc_ = std::make_shared<PointCloudShared>(*this->queue_);
        this->msg_data_buffer_ = std::make_shared<shared_vector<uint8_t>>(*this->queue_->ptr);
        this->preprocess_filter_ = std::make_shared<algorithms::filter::PreprocessFilter>(*this->queue_);
        this->voxel_ = std::make_shared<algorithms::filter::VoxelGrid>(  //
            *this->queue_, this->parameters_.runtime.source_voxel_size);

        this->registration_ = std::make_unique<algorithms::registration::Registration>(  //
            *this->queue_, this->parameters_.registration);

        const auto map_path = declare_parameter<std::string>("map/path", "");
        if (map_path.empty()) {
            throw std::invalid_argument("map/path must point to a PCD or PLY file");
        }
        this->map_ = std::make_unique<pipeline::PriorMap>(*this->queue_, map_path, this->parameters_.map);
        RCLCPP_INFO(get_logger(), "Prior map: %zu points, %zu covariance eigenvalues clamped", this->map_->size(),
                    this->map_->covariance_clamp_count());

        if (this->parameters_.bnb_enable) {
            this->bnb_ = std::make_unique<algorithms::localization::BranchAndBoundLocalizer>(
                *this->queue_, this->parameters_.branch_and_bound);
            this->bnb_->set_target(this->map_->cloud());
            RCLCPP_INFO(get_logger(), "Branch and bound target: %zu points", this->bnb_->target_size());
        }

        this->map_cloud_pub_ = create_publisher<CloudMsg>(
            "localization/prior_map", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());

        this->tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        this->pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("localization/pose", 5);
        this->status_pub_ = create_publisher<std_msgs::msg::String>("localization/status", 5);
        this->odom_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions odom_options;
        odom_options.callback_group = this->odom_group_;
        this->odom_sub_ = create_subscription<OdomMsg>(
            "sycl_lio/odom", rclcpp::QoS(5), [this](OdomMsg::ConstSharedPtr msg) { on_odom(std::move(msg)); },
            odom_options);
        this->cloud_sub_ = create_subscription<CloudMsg>(
            "sycl_lio/submap", rclcpp::QoS(5), [this](CloudMsg::ConstSharedPtr msg) { on_cloud(std::move(msg)); });

        // Publish the full prior map after the executor starts, then stop the timer.
        this->map_cloud_timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] {
            this->map_cloud_timer_->cancel();
            std_msgs::msg::Header header;
            header.frame_id = this->parameters_.frame;
            header.stamp = now();
            this->map_cloud_pub_->publish(*toROS2msg(this->map_->cloud(), header));
        });

        this->timer_ = create_wall_timer(std::chrono::milliseconds(200), [this] { publish_status(); });

        this->processing_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        const auto interval = std::max(std::chrono::nanoseconds(1),
                                       std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::duration<double>(1.0 / this->parameters_.runtime.max_rate)));
        this->processing_timer_ = create_wall_timer(
            std::min(interval, std::chrono::nanoseconds(std::chrono::milliseconds(20))), [this] { work(); },
            this->processing_group_);
    }

    // Stops timers and waits for in-flight processing before logging timings.
    ~PriorMapLocalizationNode() override {
        this->processing_timer_->cancel();
        this->timer_->cancel();
        this->map_cloud_timer_->cancel();
        std::lock_guard processing_lock(this->processing_mutex_);
        std::lock_guard state_lock(this->mutex_);
        log_processing_times();
    }

private:
    struct Pair {
        CloudMsg::ConstSharedPtr cloud;
        OdomMsg::ConstSharedPtr odom;
        int64_t time;
        size_t epoch;
    };
    struct Waiting {
        CloudMsg::ConstSharedPtr cloud;
        Clock::time_point received;
    };
    struct SensorRewindCandidate {
        int64_t first, last;
        size_t count;
        Clock::time_point started, latest_arrival;
    };
    struct Parameters {
        // SYCL device selection.
        std::string device_vendor;
        std::string device_type;

        // Frames and initial pose.
        std::string frame;
        std::string odom_frame;
        std::string base_frame;
        Eigen::Isometry3f initial_pose;

        pipeline::PriorMapOptions map;
        algorithms::registration::RegistrationParams registration;

        // Branch-and-bound relocalization.
        bool bnb_enable = false;
        algorithms::localization::BranchAndBoundParams branch_and_bound;
        struct BranchAndBoundTriggerParams {
            double lost_timeout = 0.0;
            int max_consecutive_failures = 0;
            double min_interval = 0.0;
        } bnb_trigger;

        // Scalar parameters used after construction.
        struct RuntimeParams {
            // Source point cloud preprocessing.
            int source_max_points = 0;
            double source_voxel_size = 0;
            int source_neighbors = 0;
            int source_min_points = 0;

            // Localization scheduling and freshness.
            double max_rate = 0;
            int max_pending_submaps = 0;
            double pair_wait = 0;
            double submap_wait = 0;
            double frozen_age = 0;
            double odom_age = 0;

            // Registration and quality gates.
            double corr_distance = 0;
            int min_inliers = 0;
            double min_ratio = 0;
            double max_translation = 0;
            double max_rotation = 0;
            double max_error = 0;
            double local_overlap_radius = 0;
            double min_local_overlap_ratio = 0;
            int min_local_points = 0;
        } runtime;
    };

    // Declares and validates all parameters, and applies the values used to build the node.
    Parameters declare_parameters() {
        Parameters params;
        auto& opts = params.map;
        auto& runtime = params.runtime;

        // SYCL device selection.
        params.device_vendor = declare_parameter<std::string>("sycl/device_vendor", "intel");
        params.device_type = declare_parameter<std::string>("sycl/device_type", "gpu");

        // Frames and initial pose.
        params.frame = declare_parameter<std::string>("map/frame_id", "map");
        params.odom_frame = declare_parameter<std::string>("odom/frame_id", "odom");
        params.base_frame = declare_parameter<std::string>("base/frame_id", "base_link");
        params.initial_pose = declare_pose("initial_pose");

        // Prior map construction.
        opts.file_pose = declare_pose("map_file_pose");
        opts.downsample = declare_parameter<bool>("map/downsampling/enable", true);
        opts.voxel_size = declare_parameter<double>("map/downsampling/voxel_size", 0.4);
        opts.covariance_mode = declare_parameter<std::string>("map/covariance/mode", "AUTO");
        opts.eigenvalue_floor = declare_parameter<double>("map/covariance/eigenvalue_floor", 1e-4);
        opts.covariance_neighbors = declare_parameter<int>("map/covariance/neighbor_num", 20);
        opts.local_radius = declare_parameter<double>("map/local_target/radius", 60);
        opts.prediction_margin = declare_parameter<double>("map/local_target/prediction_margin", 3);
        const int local_min_points = declare_parameter<int>("map/local_target/min_points", 50);

        // Source point cloud preprocessing.
        runtime.source_max_points = declare_parameter<int>("source/max_points", 6000);
        runtime.source_voxel_size = declare_parameter<double>("source/voxel_size", 0.4);
        runtime.source_neighbors = declare_parameter<int>("source/covariance/neighbor_num", 20);
        runtime.source_min_points = declare_parameter<int>("source/min_points", 50);

        // Localization scheduling and freshness.
        runtime.max_rate = declare_parameter<double>("localization/max_rate_hz", 1);
        runtime.max_pending_submaps = declare_parameter<int>("localization/max_pending_submaps", 2);
        runtime.pair_wait = declare_parameter<double>("localization/max_pair_wait_sec", 0.5);
        runtime.submap_wait = declare_parameter<double>("localization/max_submap_wait_sec", 3);
        runtime.frozen_age = declare_parameter<double>("localization/max_frozen_tf_age_sec", 5);
        runtime.odom_age = declare_parameter<double>("localization/max_odom_age_sec", 1);

        // Registration and quality gates.
        params.registration = declare_registration_parameters();
        runtime.corr_distance = params.registration.max_correspondence_distance;
        runtime.min_inliers = declare_parameter<int>("registration/min_correspondences", 30);
        runtime.min_ratio = declare_parameter<double>("registration/min_correspondence_ratio", 0.2);
        runtime.max_translation = declare_parameter<double>("registration/max_translation_correction", 3);
        runtime.max_rotation = declare_parameter<double>("registration/max_rotation_correction_rad", 0.5);
        runtime.max_error = declare_parameter<double>("registration/max_mean_error", 10);
        runtime.local_overlap_radius = declare_parameter<double>("registration/local_overlap_radius", 10);
        runtime.min_local_overlap_ratio = declare_parameter<double>("registration/min_local_overlap_ratio", 0.2);
        runtime.min_local_points = declare_parameter<int>("registration/min_local_points", 10);

        // Branch-and-bound relocalization.
        params.bnb_enable = declare_parameter<bool>("branch_and_bound/enable", false);
        auto& bnb = params.branch_and_bound;
        bnb.min_resolution = static_cast<float>(
            declare_parameter<double>("branch_and_bound/min_resolution", static_cast<double>(bnb.min_resolution)));
        bnb.max_level = declare_parameter<int>("branch_and_bound/max_level", bnb.max_level);
        bnb.branch_factor = declare_parameter<int>("branch_and_bound/branch_factor", bnb.branch_factor);
        bnb.correspondence_distance = static_cast<float>(declare_parameter<double>(
            "branch_and_bound/correspondence_distance", static_cast<double>(bnb.correspondence_distance)));
        bnb.radius_scale = static_cast<float>(
            declare_parameter<double>("branch_and_bound/radius_scale", static_cast<double>(bnb.radius_scale)));
        bnb.score_threshold_ratio = static_cast<float>(declare_parameter<double>(
            "branch_and_bound/score_threshold_ratio", static_cast<double>(bnb.score_threshold_ratio)));
        const int64_t score_batch_size = declare_parameter<int64_t>(
            "branch_and_bound/score_batch_size", static_cast<int64_t>(bnb.score_batch_size));
        bnb.auto_translation_range =
            declare_parameter<bool>("branch_and_bound/auto_translation_range", bnb.auto_translation_range);
        bnb.enable_timeout = declare_parameter<bool>("branch_and_bound/enable_timeout", bnb.enable_timeout);
        const int64_t timeout_ms =
            declare_parameter<int64_t>("branch_and_bound/timeout_ms", static_cast<int64_t>(bnb.timeout.count()));
        const char* xyz[3] = {"x", "y", "z"};
        const char* rpy[3] = {"roll", "pitch", "yaw"};
        for (int axis = 0; axis < 3; ++axis) {
            bnb.min_translation[axis] = static_cast<float>(declare_parameter<double>(
                std::string("branch_and_bound/min_translation/") + xyz[axis],
                static_cast<double>(bnb.min_translation[axis])));
            bnb.max_translation[axis] = static_cast<float>(declare_parameter<double>(
                std::string("branch_and_bound/max_translation/") + xyz[axis],
                static_cast<double>(bnb.max_translation[axis])));
            bnb.min_rpy[axis] = static_cast<float>(declare_parameter<double>(
                std::string("branch_and_bound/min_rpy/") + rpy[axis], static_cast<double>(bnb.min_rpy[axis])));
            bnb.max_rpy[axis] = static_cast<float>(declare_parameter<double>(
                std::string("branch_and_bound/max_rpy/") + rpy[axis], static_cast<double>(bnb.max_rpy[axis])));
        }
        params.bnb_trigger.lost_timeout = declare_parameter<double>("branch_and_bound/trigger/lost_timeout_sec", 5.0);
        params.bnb_trigger.max_consecutive_failures =
            declare_parameter<int>("branch_and_bound/trigger/max_consecutive_failures", 3);
        params.bnb_trigger.min_interval =
            declare_parameter<double>("branch_and_bound/trigger/min_interval_sec", 2.0);
        if (score_batch_size <= 0 || timeout_ms < 0) {
            throw std::invalid_argument("Invalid branch-and-bound parameters");
        }
        bnb.score_batch_size = static_cast<size_t>(score_batch_size);
        bnb.timeout = std::chrono::milliseconds(timeout_ms);

        // Validation.
        if (runtime.source_max_points < 4 || runtime.source_min_points < 4 || runtime.source_neighbors < 4 ||
            runtime.source_neighbors > 100 || runtime.source_neighbors > runtime.source_max_points ||
            runtime.max_pending_submaps < 1 || runtime.min_inliers < 1 || runtime.max_rate <= 0 ||
            runtime.pair_wait <= 0 || runtime.submap_wait <= 0 || runtime.frozen_age <= 0 || runtime.odom_age <= 0 ||
            runtime.corr_distance <= 0 || runtime.source_voxel_size <= 0 || runtime.min_ratio < 0 ||
            runtime.min_ratio > 1 || runtime.max_translation <= 0 || runtime.max_rotation <= 0 ||
            runtime.max_error <= 0 || runtime.local_overlap_radius <= 0 || runtime.min_local_points < 1 ||
            local_min_points < 1 || opts.prediction_margin < runtime.max_translation ||
            runtime.min_local_overlap_ratio < 0 || runtime.min_local_overlap_ratio > 1 ||
            params.bnb_trigger.lost_timeout <= 0 || params.bnb_trigger.max_consecutive_failures < 1 ||
            params.bnb_trigger.min_interval < 0) {
            throw std::invalid_argument("Invalid localization parameters");
        }
        opts.local_min_points = static_cast<size_t>(local_min_points);

        return params;
    }

    Eigen::Isometry3f declare_pose(const std::string& prefix) {
        geometry_msgs::msg::Pose p;
        p.position.x = declare_parameter<double>(prefix + "/x", 0.0);
        p.position.y = declare_parameter<double>(prefix + "/y", 0.0);
        p.position.z = declare_parameter<double>(prefix + "/z", 0.0);
        p.orientation.x = declare_parameter<double>(prefix + "/qx", 0.0);
        p.orientation.y = declare_parameter<double>(prefix + "/qy", 0.0);
        p.orientation.z = declare_parameter<double>(prefix + "/qz", 0.0);
        p.orientation.w = declare_parameter<double>(prefix + "/qw", 1.0);
        return pose(p);
    }

    // Publishes the corrected map->odom transform and localized pose.
    // Caller holds mutex_. Never publish a correction for an already published timestamp.
    void publish_tf_locked() {
        if (!this->correction_ || !this->latest_odom_ || !this->last_success_ ||
            (now() - *this->last_success_).seconds() > this->parameters_.runtime.frozen_age ||
            std::chrono::duration<double>(Clock::now() - this->last_odom_arrival_).count() >
                this->parameters_.runtime.odom_age) {
            return;
        }
        const auto t = stamp(this->latest_odom_->header.stamp);
        if (t <= this->last_tf_stamp_) {
            return;
        }
        geometry_msgs::msg::PoseStamped located;
        located.header = this->latest_odom_->header;
        located.header.frame_id = this->parameters_.frame;
        located.pose = to_msg(*this->correction_ * pose(this->latest_odom_->pose.pose));
        geometry_msgs::msg::TransformStamped msg;
        msg.header = this->latest_odom_->header;
        msg.header.frame_id = this->parameters_.frame;
        msg.child_frame_id = this->parameters_.odom_frame;
        const auto p = to_msg(*this->correction_);
        msg.transform.translation.x = p.position.x;
        msg.transform.translation.y = p.position.y;
        msg.transform.translation.z = p.position.z;
        msg.transform.rotation = p.orientation;
        this->tf_->sendTransform(msg);
        this->last_tf_stamp_ = t;
        this->pose_pub_->publish(located);
    }

    // Clears all localization state after a clock or sensor-time rewind. Caller holds mutex_.
    void rewind_locked() {
        this->pending_pair_.reset();
        this->waiting_.clear();
        this->odoms_.clear();
        ++this->epoch_;
        this->correction_.reset();
        this->initial_correction_.reset();
        this->last_success_.reset();
        this->consecutive_failures_ = 0;
        this->last_bnb_attempt_.reset();
        this->last_accepted_ = std::numeric_limits<int64_t>::min();
        this->latest_odom_.reset();
        this->sensor_rewind_candidate_.reset();
        this->last_tf_stamp_ = std::numeric_limits<int64_t>::min();
        this->last_cloud_.reset();
        this->reason_ = "clock rewind";
        this->inliers_ = 0;
        this->error_ = 0;
        this->local_overlap_ = 0;
        this->time_sync_warned_ = false;
        this->odom_frame_warned_ = false;
        this->cloud_frame_warned_ = false;
        this->last_odom_arrival_ = Clock::now();
        this->last_cloud_arrival_ = Clock::now();
    }

    // Resets localization when the ROS clock moves backwards. Caller holds mutex_.
    void check_clock_locked(const rclcpp::Time& current) {
        // Compare ROS time to its previous value, not to a possibly future sensor stamp.
        if (this->last_ros_time_ && (*this->last_ros_time_ - current).seconds() > 0.5) {
            RCLCPP_WARN(get_logger(), "ROS clock moved backwards; resetting localization");
            rewind_locked();
        }
        this->last_ros_time_ = current;
    }

    // Stores the newest cloud/odometry pair for processing. Pairs older than the last accepted
    // observation or the pending one are skipped, so out-of-order arrivals cannot displace newer
    // data. Caller holds mutex_.
    void pair_locked(int64_t t, const CloudMsg::ConstSharedPtr& cloud, const OdomMsg::ConstSharedPtr& odom) {
        if (t <= this->last_accepted_) {
            return;
        }
        if (this->pending_pair_ && this->pending_pair_->time >= t) {
            return;
        }
        this->pending_pair_ = Pair{cloud, odom, t, this->epoch_};
    }

    // Validates and stores odometry, pairs it with a waiting cloud, and publishes the latest TF.
    void on_odom(OdomMsg::ConstSharedPtr msg) {
        if (msg->header.frame_id != this->parameters_.odom_frame ||
            msg->child_frame_id != this->parameters_.base_frame) {
            std::lock_guard lock(this->mutex_);
            if (!this->odom_frame_warned_) {
                RCLCPP_WARN(get_logger(),
                            "Skipping odometry with unexpected frames: '%s' -> '%s' (expected '%s' -> '%s')",
                            msg->header.frame_id.c_str(), msg->child_frame_id.c_str(),
                            this->parameters_.odom_frame.c_str(), this->parameters_.base_frame.c_str());
                this->odom_frame_warned_ = true;
            }
            return;
        }
        try {
            (void)pose(msg->pose.pose);
        } catch (const std::invalid_argument& e) {
            std::lock_guard lock(this->mutex_);
            this->reason_ = e.what();
            RCLCPP_WARN(get_logger(), "Skipping invalid odometry: %s", e.what());
            return;
        }
        const auto t = stamp(msg->header.stamp);
        std::lock_guard lock(this->mutex_);
        check_clock_locked(now());
        if (this->latest_odom_ && t < stamp(this->latest_odom_->header.stamp)) {
            constexpr int64_t rewind_threshold_ns = 5000000000LL;
            constexpr int64_t rewind_span_ns = 2000000000LL;
            const auto arrival = Clock::now();
            if (stamp(this->latest_odom_->header.stamp) - t >= rewind_threshold_ns) {
                // Sensor time alone cannot distinguish a replay from late packets. Require
                // sustained progress below the latest stamp without fresh latest odometry;
                // short replays and restarts within five seconds cannot be detected safely.
                if (!this->sensor_rewind_candidate_ ||
                    this->sensor_rewind_candidate_->latest_arrival != this->last_odom_arrival_ ||
                    arrival - this->sensor_rewind_candidate_->started > std::chrono::seconds(5) ||
                    t < this->sensor_rewind_candidate_->last) {
                    this->sensor_rewind_candidate_ = SensorRewindCandidate{t, t, 1, arrival, this->last_odom_arrival_};
                } else if (t > this->sensor_rewind_candidate_->last) {
                    this->sensor_rewind_candidate_->last = t;
                    ++this->sensor_rewind_candidate_->count;
                    if (this->sensor_rewind_candidate_->count >= 3 &&
                        t - this->sensor_rewind_candidate_->first >= rewind_span_ns &&
                        arrival - this->sensor_rewind_candidate_->started >= std::chrono::milliseconds(500)) {
                        RCLCPP_WARN(get_logger(), "Odometry stamps moved backwards; resetting localization");
                        rewind_locked();
                        this->reason_ = "sensor stamp rewind";
                    }
                }
            } else {
                this->sensor_rewind_candidate_.reset();
            }
        } else {
            this->sensor_rewind_candidate_.reset();
        }
        if (this->latest_odom_ && t < stamp(this->latest_odom_->header.stamp)) {
            // Keep a late odometry message available to pair with its scan,
            // without moving the latest TF publication timestamp backwards.
            this->odoms_[t] = msg;
            if (auto it = this->waiting_.find(t); it != this->waiting_.end()) {
                pair_locked(t, it->second.cloud, msg);
                this->waiting_.erase(it);
            }
            while (this->odoms_.size() > 100) {
                this->odoms_.erase(this->odoms_.begin());
            }
            return;
        }
        this->latest_odom_ = msg;
        if (!this->initial_correction_) {
            this->initial_correction_ = this->parameters_.initial_pose * pose(msg->pose.pose).inverse();
        }
        this->last_odom_arrival_ = Clock::now();
        if (!this->time_sync_warned_ &&
            std::abs((now() - rclcpp::Time(msg->header.stamp)).seconds()) > this->parameters_.runtime.frozen_age) {
            RCLCPP_WARN(get_logger(),
                        "Odometry stamp differs from ROS time; check use_sim_time and clock synchronization");
            this->time_sync_warned_ = true;
        }
        this->odoms_[t] = msg;
        if (auto it = this->waiting_.find(t); it != this->waiting_.end()) {
            pair_locked(t, it->second.cloud, msg);
            this->waiting_.erase(it);
        }
        while (this->odoms_.size() > 100) {
            this->odoms_.erase(this->odoms_.begin());
        }
        publish_tf_locked();
    }

    // Ignores unchanged clouds after a successful match; otherwise pairs the cloud with odometry.
    void on_cloud(CloudMsg::ConstSharedPtr msg) {
        if (msg->header.frame_id != this->parameters_.odom_frame) {
            std::lock_guard lock(this->mutex_);
            if (!this->cloud_frame_warned_) {
                RCLCPP_WARN(get_logger(), "Skipping submap with unexpected frame: '%s' (expected '%s')",
                            msg->header.frame_id.c_str(), this->parameters_.odom_frame.c_str());
                this->cloud_frame_warned_ = true;
            }
            return;
        }
        const auto t = stamp(msg->header.stamp);
        for (;;) {
            CloudMsg::ConstSharedPtr previous;
            size_t epoch;
            {
                std::lock_guard lock(this->mutex_);
                previous = this->last_cloud_;
                epoch = this->epoch_;
            }
            const bool unchanged = previous && previous->width == msg->width && previous->height == msg->height &&
                                   previous->fields == msg->fields && previous->is_bigendian == msg->is_bigendian &&
                                   previous->point_step == msg->point_step && previous->row_step == msg->row_step &&
                                   previous->is_dense == msg->is_dense && previous->data == msg->data;
            std::lock_guard lock(this->mutex_);
            if (epoch != this->epoch_) {
                return;
            }
            if (previous != this->last_cloud_) {
                continue;
            }
            if (!unchanged) {
                this->last_cloud_ = msg;
                this->last_cloud_arrival_ = Clock::now();
            }
            // A previous failure can be retried, but identical successful matches must
            // not refresh the TF freshness deadline without a new map observation.
            if (unchanged && this->last_success_ && this->reason_.empty()) {
                return;
            }
            if (const auto it = this->odoms_.find(t); it != this->odoms_.end()) {
                pair_locked(t, msg, it->second);
            } else {
                this->waiting_[t] = {msg, Clock::now()};
            }

            while (this->waiting_.size() > static_cast<size_t>(this->parameters_.runtime.max_pending_submaps)) {
                this->waiting_.erase(this->waiting_.begin());
            }
            return;
        }
    }

    // Drops expired waiting clouds and publishes the current tracking state.
    void publish_status() {
        std_msgs::msg::String status;
        std::lock_guard lock(this->mutex_);
        check_clock_locked(now());
        for (auto it = this->waiting_.begin(); it != this->waiting_.end();) {
            if (std::chrono::duration<double>(Clock::now() - it->second.received).count() >
                this->parameters_.runtime.pair_wait) {
                this->reason_ = "matching odometry missing";
                it = this->waiting_.erase(it);
            } else
                ++it;
        }
        const auto age = this->last_success_ ? (now() - *this->last_success_).seconds() : -1.0;
        const auto odom_elapsed = std::chrono::duration<double>(Clock::now() - this->last_odom_arrival_).count();
        const auto cloud_elapsed = std::chrono::duration<double>(Clock::now() - this->last_cloud_arrival_).count();
        const std::string reason = this->reason_.empty() && cloud_elapsed > this->parameters_.runtime.submap_wait
                                       ? "submap unavailable or unchanged"
                                       : this->reason_;
        const char* state =
            !this->last_success_
                ? "UNINITIALIZED"
                : ((age > this->parameters_.runtime.frozen_age || odom_elapsed > this->parameters_.runtime.odom_age)
                       ? "LOST"
                       : ((cloud_elapsed > this->parameters_.runtime.submap_wait ||
                           age > this->parameters_.runtime.submap_wait || !this->reason_.empty())
                              ? "DEGRADED"
                              : "TRACKING"));
        status.data =
            std::string(state) +
            " last_success_ns=" + std::to_string(this->last_success_ ? this->last_success_->nanoseconds() : 0) +
            " age_sec=" + std::to_string(age) + " inliers=" + std::to_string(this->inliers_) +
            " mean_error=" + std::to_string(this->error_) + " local_overlap=" + std::to_string(this->local_overlap_) +
            " reason=" + reason;
        this->status_pub_->publish(status);
    }

    // Returns true when a global search should run for a new job. Caller holds mutex_.
    bool should_relocalize_locked(const rclcpp::Time& time) const {
        if (!this->bnb_) {
            return false;
        }
        if (this->last_bnb_attempt_ &&
            Clock::now() - *this->last_bnb_attempt_ <
                std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(this->parameters_.bnb_trigger.min_interval))) {
            return false;
        }
        if (!this->correction_) {
            return true;
        }
        if (this->consecutive_failures_ >= this->parameters_.bnb_trigger.max_consecutive_failures) {
            return true;
        }
        return this->last_success_ &&
               (time - *this->last_success_).seconds() > this->parameters_.bnb_trigger.lost_timeout;
    }

    // Pops the newest pending pair at the configured rate and runs registration.
    void work() {
        std::lock_guard processing_lock(this->processing_mutex_);
        Pair job;
        {
            std::lock_guard lock(this->mutex_);
            const auto current = Clock::now();
            if (!this->pending_pair_ || current < this->next_processing_) return;
            // A newer frame may have completed while this one was queued.
            if (this->pending_pair_->time <= this->last_accepted_) {
                this->pending_pair_.reset();
                return;
            }
            job = *this->pending_pair_;
            this->pending_pair_.reset();
            this->next_processing_ =
                current + std::chrono::duration_cast<Clock::duration>(
                              std::chrono::duration<double>(1.0 / this->parameters_.runtime.max_rate));
        }
        try {
            process(job);
        } catch (const std::exception& e) {
            std::lock_guard lock(this->mutex_);
            if (job.epoch != this->epoch_) return;
            ++this->consecutive_failures_;
            this->reason_ = e.what();
            RCLCPP_WARN(get_logger(), "Map matching failed: %s", e.what());
        }
    }

    void print_processing_time(const std::string& name, double us) const {
        constexpr size_t width = 24;
        RCLCPP_INFO(get_logger(), "%s: %*s%9.2f us", name.c_str(),
                    static_cast<int>(name.size() < width ? width - name.size() : 0), "", us);
    }

    // Records and prints the duration of one processing stage.
    void record_processing_time(const std::string& name, Clock::time_point begin, Clock::time_point end) {
        const double us = std::chrono::duration<double, std::micro>(end - begin).count();
        this->processing_times_[name].push_back(us);
        print_processing_time(name, us);
    }

    // Prints max, mean, and median durations for each processing stage.
    void log_processing_times() {
        int statistic = 0;
        for (const char* heading : {"MAX processing time", "MEAN processing time", "MEDIAN processing time"}) {
            RCLCPP_INFO(get_logger(), "");
            RCLCPP_INFO(get_logger(), "%s", heading);
            for (auto& [name, times] : this->processing_times_) {
                if (times.empty()) continue;
                double value;
                if (statistic == 0) {
                    value = *std::max_element(times.begin(), times.end());
                } else if (statistic == 1) {
                    value = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                } else {
                    std::sort(times.begin(), times.end());
                    value = times[times.size() / 2];
                }
                print_processing_time(name, value);
            }
            ++statistic;
        }
        RCLCPP_INFO(get_logger(), "");
    }

    // All RegistrationParams fields are exposed under `registration/*`.
    algorithms::registration::RegistrationParams declare_registration_parameters() {
        using algorithms::registration::OptimizationMethod_from_string;
        using algorithms::registration::RegType_from_string;
        using algorithms::robust::RobustLossType_from_string;

        algorithms::registration::RegistrationParams params;
        params.reg_type = RegType_from_string(declare_parameter<std::string>("registration/type", "gicp"));
        params.max_correspondence_distance =
            declare_parameter<double>("registration/max_correspondence_distance", params.max_correspondence_distance);
        const int64_t max_iterations =
            declare_parameter<int64_t>("registration/max_iterations", static_cast<int64_t>(params.max_iterations));
        params.max_iterations = static_cast<size_t>(max_iterations);
        params.criteria.translation =
            declare_parameter<double>("registration/criteria/translation", params.criteria.translation);
        params.criteria.rotation =
            declare_parameter<double>("registration/criteria/rotation", params.criteria.rotation);
        params.verbose = declare_parameter<bool>("registration/verbose", params.verbose);

        params.robust.type =
            RobustLossType_from_string(declare_parameter<std::string>("registration/robust/type", "NONE"));
        params.robust.default_scale =
            declare_parameter<double>("registration/robust/default_scale", params.robust.default_scale);

        params.rotation_constraint.enable =
            declare_parameter<bool>("registration/rotation_constraint/enable", params.rotation_constraint.enable);
        params.rotation_constraint.weight =
            declare_parameter<double>("registration/rotation_constraint/weight", params.rotation_constraint.weight);
        params.rotation_constraint.robust.default_scale = declare_parameter<double>(
            "registration/rotation_constraint/robust/default_scale", params.rotation_constraint.robust.default_scale);

        params.genz.planarity_threshold =
            declare_parameter<double>("registration/genz/planarity_threshold", params.genz.planarity_threshold);

        params.optimization_method =
            OptimizationMethod_from_string(declare_parameter<std::string>("registration/optimization_method", "GN"));
        params.gn.lambda = declare_parameter<double>("registration/gn/lambda", params.gn.lambda);
        const int64_t lm_max_inner_iterations = declare_parameter<int64_t>(
            "registration/lm/max_inner_iterations", static_cast<int64_t>(params.lm.max_inner_iterations));
        params.lm.max_inner_iterations = static_cast<size_t>(lm_max_inner_iterations);
        params.lm.lambda_factor = declare_parameter<double>("registration/lm/lambda_factor", params.lm.lambda_factor);
        params.lm.init_lambda = declare_parameter<double>("registration/lm/init_lambda", params.lm.init_lambda);
        params.lm.max_lambda = declare_parameter<double>("registration/lm/max_lambda", params.lm.max_lambda);
        params.lm.min_lambda = declare_parameter<double>("registration/lm/min_lambda", params.lm.min_lambda);
        params.dogleg.initial_trust_region_radius = declare_parameter<double>(
            "registration/dogleg/initial_trust_region_radius", params.dogleg.initial_trust_region_radius);
        params.dogleg.min_trust_region_radius = declare_parameter<double>("registration/dogleg/min_trust_region_radius",
                                                                          params.dogleg.min_trust_region_radius);
        params.dogleg.max_trust_region_radius = declare_parameter<double>("registration/dogleg/max_trust_region_radius",
                                                                          params.dogleg.max_trust_region_radius);
        params.dogleg.eta1 = declare_parameter<double>("registration/dogleg/eta1", params.dogleg.eta1);
        params.dogleg.eta2 = declare_parameter<double>("registration/dogleg/eta2", params.dogleg.eta2);
        params.dogleg.gamma_decrease =
            declare_parameter<double>("registration/dogleg/gamma_decrease", params.dogleg.gamma_decrease);
        params.dogleg.gamma_increase =
            declare_parameter<double>("registration/dogleg/gamma_increase", params.dogleg.gamma_increase);

        params.degenerate_reg.type = algorithms::registration::DegenerateRegularizationType::none;
        params.map_prior.enabled = false;

        if (params.max_correspondence_distance <= 0 || params.max_iterations < 1 || params.criteria.translation <= 0 ||
            params.criteria.rotation <= 0 || params.robust.default_scale <= 0 ||
            params.rotation_constraint.weight < 0 || params.rotation_constraint.robust.default_scale <= 0 ||
            params.genz.planarity_threshold < 0 || params.genz.planarity_threshold > 1 || params.gn.lambda <= 0 ||
            params.lm.max_inner_iterations < 1 || params.lm.lambda_factor <= 1 || params.lm.init_lambda <= 0 ||
            params.lm.min_lambda <= 0 || params.lm.max_lambda < params.lm.min_lambda ||
            params.dogleg.min_trust_region_radius <= 0 ||
            params.dogleg.initial_trust_region_radius < params.dogleg.min_trust_region_radius ||
            params.dogleg.max_trust_region_radius < params.dogleg.initial_trust_region_radius ||
            params.dogleg.eta1 < 0 || params.dogleg.eta1 > params.dogleg.eta2 || params.dogleg.eta2 > 1 ||
            params.dogleg.gamma_decrease <= 0 || params.dogleg.gamma_decrease >= 1 ||
            params.dogleg.gamma_increase <= 1) {
            throw std::invalid_argument("Invalid registration parameters");
        }
        return params;
    }

    // Converts a submap, matches it against the local prior map, and applies the quality gates.
    void process(const Pair& job) {
        // Reject stale jobs and use the latest correction as the initial guess.
        Eigen::Isometry3f guess;
        bool relocalize = false;
        {
            std::lock_guard lock(this->mutex_);
            if (job.epoch != this->epoch_ || job.time <= this->last_accepted_ ||
                (this->latest_odom_ && job.time > stamp(this->latest_odom_->header.stamp)))
                return;
            guess = this->correction_.value_or(*this->initial_correction_);
            relocalize = this->should_relocalize_locked(now());
            if (relocalize) {
                this->last_bnb_attempt_ = Clock::now();
            }
        }
        const auto started = Clock::now();

        // Convert the whole submap on the device. Non-finite points are dropped by VoxelGrid.
        if (!fromROS2msg(*this->queue_, *job.cloud, this->source_pc_, this->msg_data_buffer_, false, false, false) ||
            this->source_pc_->size() == 0) {
            throw std::runtime_error("failed to convert submap");
        }
        if (this->source_pc_->size() < static_cast<size_t>(this->parameters_.runtime.source_min_points)) {
            throw std::runtime_error("source too small");
        }
        const auto converted = Clock::now();

        // Downsample on the device; this also removes non-finite points.
        this->voxel_->downsampling(*this->source_pc_, *this->source_pc_);
        if (this->source_pc_->size() < static_cast<size_t>(std::max(this->parameters_.runtime.source_min_points,
                                                                    this->parameters_.runtime.source_neighbors))) {
            throw std::runtime_error("downsampled source too small");
        }
        const auto reduced = Clock::now();

        // Build the source KD-tree and estimate covariances over the whole cloud.
        const auto source_tree = algorithms::knn::KDTree::build(*this->queue_, *this->source_pc_);
        const auto indexed = Clock::now();
        algorithms::covariance::estimate_async(*source_tree, *this->source_pc_,
                                               this->parameters_.runtime.source_neighbors)
            .wait_and_throw();
        const auto covariances = Clock::now();

        // Randomly sample the registration input after covariance estimation.
        this->preprocess_filter_->random_sampling(*this->source_pc_, *this->registration_pc_,
                                                  static_cast<size_t>(this->parameters_.runtime.source_max_points));
        if (this->registration_pc_->size() <
            static_cast<size_t>(
                std::max(this->parameters_.runtime.source_min_points, this->parameters_.runtime.source_neighbors))) {
            throw std::runtime_error("sampled source too small");
        }
        const auto sampled = Clock::now();

        // Run a global search when tracking is not established. The scan stays in the odom frame,
        // and the search estimates the base-link pose.
        if (relocalize) {
            const Eigen::Isometry3f base_from_odom = pose(job.odom->pose.pose).inverse();
            try {
                const auto bnb = this->bnb_->localize(*this->source_pc_, base_from_odom);
                RCLCPP_INFO(get_logger(),
                            "Branch and bound: converged=%d inlier=%u score_ratio=%.3f elapsed_ms=%.2f "
                            "candidates=%zu timed_out=%d",
                            static_cast<int>(bnb.converged), bnb.inlier, bnb.score_ratio, bnb.elapsed_ms,
                            bnb.evaluated_candidates, static_cast<int>(bnb.timed_out));
                if (bnb.converged) {
                    guess = Eigen::Isometry3f(bnb.T.matrix() * base_from_odom.matrix());
                }
            } catch (const std::exception& e) {
                RCLCPP_WARN(get_logger(), "Branch and bound failed: %s", e.what());
            }
        }
        const auto searched = Clock::now();

        // Select the local prior map that covers the predicted source extent.
        const auto center = (guess * pose(job.odom->pose.pose)).translation();
        float reach = 0;
        for (const auto& p : *this->registration_pc_->points) {
            reach = std::max(reach, (guess * p.head<3>() - center).norm());
        }
        const auto target = this->map_->target(center, reach + this->parameters_.runtime.corr_distance);
        if (!target) {
            throw std::runtime_error("local map insufficient or required extent exceeds radius");
        }
        const auto localized = Clock::now();

        // Run registration and reject results that fail the quality gates.
        const auto result =
            this->registration_->align(*this->registration_pc_, *target->cloud, *target->tree, guess.matrix());
        const auto matched = Clock::now();
        const auto delta = guess.inverse() * result.T;
        const float mean_error = result.inlier ? result.error / result.inlier : INFINITY;
        if (!result.T.matrix().allFinite() || !std::isfinite(mean_error) ||
            result.inlier < static_cast<size_t>(this->parameters_.runtime.min_inliers) ||
            static_cast<double>(result.inlier) / this->registration_pc_->size() < this->parameters_.runtime.min_ratio ||
            delta.translation().norm() > this->parameters_.runtime.max_translation ||
            Eigen::AngleAxisf(delta.rotation()).angle() > this->parameters_.runtime.max_rotation ||
            mean_error > this->parameters_.runtime.max_error) {
            throw std::runtime_error("registration rejected by quality gates");
        }

        // Verify that the map near the current robot position actually overlaps the source.
        // Older submap points can match a map even after the robot leaves it.
        target->tree->nearest_neighbor_search(*this->registration_pc_, this->neighbors_, {}, result.T.matrix());
        const auto current_odom_position = pose(job.odom->pose.pose).translation();
        size_t nearby = 0, nearby_matched = 0;
        for (size_t i = 0; i < this->registration_pc_->size(); ++i) {
            if (((*this->registration_pc_->points)[i].head<3>() - current_odom_position).squaredNorm() >
                this->parameters_.runtime.local_overlap_radius * this->parameters_.runtime.local_overlap_radius)
                continue;
            ++nearby;
            if ((*this->neighbors_.distances)[i] <=
                this->parameters_.runtime.corr_distance * this->parameters_.runtime.corr_distance)
                ++nearby_matched;
        }
        const float local_overlap = nearby ? static_cast<float>(nearby_matched) / nearby : 0.0f;
        if (nearby < static_cast<size_t>(this->parameters_.runtime.min_local_points) ||
            local_overlap < this->parameters_.runtime.min_local_overlap_ratio) {
            throw std::runtime_error("insufficient map overlap near current robot position");
        }
        const auto checked = Clock::now();

        // Commit the accepted correction unless a newer job or rewind superseded it.
        {
            std::lock_guard lock(this->mutex_);
            if (job.epoch != this->epoch_ || job.time <= this->last_accepted_ ||
                (this->latest_odom_ && job.time > stamp(this->latest_odom_->header.stamp))) {
                return;
            }
            this->correction_ = result.T;
            this->last_accepted_ = job.time;
            this->last_success_ = rclcpp::Time(job.cloud->header.stamp);
            this->inliers_ = result.inlier;
            this->error_ = mean_error;
            this->local_overlap_ = local_overlap;
            this->consecutive_failures_ = 0;
            this->reason_.clear();
            if ((now() - *this->last_success_).seconds() > this->parameters_.runtime.frozen_age) {
                this->reason_ = "registration completed after TF freshness deadline";
                RCLCPP_WARN(get_logger(), "%s", this->reason_.c_str());
            }
            publish_tf_locked();
        }
        const auto published = Clock::now();

        // Report correspondences and per-stage timings.
        {
            RCLCPP_INFO(get_logger(), "Matched %zu correspondences", static_cast<size_t>(result.inlier));
            record_processing_time("0. from ROS 2 msg", started, converted);
            record_processing_time("1. downsampling", converted, reduced);
            record_processing_time("2. source tree", reduced, indexed);
            record_processing_time("3. covariance", indexed, covariances);
            record_processing_time("4. random sampling", covariances, sampled);
            record_processing_time("5. branch and bound", sampled, searched);
            record_processing_time("6. local target", searched, localized);
            record_processing_time("7. registration", localized, matched);
            record_processing_time("8. overlap check", matched, checked);
            record_processing_time("9. publish ROS 2 msg", checked, published);
            record_processing_time("10. total", started, published);
            RCLCPP_INFO(get_logger(), "");
        }
    }

    // Core components.
    std::unique_ptr<sycl_utils::DeviceQueue> queue_;
    std::unique_ptr<pipeline::PriorMap> map_;
    std::unique_ptr<algorithms::registration::Registration> registration_;
    std::unique_ptr<algorithms::localization::BranchAndBoundLocalizer> bnb_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;

    // Reused conversion, sampling, and point cloud buffers.
    PointCloudShared::Ptr source_pc_ = nullptr;
    PointCloudShared::Ptr registration_pc_ = nullptr;
    shared_vector_ptr<uint8_t> msg_data_buffer_ = nullptr;
    algorithms::filter::PreprocessFilter::Ptr preprocess_filter_ = nullptr;
    algorithms::filter::VoxelGrid::Ptr voxel_ = nullptr;
    algorithms::knn::KNNResult neighbors_;

    // ROS 2 interfaces.
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<CloudMsg>::SharedPtr map_cloud_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::CallbackGroup::SharedPtr odom_group_;
    rclcpp::Subscription<OdomMsg>::SharedPtr odom_sub_;
    rclcpp::Subscription<CloudMsg>::SharedPtr cloud_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_cloud_timer_;
    rclcpp::CallbackGroup::SharedPtr processing_group_;
    rclcpp::TimerBase::SharedPtr processing_timer_;

    // Status and pose correction.
    std::string reason_;
    std::optional<Eigen::Isometry3f> correction_;
    std::optional<Eigen::Isometry3f> initial_correction_;
    std::optional<rclcpp::Time> last_success_;
    std::optional<rclcpp::Time> last_ros_time_;

    // One-shot warning flags.
    bool time_sync_warned_ = false;
    bool odom_frame_warned_ = false, cloud_frame_warned_ = false;

    // Latest sensor arrivals.
    OdomMsg::ConstSharedPtr latest_odom_;
    Clock::time_point last_odom_arrival_ = Clock::now(), last_cloud_arrival_ = Clock::now();

    // Mutual exclusion.
    std::mutex mutex_;
    std::mutex processing_mutex_;

    // Processing scheduling.
    Clock::time_point next_processing_ = Clock::time_point::min();

    // Latest paired observation awaiting processing, and messages waiting for pairing.
    std::optional<Pair> pending_pair_;
    std::map<int64_t, OdomMsg::ConstSharedPtr> odoms_;
    std::map<int64_t, Waiting> waiting_;

    // Branch-and-bound scheduling.
    int consecutive_failures_ = 0;
    std::optional<Clock::time_point> last_bnb_attempt_;

    // Output deduplication and sensor rewind detection.
    int64_t last_tf_stamp_ = std::numeric_limits<int64_t>::min();
    int64_t last_accepted_ = std::numeric_limits<int64_t>::min();
    std::optional<SensorRewindCandidate> sensor_rewind_candidate_;
    size_t epoch_ = 0;
    CloudMsg::ConstSharedPtr last_cloud_;

    // Latest registration diagnostics.
    size_t inliers_ = 0;
    float error_ = 0;
    float local_overlap_ = 0;
    std::map<std::string, std::vector<double>> processing_times_;

    // Declared parameters (immutable after construction).
    const Parameters parameters_;
};

}  // namespace sycl_points::ros2

RCLCPP_COMPONENTS_REGISTER_NODE(sycl_points::ros2::PriorMapLocalizationNode)
