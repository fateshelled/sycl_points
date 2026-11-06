#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <sycl_points/algorithms/registration/degenerate_regularization.hpp>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "sycl_points/algorithms/registration/registration.hpp"
#include "sycl_points/pipeline/prior_map.hpp"
#include "sycl_points/ros2/convert.hpp"

namespace sycl_points::ros2 {
namespace {
using CloudMsg = sensor_msgs::msg::PointCloud2;
using OdomMsg = nav_msgs::msg::Odometry;
using Clock = std::chrono::steady_clock;

int64_t stamp(const builtin_interfaces::msg::Time& t) {
    return static_cast<int64_t>(t.sec) * 1000000000LL + t.nanosec;
}

Eigen::Isometry3f pose(const geometry_msgs::msg::Pose& p) {
    Eigen::Quaternionf q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
    if (!q.coeffs().allFinite() || q.norm() < 1e-6f) throw std::invalid_argument("Invalid pose quaternion");
    Eigen::Isometry3f result = Eigen::Isometry3f::Identity();
    result.linear() = q.normalized().toRotationMatrix();
    result.translation() = Eigen::Vector3f(p.position.x, p.position.y, p.position.z);
    if (!result.matrix().allFinite()) throw std::invalid_argument("Invalid pose translation");
    return result;
}

geometry_msgs::msg::Pose to_msg(const Eigen::Isometry3f& t) {
    geometry_msgs::msg::Pose p;
    p.position.x = t.translation().x();
    p.position.y = t.translation().y();
    p.position.z = t.translation().z();
    Eigen::Quaternionf q(t.rotation());
    p.orientation.x = q.x(); p.orientation.y = q.y(); p.orientation.z = q.z(); p.orientation.w = q.w();
    return p;
}
}  // namespace

class PriorMapLocalizationNode : public rclcpp::Node {
public:
    explicit PriorMapLocalizationNode(const rclcpp::NodeOptions& options)
        : Node("prior_map_localization", options), parameters_(declare_parameters()) {
        queue_ = std::make_unique<sycl_utils::DeviceQueue>(
            sycl::device(sycl_utils::device_selector::default_selector_v));
        const auto map_path = declare_parameter<std::string>("map/path", "");
        if (map_path.empty()) throw std::invalid_argument("map/path must point to a PCD or PLY file");
        map_ = std::make_unique<pipeline::PriorMap>(*queue_, map_path, parameters_.map);
        RCLCPP_INFO(get_logger(), "Prior map: %zu points, %zu covariance eigenvalues clamped",
                    map_->size(), map_->covariance_clamp_count());
        map_cloud_pub_ = create_publisher<CloudMsg>("localization/prior_map",
            rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());
        // Publish the full prior map after the executor starts, then stop the timer.
        map_cloud_timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] {
            map_cloud_timer_->cancel();
            std_msgs::msg::Header header;
            header.frame_id = parameters_.frame;
            header.stamp = now();
            map_cloud_pub_->publish(*toROS2msg(map_->cloud(), header));
        });
        registration_ =
            std::make_unique<algorithms::registration::Registration>(*queue_, parameters_.registration);
        tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("localization/pose", 5);
        status_pub_ = create_publisher<std_msgs::msg::String>("localization/status", 5);
        odom_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions odom_options;
        odom_options.callback_group = odom_group_;
        odom_sub_ = create_subscription<OdomMsg>("sycl_lio/odom", rclcpp::QoS(5),
            [this](OdomMsg::ConstSharedPtr msg) { on_odom(std::move(msg)); }, odom_options);
        cloud_sub_ = create_subscription<CloudMsg>("sycl_lio/submap", rclcpp::QoS(5),
            [this](CloudMsg::ConstSharedPtr msg) { on_cloud(std::move(msg)); });
        timer_ = create_wall_timer(std::chrono::milliseconds(200), [this] { publish_status(); });
        processing_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        const auto interval = std::max(std::chrono::nanoseconds(1),
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / parameters_.runtime.max_rate)));
        processing_timer_ = create_wall_timer(std::min(interval, std::chrono::nanoseconds(std::chrono::milliseconds(20))),
            [this] { work(); }, processing_group_);
    }

    ~PriorMapLocalizationNode() override {
        processing_timer_->cancel();
        timer_->cancel();
        map_cloud_timer_->cancel();
        std::lock_guard processing_lock(processing_mutex_);
        std::lock_guard state_lock(mutex_);
        log_processing_times();
    }

private:
    struct Pair { CloudMsg::ConstSharedPtr cloud; OdomMsg::ConstSharedPtr odom; int64_t time; size_t epoch; };
    struct Waiting { CloudMsg::ConstSharedPtr cloud; Clock::time_point received; };
    struct SensorRewindCandidate {
        int64_t first, last;
        size_t count;
        Clock::time_point started, latest_arrival;
    };
    struct Parameters {
        // Frames and initial pose.
        std::string frame;
        std::string odom_frame;
        std::string base_frame;
        Eigen::Isometry3f initial_pose;

        pipeline::PriorMapOptions map;
        algorithms::registration::RegistrationParams registration;

        // Scalar parameters used after construction.
        struct RuntimeParams {
            // Source point cloud preprocessing.
            int source_max_points = 0;
            double source_voxel_size = 0;
            int source_neighbors = 0;
            int source_min_points = 0;

            // Localization scheduling and freshness.
            double max_rate = 0;
            int max_pending = 0;
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
        runtime.max_pending = declare_parameter<int>("localization/max_pending_submaps", 2);
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

        // Validation.
        if (runtime.source_max_points < 4 || runtime.source_min_points < 4 || runtime.source_neighbors < 4 ||
            runtime.source_neighbors > 100 || runtime.source_neighbors > runtime.source_max_points ||
            runtime.max_pending < 1 || runtime.min_inliers < 1 || runtime.max_rate <= 0 ||
            runtime.pair_wait <= 0 || runtime.submap_wait <= 0 || runtime.frozen_age <= 0 ||
            runtime.odom_age <= 0 || runtime.corr_distance <= 0 || runtime.source_voxel_size <= 0 ||
            runtime.min_ratio < 0 || runtime.min_ratio > 1 || runtime.max_translation <= 0 ||
            runtime.max_rotation <= 0 || runtime.max_error <= 0 || runtime.local_overlap_radius <= 0 ||
            runtime.min_local_points < 1 || local_min_points < 1 ||
            opts.prediction_margin < runtime.max_translation ||
            runtime.min_local_overlap_ratio < 0 || runtime.min_local_overlap_ratio > 1)
            throw std::invalid_argument("Invalid localization parameters");
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

    // Caller holds mutex_. Never publish a correction for an already published timestamp.
    void publish_tf_locked() {
        if (!correction_ || !latest_odom_ || !last_success_ ||
            (now() - *last_success_).seconds() > parameters_.runtime.frozen_age ||
            std::chrono::duration<double>(Clock::now() - last_odom_arrival_).count() > parameters_.runtime.odom_age) return;
        const auto t = stamp(latest_odom_->header.stamp);
        if (t <= last_tf_stamp_) return;
        geometry_msgs::msg::PoseStamped located;
        located.header = latest_odom_->header;
        located.header.frame_id = parameters_.frame;
        located.pose = to_msg(*correction_ * pose(latest_odom_->pose.pose));
        geometry_msgs::msg::TransformStamped msg;
        msg.header = latest_odom_->header;
        msg.header.frame_id = parameters_.frame;
        msg.child_frame_id = parameters_.odom_frame;
        const auto p = to_msg(*correction_);
        msg.transform.translation.x = p.position.x;
        msg.transform.translation.y = p.position.y;
        msg.transform.translation.z = p.position.z;
        msg.transform.rotation = p.orientation;
        tf_->sendTransform(msg);
        last_tf_stamp_ = t;
        pose_pub_->publish(located);
    }

    void rewind_locked() {
        pending_.clear(); waiting_.clear(); odoms_.clear();
        ++epoch_;
        correction_.reset(); initial_correction_.reset(); last_success_.reset();
        last_accepted_ = std::numeric_limits<int64_t>::min();
        latest_odom_.reset();
        sensor_rewind_candidate_.reset();
        last_tf_stamp_ = std::numeric_limits<int64_t>::min();
        last_cloud_.reset(); reason_ = "clock rewind";
        inliers_ = 0; error_ = 0; local_overlap_ = 0;
        time_sync_warned_ = false;
        odom_frame_warned_ = false; cloud_frame_warned_ = false;
        last_odom_arrival_ = Clock::now(); last_cloud_arrival_ = Clock::now();
    }

    void check_clock_locked(const rclcpp::Time& current) {
        // Compare ROS time to its previous value, not to a possibly future sensor stamp.
        if (last_ros_time_ && (*last_ros_time_ - current).seconds() > 0.5) {
            RCLCPP_WARN(get_logger(), "ROS clock moved backwards; resetting localization");
            rewind_locked();
        }
        last_ros_time_ = current;
    }

    void pair_locked(int64_t t, const CloudMsg::ConstSharedPtr& cloud, const OdomMsg::ConstSharedPtr& odom) {
        if (t <= last_accepted_) return;
        pending_.push_back({cloud, odom, t, epoch_});
        while (pending_.size() > static_cast<size_t>(parameters_.runtime.max_pending)) pending_.pop_front();
    }

    void on_odom(OdomMsg::ConstSharedPtr msg) {
        if (msg->header.frame_id != parameters_.odom_frame || msg->child_frame_id != parameters_.base_frame) {
            std::lock_guard lock(mutex_);
            if (!odom_frame_warned_) {
                RCLCPP_WARN(get_logger(), "Skipping odometry with unexpected frames: '%s' -> '%s' (expected '%s' -> '%s')",
                            msg->header.frame_id.c_str(), msg->child_frame_id.c_str(),
                            parameters_.odom_frame.c_str(), parameters_.base_frame.c_str());
                odom_frame_warned_ = true;
            }
            return;
        }
        try {
            (void)pose(msg->pose.pose);
        } catch (const std::invalid_argument& e) {
            std::lock_guard lock(mutex_);
            reason_ = e.what();
            RCLCPP_WARN(get_logger(), "Skipping invalid odometry: %s", e.what());
            return;
        }
        const auto t = stamp(msg->header.stamp);
        std::lock_guard lock(mutex_);
        check_clock_locked(now());
        if (latest_odom_ && t < stamp(latest_odom_->header.stamp)) {
            constexpr int64_t rewind_threshold_ns = 5000000000LL;
            constexpr int64_t rewind_span_ns = 2000000000LL;
            const auto arrival = Clock::now();
            if (stamp(latest_odom_->header.stamp) - t >= rewind_threshold_ns) {
                // Sensor time alone cannot distinguish a replay from late packets. Require
                // sustained progress below the latest stamp without fresh latest odometry;
                // short replays and restarts within five seconds cannot be detected safely.
                if (!sensor_rewind_candidate_ ||
                    sensor_rewind_candidate_->latest_arrival != last_odom_arrival_ ||
                    arrival - sensor_rewind_candidate_->started > std::chrono::seconds(5) ||
                    t < sensor_rewind_candidate_->last) {
                    sensor_rewind_candidate_ = SensorRewindCandidate{t, t, 1, arrival, last_odom_arrival_};
                } else if (t > sensor_rewind_candidate_->last) {
                    sensor_rewind_candidate_->last = t;
                    ++sensor_rewind_candidate_->count;
                    if (sensor_rewind_candidate_->count >= 3 &&
                        t - sensor_rewind_candidate_->first >= rewind_span_ns &&
                        arrival - sensor_rewind_candidate_->started >= std::chrono::milliseconds(500)) {
                        RCLCPP_WARN(get_logger(), "Odometry stamps moved backwards; resetting localization");
                        rewind_locked();
                        reason_ = "sensor stamp rewind";
                    }
                }
            } else {
                sensor_rewind_candidate_.reset();
            }
        } else {
            sensor_rewind_candidate_.reset();
        }
        if (latest_odom_ && t < stamp(latest_odom_->header.stamp)) {
            // Keep a late odometry message available to pair with its scan,
            // without moving the latest TF publication timestamp backwards.
            odoms_[t] = msg;
            if (auto it = waiting_.find(t); it != waiting_.end()) {
                pair_locked(t, it->second.cloud, msg);
                waiting_.erase(it);
            }
            while (odoms_.size() > 100) odoms_.erase(odoms_.begin());
            return;
        }
        latest_odom_ = msg;
        if (!initial_correction_) initial_correction_ = parameters_.initial_pose * pose(msg->pose.pose).inverse();
        last_odom_arrival_ = Clock::now();
        if (!time_sync_warned_ && std::abs((now() - rclcpp::Time(msg->header.stamp)).seconds()) > parameters_.runtime.frozen_age) {
            RCLCPP_WARN(get_logger(), "Odometry stamp differs from ROS time; check use_sim_time and clock synchronization");
            time_sync_warned_ = true;
        }
        odoms_[t] = msg;
        if (auto it = waiting_.find(t); it != waiting_.end()) {
            pair_locked(t, it->second.cloud, msg);
            waiting_.erase(it);
        }
        while (odoms_.size() > 100) odoms_.erase(odoms_.begin());
        publish_tf_locked();
    }

    void on_cloud(CloudMsg::ConstSharedPtr msg) {
        if (msg->header.frame_id != parameters_.odom_frame) {
            std::lock_guard lock(mutex_);
            if (!cloud_frame_warned_) {
                RCLCPP_WARN(get_logger(), "Skipping submap with unexpected frame: '%s' (expected '%s')",
                            msg->header.frame_id.c_str(), parameters_.odom_frame.c_str());
                cloud_frame_warned_ = true;
            }
            return;
        }
        const auto t = stamp(msg->header.stamp);
        for (;;) {
            CloudMsg::ConstSharedPtr previous;
            size_t epoch;
            {
                std::lock_guard lock(mutex_);
                previous = last_cloud_;
                epoch = epoch_;
            }
            const bool unchanged = previous && previous->width == msg->width &&
                previous->height == msg->height && previous->fields == msg->fields &&
                previous->is_bigendian == msg->is_bigendian && previous->point_step == msg->point_step &&
                previous->row_step == msg->row_step && previous->is_dense == msg->is_dense &&
                previous->data == msg->data;
            std::lock_guard lock(mutex_);
            if (epoch != epoch_) return;
            if (previous != last_cloud_) continue;
            if (!unchanged) {
                last_cloud_ = msg;
                last_cloud_arrival_ = Clock::now();
            }
            // A previous failure can be retried, but identical successful matches must
            // not refresh the TF freshness deadline without a new map observation.
            if (unchanged && last_success_ && reason_.empty()) return;
            if (const auto it = odoms_.find(t); it != odoms_.end()) pair_locked(t, msg, it->second);
            else waiting_[t] = {msg, Clock::now()};
            while (waiting_.size() > static_cast<size_t>(parameters_.runtime.max_pending)) waiting_.erase(waiting_.begin());
            return;
        }
    }

    void publish_status() {
        std_msgs::msg::String status;
        std::lock_guard lock(mutex_);
        check_clock_locked(now());
        for (auto it = waiting_.begin(); it != waiting_.end();) {
            if (std::chrono::duration<double>(Clock::now() - it->second.received).count() > parameters_.runtime.pair_wait) {
                reason_ = "matching odometry missing";
                it = waiting_.erase(it);
            } else ++it;
        }
        const auto age = last_success_ ? (now() - *last_success_).seconds() : -1.0;
        const auto odom_elapsed = std::chrono::duration<double>(Clock::now() - last_odom_arrival_).count();
        const auto cloud_elapsed = std::chrono::duration<double>(Clock::now() - last_cloud_arrival_).count();
        const std::string reason = reason_.empty() && cloud_elapsed > parameters_.runtime.submap_wait
                                       ? "submap unavailable or unchanged" : reason_;
        const char* state = !last_success_ ? "UNINITIALIZED" :
            (age > parameters_.runtime.frozen_age || odom_elapsed > parameters_.runtime.odom_age ? "LOST" :
             (cloud_elapsed > parameters_.runtime.submap_wait || age > parameters_.runtime.submap_wait || !reason_.empty() ? "DEGRADED" : "TRACKING"));
        status.data = std::string(state) + " last_success_ns=" +
            std::to_string(last_success_ ? last_success_->nanoseconds() : 0) +
            " age_sec=" + std::to_string(age) + " inliers=" + std::to_string(inliers_) +
            " mean_error=" + std::to_string(error_) + " local_overlap=" +
            std::to_string(local_overlap_) + " reason=" + reason;
        status_pub_->publish(status);
    }

    void work() {
        std::lock_guard processing_lock(processing_mutex_);
        Pair job;
        {
            std::lock_guard lock(mutex_);
            const auto current = Clock::now();
            if (pending_.empty() || current < next_processing_) return;
            job = pending_.back(); pending_.clear();
            next_processing_ = current + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(1.0 / parameters_.runtime.max_rate));
        }
        try { process(job); }
        catch (const std::exception& e) {
            std::lock_guard lock(mutex_);
            if (job.epoch != epoch_) return;
            reason_ = e.what();
            RCLCPP_WARN(get_logger(), "Map matching failed: %s", e.what());
        }
    }

    void print_processing_time(const std::string& name, double us) const {
        constexpr size_t width = 24;
        RCLCPP_INFO(get_logger(), "%s: %*s%9.2f us", name.c_str(),
                    static_cast<int>(name.size() < width ? width - name.size() : 0), "", us);
    }

    void record_processing_time(const std::string& name, Clock::time_point begin, Clock::time_point end) {
        const double us = std::chrono::duration<double, std::micro>(end - begin).count();
        processing_times_[name].push_back(us);
        print_processing_time(name, us);
    }

    void log_processing_times() {
        int statistic = 0;
        for (const char* heading : {"MAX processing time", "MEAN processing time", "MEDIAN processing time"}) {
            RCLCPP_INFO(get_logger(), "");
            RCLCPP_INFO(get_logger(), "%s", heading);
            for (auto& [name, times] : processing_times_) {
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
        params.reg_type =
            RegType_from_string(declare_parameter<std::string>("registration/type", "gicp"));
        params.max_correspondence_distance = declare_parameter<double>(
            "registration/max_correspondence_distance", params.max_correspondence_distance);
        const int64_t max_iterations = declare_parameter<int64_t>(
            "registration/max_iterations", static_cast<int64_t>(params.max_iterations));
        params.max_iterations = static_cast<size_t>(max_iterations);
        params.criteria.translation = declare_parameter<double>(
            "registration/criteria/translation", params.criteria.translation);
        params.criteria.rotation = declare_parameter<double>(
            "registration/criteria/rotation", params.criteria.rotation);
        params.verbose = declare_parameter<bool>("registration/verbose", params.verbose);

        params.robust.type = RobustLossType_from_string(
            declare_parameter<std::string>("registration/robust/type", "NONE"));
        params.robust.default_scale = declare_parameter<double>(
            "registration/robust/default_scale", params.robust.default_scale);

        params.rotation_constraint.enable = declare_parameter<bool>(
            "registration/rotation_constraint/enable", params.rotation_constraint.enable);
        params.rotation_constraint.weight = declare_parameter<double>(
            "registration/rotation_constraint/weight", params.rotation_constraint.weight);
        params.rotation_constraint.robust.default_scale = declare_parameter<double>(
            "registration/rotation_constraint/robust/default_scale",
            params.rotation_constraint.robust.default_scale);

        params.genz.planarity_threshold = declare_parameter<double>(
            "registration/genz/planarity_threshold", params.genz.planarity_threshold);

        params.optimization_method = OptimizationMethod_from_string(
            declare_parameter<std::string>("registration/optimization_method", "GN"));
        params.gn.lambda = declare_parameter<double>("registration/gn/lambda", params.gn.lambda);
        const int64_t lm_max_inner_iterations = declare_parameter<int64_t>(
            "registration/lm/max_inner_iterations", static_cast<int64_t>(params.lm.max_inner_iterations));
        params.lm.max_inner_iterations = static_cast<size_t>(lm_max_inner_iterations);
        params.lm.lambda_factor = declare_parameter<double>(
            "registration/lm/lambda_factor", params.lm.lambda_factor);
        params.lm.init_lambda = declare_parameter<double>(
            "registration/lm/init_lambda", params.lm.init_lambda);
        params.lm.max_lambda = declare_parameter<double>(
            "registration/lm/max_lambda", params.lm.max_lambda);
        params.lm.min_lambda = declare_parameter<double>(
            "registration/lm/min_lambda", params.lm.min_lambda);
        params.dogleg.initial_trust_region_radius = declare_parameter<double>(
            "registration/dogleg/initial_trust_region_radius", params.dogleg.initial_trust_region_radius);
        params.dogleg.min_trust_region_radius = declare_parameter<double>(
            "registration/dogleg/min_trust_region_radius", params.dogleg.min_trust_region_radius);
        params.dogleg.max_trust_region_radius = declare_parameter<double>(
            "registration/dogleg/max_trust_region_radius", params.dogleg.max_trust_region_radius);
        params.dogleg.eta1 = declare_parameter<double>("registration/dogleg/eta1", params.dogleg.eta1);
        params.dogleg.eta2 = declare_parameter<double>("registration/dogleg/eta2", params.dogleg.eta2);
        params.dogleg.gamma_decrease = declare_parameter<double>(
            "registration/dogleg/gamma_decrease", params.dogleg.gamma_decrease);
        params.dogleg.gamma_increase = declare_parameter<double>(
            "registration/dogleg/gamma_increase", params.dogleg.gamma_increase);

        params.degenerate_reg.type = algorithms::registration::DegenerateRegularizationType::none;
        params.map_prior.enabled = false;

        if (params.max_correspondence_distance <= 0 || params.max_iterations < 1 ||
            params.criteria.translation <= 0 || params.criteria.rotation <= 0 ||
            params.robust.default_scale <= 0 || params.rotation_constraint.weight < 0 ||
            params.rotation_constraint.robust.default_scale <= 0 ||
            params.genz.planarity_threshold < 0 || params.genz.planarity_threshold > 1 ||
            params.gn.lambda <= 0 || params.lm.max_inner_iterations < 1 || params.lm.lambda_factor <= 1 ||
            params.lm.init_lambda <= 0 || params.lm.min_lambda <= 0 ||
            params.lm.max_lambda < params.lm.min_lambda ||
            params.dogleg.min_trust_region_radius <= 0 ||
            params.dogleg.initial_trust_region_radius < params.dogleg.min_trust_region_radius ||
            params.dogleg.max_trust_region_radius < params.dogleg.initial_trust_region_radius ||
            params.dogleg.eta1 < 0 || params.dogleg.eta1 > params.dogleg.eta2 || params.dogleg.eta2 > 1 ||
            params.dogleg.gamma_decrease <= 0 || params.dogleg.gamma_decrease >= 1 ||
            params.dogleg.gamma_increase <= 1)
            throw std::invalid_argument("Invalid registration parameters");
        return params;
    }

    void process(const Pair& job) {
        Eigen::Isometry3f guess;
        {
            std::lock_guard lock(mutex_);
            if (job.epoch != epoch_ || job.time <= last_accepted_ ||
                (latest_odom_ && job.time > stamp(latest_odom_->header.stamp))) return;
            guess = correction_.value_or(*initial_correction_);
        }
        const auto started = Clock::now();
        PointCloudCPU cpu;
        sensor_msgs::PointCloud2ConstIterator<float> x(*job.cloud, "x"), y(*job.cloud, "y"), z(*job.cloud, "z");
        const size_t total = static_cast<size_t>(job.cloud->width) * job.cloud->height;
        const size_t stride = std::max<size_t>(1, (total + parameters_.runtime.source_max_points - 1) / parameters_.runtime.source_max_points);
        for (size_t i = 0; i < total; ++i, ++x, ++y, ++z) {
            if (i % stride == 0 && std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z))
                cpu.points->emplace_back(*x, *y, *z, 1.0f);
        }
        if (cpu.size() < static_cast<size_t>(parameters_.runtime.source_min_points)) throw std::runtime_error("source too small");
        const auto converted = Clock::now();
        auto source = PointCloudShared(*queue_, cpu);
        algorithms::filter::VoxelGrid voxel(*queue_, parameters_.runtime.source_voxel_size);
        voxel.downsampling(source, source);
        if (source.size() < static_cast<size_t>(std::max(parameters_.runtime.source_min_points, parameters_.runtime.source_neighbors)))
            throw std::runtime_error("downsampled source too small");
        const auto reduced = Clock::now();
        const auto source_tree = algorithms::knn::KDTree::build(*queue_, source);
        const auto indexed = Clock::now();
        algorithms::covariance::estimate_async(*source_tree, source, parameters_.runtime.source_neighbors).wait_and_throw();
        const auto covariances = Clock::now();
        const auto center = (guess * pose(job.odom->pose.pose)).translation();
        float reach = 0;
        for (const auto& p : *source.points)
            reach = std::max(reach, (guess * p.head<3>() - center).norm());
        const auto target = map_->target(center, reach + parameters_.runtime.corr_distance);
        if (!target) throw std::runtime_error("local map insufficient or required extent exceeds radius");
        const auto localized = Clock::now();
        const auto result = registration_->align(source, *target->cloud, *target->tree, guess.matrix());
        const auto matched = Clock::now();
        const auto delta = guess.inverse() * result.T;
        const float mean_error = result.inlier ? result.error / result.inlier : INFINITY;
        if (!result.T.matrix().allFinite() || !std::isfinite(mean_error) ||
            result.inlier < static_cast<size_t>(parameters_.runtime.min_inliers) ||
            static_cast<double>(result.inlier) / source.size() < parameters_.runtime.min_ratio ||
            delta.translation().norm() > parameters_.runtime.max_translation ||
            Eigen::AngleAxisf(delta.rotation()).angle() > parameters_.runtime.max_rotation || mean_error > parameters_.runtime.max_error)
            throw std::runtime_error("registration rejected by quality gates");
        // Older submap points can match a map even after the robot leaves it.
        algorithms::knn::KNNResult neighbors;
        target->tree->nearest_neighbor_search(source, neighbors, {}, result.T.matrix());
        const auto current_odom_position = pose(job.odom->pose.pose).translation();
        size_t nearby = 0, nearby_matched = 0;
        for (size_t i = 0; i < source.size(); ++i) {
            if (((*source.points)[i].head<3>() - current_odom_position).squaredNorm() >
                parameters_.runtime.local_overlap_radius * parameters_.runtime.local_overlap_radius) continue;
            ++nearby;
            if ((*neighbors.distances)[i] <= parameters_.runtime.corr_distance * parameters_.runtime.corr_distance) ++nearby_matched;
        }
        const float local_overlap = nearby ? static_cast<float>(nearby_matched) / nearby : 0.0f;
        if (nearby < static_cast<size_t>(parameters_.runtime.min_local_points) || local_overlap < parameters_.runtime.min_local_overlap_ratio)
            throw std::runtime_error("insufficient map overlap near current robot position");
        const auto checked = Clock::now();
        {
            std::lock_guard lock(mutex_);
            if (job.epoch != epoch_ || job.time <= last_accepted_ ||
                (latest_odom_ && job.time > stamp(latest_odom_->header.stamp))) return;
            correction_ = result.T;
            last_accepted_ = job.time;
            last_success_ = rclcpp::Time(job.cloud->header.stamp);
            inliers_ = result.inlier;
            error_ = mean_error;
            local_overlap_ = local_overlap;
            reason_.clear();
            if ((now() - *last_success_).seconds() > parameters_.runtime.frozen_age) {
                reason_ = "registration completed after TF freshness deadline";
                RCLCPP_WARN(get_logger(), "%s", reason_.c_str());
            }
            publish_tf_locked();
        }
        const auto published = Clock::now();
        RCLCPP_INFO(get_logger(), "Matched %zu correspondences", static_cast<size_t>(result.inlier));
        record_processing_time("0. from ROS 2 msg", started, converted);
        record_processing_time("1. downsampling", converted, reduced);
        record_processing_time("2. source tree", reduced, indexed);
        record_processing_time("3. covariance", indexed, covariances);
        record_processing_time("4. local target", covariances, localized);
        record_processing_time("5. registration", localized, matched);
        record_processing_time("6. overlap check", matched, checked);
        record_processing_time("7. publish ROS 2 msg", checked, published);
        record_processing_time("8. total", started, published);
        RCLCPP_INFO(get_logger(), "");
    }

    // Core components.
    std::unique_ptr<sycl_utils::DeviceQueue> queue_;
    std::unique_ptr<pipeline::PriorMap> map_;
    std::unique_ptr<algorithms::registration::Registration> registration_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;

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

    // Pending messages waiting for pairing.
    std::deque<Pair> pending_;
    std::map<int64_t, OdomMsg::ConstSharedPtr> odoms_;
    std::map<int64_t, Waiting> waiting_;

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
