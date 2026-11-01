#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "sycl_points/algorithms/registration/registration.hpp"
#include "sycl_points/pipeline/prior_map.hpp"

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
    PriorMapLocalizationNode() : Node("prior_map_localization") {
        const auto number = [this](const std::string& key, double value) { return declare_parameter(key, value); };
        const auto integer = [this](const std::string& key, int value) { return declare_parameter(key, value); };
        frame_ = declare_parameter<std::string>("map/frame_id", "map");
        odom_frame_ = declare_parameter<std::string>("odom/frame_id", "odom");
        base_frame_ = declare_parameter<std::string>("base/frame_id", "base_link");
        auto configured_pose = [&](const std::string& prefix) {
            geometry_msgs::msg::Pose p;
            p.position.x = number(prefix + "/x", 0);
            p.position.y = number(prefix + "/y", 0);
            p.position.z = number(prefix + "/z", 0);
            p.orientation.x = number(prefix + "/qx", 0);
            p.orientation.y = number(prefix + "/qy", 0);
            p.orientation.z = number(prefix + "/qz", 0);
            p.orientation.w = number(prefix + "/qw", 1);
            return pose(p);
        };
        initial_pose_ = configured_pose("initial_pose");
        pipeline::PriorMapOptions opts;
        opts.file_pose = configured_pose("map_file_pose");
        opts.downsample = declare_parameter("map/downsampling/enable", true);
        opts.voxel_size = number("map/downsampling/voxel_size", 0.4);
        opts.covariance_mode = declare_parameter<std::string>("map/covariance/mode", "AUTO");
        opts.eigenvalue_floor = number("map/covariance/eigenvalue_floor", 1e-4);
        opts.covariance_neighbors = integer("map/covariance/neighbor_num", 20);
        opts.local_radius = number("map/local_target/radius", 100);
        opts.prediction_margin = number("map/local_target/prediction_margin", 2);
        opts.local_min_points = integer("map/local_target/min_points", 50);
        source_max_points_ = integer("source/max_points", 6000);
        source_voxel_size_ = number("source/voxel_size", 0.4);
        source_neighbors_ = integer("source/covariance/neighbor_num", 20);
        source_min_points_ = integer("source/min_points", 50);
        max_rate_ = number("localization/max_rate_hz", 1);
        max_pending_ = integer("localization/max_pending_submaps", 2);
        pair_wait_ = number("localization/max_pair_wait_sec", 0.5);
        submap_wait_ = number("localization/max_submap_wait_sec", 3);
        frozen_age_ = number("localization/max_frozen_tf_age_sec", 5);
        odom_age_ = number("localization/max_odom_age_sec", 1);
        corr_distance_ = number("registration/max_correspondence_distance", 2);
        min_inliers_ = integer("registration/min_correspondences", 30);
        min_ratio_ = number("registration/min_correspondence_ratio", 0.2);
        max_translation_ = number("registration/max_translation_correction", 3);
        max_rotation_ = number("registration/max_rotation_correction_rad", 0.5);
        max_error_ = number("registration/max_mean_error", 10);
        if (source_max_points_ < 4 || source_min_points_ < 4 || source_neighbors_ < 4 || source_neighbors_ > 100 ||
            source_neighbors_ > source_max_points_ ||
            max_pending_ < 1 || min_inliers_ < 1 || max_rate_ <= 0 || pair_wait_ <= 0 || submap_wait_ <= 0 ||
            frozen_age_ <= 0 || odom_age_ <= 0 || corr_distance_ <= 0 || source_voxel_size_ <= 0 ||
            min_ratio_ < 0 || min_ratio_ > 1 || max_translation_ <= 0 || max_rotation_ <= 0 || max_error_ <= 0)
            throw std::invalid_argument("Invalid localization parameters");

        queue_ = std::make_unique<sycl_utils::DeviceQueue>(
            sycl::device(sycl_utils::device_selector::default_selector_v));
        map_ = std::make_unique<pipeline::PriorMap>(*queue_, declare_parameter<std::string>("map/path", ""), opts);
        algorithms::registration::RegistrationParams params;
        params.reg_type = algorithms::registration::RegType::GICP;
        params.max_correspondence_distance = corr_distance_;
        params.max_iterations = integer("registration/max_iterations", 20);
        registration_ = std::make_unique<algorithms::registration::Registration>(*queue_, params);
        tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("localization/pose", 5);
        status_pub_ = create_publisher<std_msgs::msg::String>("localization/status", 5);
        odom_sub_ = create_subscription<OdomMsg>("sycl_lio/odom", rclcpp::QoS(5),
            [this](OdomMsg::ConstSharedPtr msg) { on_odom(std::move(msg)); });
        cloud_sub_ = create_subscription<CloudMsg>("sycl_lio/submap", rclcpp::QoS(5),
            [this](CloudMsg::ConstSharedPtr msg) { on_cloud(std::move(msg)); });
        timer_ = create_wall_timer(std::chrono::milliseconds(200), [this] { publish_status(); });
        worker_ = std::thread([this] { work(); });
    }

    ~PriorMapLocalizationNode() override {
        { std::lock_guard lock(mutex_); stopping_ = true; }
        cv_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

private:
    struct Pair { CloudMsg::ConstSharedPtr cloud; OdomMsg::ConstSharedPtr odom; int64_t time; size_t epoch; };
    struct Waiting { CloudMsg::ConstSharedPtr cloud; Clock::time_point received; };

    // Caller holds mutex_. Never publish a correction for an already published timestamp.
    void publish_tf_locked() {
        if (!correction_ || !latest_odom_ || !last_success_ ||
            (now() - *last_success_).seconds() > frozen_age_ ||
            std::chrono::duration<double>(Clock::now() - last_odom_arrival_).count() > odom_age_) return;
        const auto t = stamp(latest_odom_->header.stamp);
        if (t <= last_tf_stamp_) return;
        geometry_msgs::msg::TransformStamped msg;
        msg.header = latest_odom_->header;
        msg.header.frame_id = frame_;
        msg.child_frame_id = odom_frame_;
        const auto p = to_msg(*correction_);
        msg.transform.translation.x = p.position.x;
        msg.transform.translation.y = p.position.y;
        msg.transform.translation.z = p.position.z;
        msg.transform.rotation = p.orientation;
        tf_->sendTransform(msg);
        last_tf_stamp_ = t;
        geometry_msgs::msg::PoseStamped located;
        located.header = msg.header;
        located.pose = to_msg(*correction_ * pose(latest_odom_->pose.pose));
        pose_pub_->publish(located);
    }

    void rewind_locked() {
        pending_.clear(); waiting_.clear(); odoms_.clear();
        ++epoch_;
        correction_.reset(); last_success_.reset(); last_accepted_ = 0;
        last_tf_stamp_ = 0; last_cloud_hash_.reset(); reason_ = "clock rewind";
    }

    void pair_locked(int64_t t, const CloudMsg::ConstSharedPtr& cloud, const OdomMsg::ConstSharedPtr& odom) {
        if (t <= last_accepted_) return;
        pending_.push_back({cloud, odom, t, epoch_});
        while (pending_.size() > static_cast<size_t>(max_pending_)) pending_.pop_front();
        cv_.notify_one();
    }

    void on_odom(OdomMsg::ConstSharedPtr msg) {
        if (msg->header.frame_id != odom_frame_ || msg->child_frame_id != base_frame_) return;
        const auto t = stamp(msg->header.stamp);
        std::lock_guard lock(mutex_);
        if (latest_odom_ && t < stamp(latest_odom_->header.stamp)) rewind_locked();
        latest_odom_ = msg;
        last_odom_arrival_ = Clock::now();
        odoms_[t] = msg;
        if (auto it = waiting_.find(t); it != waiting_.end()) {
            pair_locked(t, it->second.cloud, msg);
            waiting_.erase(it);
        }
        while (odoms_.size() > 100) odoms_.erase(odoms_.begin());
        publish_tf_locked();
    }

    void on_cloud(CloudMsg::ConstSharedPtr msg) {
        if (msg->header.frame_id != odom_frame_) return;
        const auto t = stamp(msg->header.stamp);
        std::lock_guard lock(mutex_);
        size_t hash = msg->width * static_cast<size_t>(msg->height);
        if (msg->point_step && !msg->data.empty()) {
            const size_t count = msg->data.size() / msg->point_step;
            for (size_t sample = 0; sample < std::min<size_t>(32, count); ++sample) {
                const size_t offset = (sample * count / std::min<size_t>(32, count)) * msg->point_step;
                uint32_t bits = 0;
                std::memcpy(&bits, msg->data.data() + offset, std::min<size_t>(4, msg->point_step));
                hash ^= std::hash<uint32_t>{}(bits) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
        }
        if (last_cloud_hash_ && *last_cloud_hash_ == hash) return;
        last_cloud_hash_ = hash;
        last_cloud_arrival_ = Clock::now();
        if (const auto it = odoms_.find(t); it != odoms_.end()) pair_locked(t, msg, it->second);
        else waiting_[t] = {msg, Clock::now()};
        while (waiting_.size() > static_cast<size_t>(max_pending_)) waiting_.erase(waiting_.begin());
    }

    void publish_status() {
        std_msgs::msg::String status;
        std::lock_guard lock(mutex_);
        if (last_success_ && now() < *last_success_) rewind_locked();
        for (auto it = waiting_.begin(); it != waiting_.end();) {
            if (std::chrono::duration<double>(Clock::now() - it->second.received).count() > pair_wait_) {
                reason_ = "matching odometry missing";
                it = waiting_.erase(it);
            } else ++it;
        }
        const auto age = last_success_ ? (now() - *last_success_).seconds() : -1.0;
        const auto odom_elapsed = std::chrono::duration<double>(Clock::now() - last_odom_arrival_).count();
        const auto cloud_elapsed = std::chrono::duration<double>(Clock::now() - last_cloud_arrival_).count();
        if (cloud_elapsed > submap_wait_ && reason_.empty()) reason_ = "submap unavailable or unchanged";
        const char* state = !last_success_ ? "UNINITIALIZED" :
            (age > frozen_age_ || odom_elapsed > odom_age_ ? "LOST" :
             (cloud_elapsed > submap_wait_ || age > submap_wait_ || !reason_.empty() ? "DEGRADED" : "TRACKING"));
        status.data = std::string(state) + " last_success_ns=" +
            std::to_string(last_success_ ? last_success_->nanoseconds() : 0) +
            " age_sec=" + std::to_string(age) + " inliers=" + std::to_string(inliers_) +
            " mean_error=" + std::to_string(error_) + " reason=" + reason_;
        status_pub_->publish(status);
    }

    void work() {
        auto next = Clock::now();
        while (true) {
            Pair job;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
                if (stopping_) return;
                lock.unlock();
                std::this_thread::sleep_until(next);
                lock.lock();
                if (stopping_) return;
                if (pending_.empty()) continue;
                job = pending_.back(); pending_.clear();
            }
            next = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1 / max_rate_));
            try { process(job); }
            catch (const std::exception& e) {
                std::lock_guard lock(mutex_); reason_ = e.what();
                RCLCPP_WARN(get_logger(), "Map matching failed: %s", e.what());
            }
        }
    }

    void process(const Pair& job) {
        Eigen::Isometry3f guess;
        {
            std::lock_guard lock(mutex_);
            if (job.epoch != epoch_ || job.time <= last_accepted_ ||
                (latest_odom_ && job.time > stamp(latest_odom_->header.stamp))) return;
            guess = correction_.value_or(initial_pose_ * pose(job.odom->pose.pose).inverse());
        }
        const auto started = Clock::now();
        PointCloudCPU cpu;
        sensor_msgs::PointCloud2ConstIterator<float> x(*job.cloud, "x"), y(*job.cloud, "y"), z(*job.cloud, "z");
        const size_t total = static_cast<size_t>(job.cloud->width) * job.cloud->height;
        const size_t stride = std::max<size_t>(1, (total + source_max_points_ - 1) / source_max_points_);
        for (size_t i = 0; i < total; ++i, ++x, ++y, ++z) {
            if (i % stride == 0 && std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z))
                cpu.points->emplace_back(*x, *y, *z, 1.0f);
        }
        if (cpu.size() < static_cast<size_t>(source_min_points_)) throw std::runtime_error("source too small");
        const auto converted = Clock::now();
        auto source = PointCloudShared(*queue_, cpu);
        algorithms::filter::VoxelGrid voxel(*queue_, source_voxel_size_);
        voxel.downsampling(source, source);
        if (source.size() < static_cast<size_t>(std::max(source_min_points_, source_neighbors_)))
            throw std::runtime_error("downsampled source too small");
        const auto reduced = Clock::now();
        const auto source_tree = algorithms::knn::KDTree::build(*queue_, source);
        const auto indexed = Clock::now();
        algorithms::covariance::estimate_async(*source_tree, source, source_neighbors_).wait_and_throw();
        const auto covariances = Clock::now();
        const auto center = (guess * pose(job.odom->pose.pose)).translation();
        float reach = 0;
        for (const auto& p : *source.points)
            reach = std::max(reach, (guess * p.head<3>() - center).norm());
        const auto target = map_->target(center, reach + corr_distance_);
        if (!target) throw std::runtime_error("local map insufficient or required extent exceeds radius");
        const auto localized = Clock::now();
        const auto result = registration_->align(source, *target->cloud, *target->tree, guess.matrix());
        const auto matched = Clock::now();
        const auto delta = guess.inverse() * result.T;
        const float mean_error = result.inlier ? result.error / result.inlier : INFINITY;
        if (!result.T.matrix().allFinite() || !std::isfinite(mean_error) ||
            result.inlier < static_cast<size_t>(min_inliers_) ||
            static_cast<double>(result.inlier) / source.size() < min_ratio_ ||
            delta.translation().norm() > max_translation_ ||
            Eigen::AngleAxisf(delta.rotation()).angle() > max_rotation_ || mean_error > max_error_)
            throw std::runtime_error("registration rejected by quality gates");
        {
            std::lock_guard lock(mutex_);
            if (job.epoch != epoch_ || job.time <= last_accepted_ ||
                (latest_odom_ && job.time > stamp(latest_odom_->header.stamp))) return;
            correction_ = result.T;
            last_accepted_ = job.time;
            last_success_ = rclcpp::Time(job.cloud->header.stamp);
            inliers_ = result.inlier;
            error_ = mean_error;
            reason_.clear();
            publish_tf_locked();
        }
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        RCLCPP_DEBUG(get_logger(),
            "Matched %zu correspondences; transfer %.1f, voxel %.1f, source tree %.1f, covariance %.1f, target %.1f, registration %.1f ms",
            inliers_, ms(started, converted), ms(converted, reduced), ms(reduced, indexed),
            ms(indexed, covariances), ms(covariances, localized), ms(localized, matched));
    }

    std::unique_ptr<sycl_utils::DeviceQueue> queue_;
    std::unique_ptr<pipeline::PriorMap> map_;
    std::unique_ptr<algorithms::registration::Registration> registration_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Subscription<OdomMsg>::SharedPtr odom_sub_;
    rclcpp::Subscription<CloudMsg>::SharedPtr cloud_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::string frame_, odom_frame_, base_frame_, reason_;
    Eigen::Isometry3f initial_pose_;
    std::optional<Eigen::Isometry3f> correction_;
    std::optional<rclcpp::Time> last_success_;
    OdomMsg::ConstSharedPtr latest_odom_;
    Clock::time_point last_odom_arrival_ = Clock::now(), last_cloud_arrival_ = Clock::now();
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    bool stopping_ = false;
    std::deque<Pair> pending_;
    std::map<int64_t, OdomMsg::ConstSharedPtr> odoms_;
    std::map<int64_t, Waiting> waiting_;
    int64_t last_tf_stamp_ = 0, last_accepted_ = 0;
    size_t epoch_ = 0;
    std::optional<size_t> last_cloud_hash_;
    size_t inliers_ = 0;
    float error_ = 0;
    int source_max_points_, source_neighbors_, source_min_points_, max_pending_, min_inliers_;
    double source_voxel_size_, max_rate_, pair_wait_, submap_wait_, frozen_age_, odom_age_, corr_distance_;
    double min_ratio_, max_translation_, max_rotation_, max_error_;
};

}  // namespace sycl_points::ros2

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<sycl_points::ros2::PriorMapLocalizationNode>());
    } catch (const std::exception& e) {
        fprintf(stderr, "Prior map localization: %s\n", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
