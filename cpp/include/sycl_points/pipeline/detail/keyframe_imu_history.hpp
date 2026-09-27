#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <vector>

#include "sycl_points/algorithms/imu/imu_preintegration.hpp"

namespace sycl_points::pipeline::graph_odometry::detail {

class KeyframeImuHistory {
public:
    enum class Coverage { ready, waiting_for_future, recovery_required };

    struct Limits {
        double max_duration_sec = 5.0;
        size_t max_samples = 4096;
    };

    explicit KeyframeImuHistory(const Limits& limits) : limits_(limits) {
        if (!std::isfinite(limits_.max_duration_sec) || limits_.max_duration_sec <= 0.0 ||
            limits_.max_samples < 2) {
            throw std::invalid_argument("KeyframeImuHistory limits must be finite and positive");
        }
    }

    void append(const imu::IMUMeasurement& measurement) {
        if (source_timestamp_ < 0.0) return;
        measurements_.push_back(measurement);
        enforce_limits();
    }

    bool reset(double source_timestamp, const std::deque<imu::IMUMeasurement>& measurements) {
        if (measurements.empty()) return false;
        auto after = std::lower_bound(
            measurements.begin(), measurements.end(), source_timestamp,
            [](const imu::IMUMeasurement& measurement, double timestamp) {
                return measurement.timestamp < timestamp;
            });
        if (after == measurements.end() ||
            (after == measurements.begin() && after->timestamp != source_timestamp)) {
            return false;
        }

        imu::IMUMeasurement boundary;
        if (after->timestamp == source_timestamp) {
            boundary = *after;
        } else {
            boundary = imu::interpolate_measurement(*std::prev(after), *after, source_timestamp);
        }
        source_timestamp_ = source_timestamp;
        overflowed_ = false;
        measurements_.clear();
        measurements_.push_back(boundary);
        for (; after != measurements.end(); ++after) {
            if (after->timestamp > source_timestamp) measurements_.push_back(*after);
        }
        enforce_limits();
        return !overflowed_;
    }

    Coverage coverage(double end_timestamp) const {
        if (overflowed_ || source_timestamp_ < 0.0 || measurements_.empty() ||
            measurements_.front().timestamp > source_timestamp_ + kTolerance) {
            return Coverage::recovery_required;
        }
        if (measurements_.back().timestamp + kTolerance < end_timestamp) {
            return Coverage::waiting_for_future;
        }
        return Coverage::ready;
    }

    bool build_window(double end_timestamp, std::vector<imu::IMUMeasurement>& output) const {
        if (coverage(end_timestamp) != Coverage::ready) {
            output.clear();
            return false;
        }
        return imu::build_measurement_window(measurements_, source_timestamp_, end_timestamp, output);
    }

    bool should_force_keyframe(double end_timestamp) const {
        if (coverage(end_timestamp) != Coverage::ready) return false;
        const auto end = std::upper_bound(
            measurements_.begin(), measurements_.end(), end_timestamp + kTolerance,
            [](double timestamp, const imu::IMUMeasurement& measurement) {
                return timestamp < measurement.timestamp;
            });
        const size_t samples = static_cast<size_t>(std::distance(measurements_.begin(), end));
        return end_timestamp - source_timestamp_ >= kForceRatio * limits_.max_duration_sec ||
               samples >= static_cast<size_t>(std::ceil(kForceRatio * limits_.max_samples));
    }

    bool overflowed() const { return overflowed_; }
    double source_timestamp() const { return source_timestamp_; }
    double latest_timestamp() const {
        return measurements_.empty() ? -1.0 : measurements_.back().timestamp;
    }
    size_t size() const { return measurements_.size(); }

private:
    void enforce_limits() {
        while (measurements_.size() > limits_.max_samples ||
               (!measurements_.empty() &&
                measurements_.back().timestamp - measurements_.front().timestamp >
                    limits_.max_duration_sec)) {
            if (measurements_.front().timestamp <= source_timestamp_ + kTolerance) overflowed_ = true;
            measurements_.pop_front();
        }
    }

    static constexpr double kForceRatio = 0.8;
    static constexpr double kTolerance = 1e-9;
    Limits limits_;
    std::deque<imu::IMUMeasurement> measurements_;
    double source_timestamp_ = -1.0;
    bool overflowed_ = false;
};

}  // namespace sycl_points::pipeline::graph_odometry::detail
