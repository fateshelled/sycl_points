#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <cstring>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vector>

#include "sycl_points/points/point_cloud.hpp"

namespace sycl_points {
namespace ros2 {

// Designed for Ouster LiDAR, which provides 'intensity' (raw signal), 'ambient' (background
// light), and 'ring' (beam index) fields in the PointCloud2 message.
//
// Algorithm:
//   range_sq = x^2 + y^2 + z^2
//   en_ref   = intensity * range_sq          (range compensation)
//   en_amb   = ambient   / range_sq          (ambient compensation)
//   per-ring normalization: divide by ring mean (removes horizontal banding)
//   intensity = clip(en_ref + en_amb, 0.0, clip_max)
//
// Per-ring means are retained across calls and updated with EMA each scan.
// Rings with no points in the current scan keep their previous mean unchanged.
// Field offsets are parsed once on the first valid scan and cached.
//
// Requires intensity, ambient (UINT16), and ring fields in msg. Returns early if any is absent.
// When enabled, scan/intensity_correction is automatically skipped (pointcloud_processing.hpp).
class EnhancedReflectivityCorrector {
public:
    static constexpr size_t MAX_RINGS = 256;

    explicit EnhancedReflectivityCorrector(float ema_alpha = 0.5f) : ema_alpha_(ema_alpha) {
        this->ring_mean_ref_.fill(0.0f);
        this->ring_mean_amb_.fill(0.0f);
    }

    void set_ema_alpha(float alpha) { this->ema_alpha_ = alpha; }

    void apply(PointCloudShared& cloud, const sensor_msgs::msg::PointCloud2& msg, float clip_max = 5.0f) {
        if (!cloud.has_intensity()) return;
        if (!this->parse_fields(msg)) return;

        const size_t N = cloud.size();
        const size_t point_step = msg.point_step;
        const auto* msg_bytes = msg.data.data();

        const auto& points = *cloud.points;
        auto& intensities = *cloud.intensities;

        this->en_ref_.assign(N, 0.0f);
        this->en_amb_.assign(N, 0.0f);
        this->ring_idx_.resize(N);

        std::array<float, MAX_RINGS> ring_sum_ref{};
        std::array<float, MAX_RINGS> ring_sum_amb{};
        std::array<int32_t, MAX_RINGS> ring_count{};
        ring_sum_ref.fill(0.0f);
        ring_sum_amb.fill(0.0f);
        ring_count.fill(0);

        // Pass 1: range compensation and per-ring accumulation
        for (size_t i = 0; i < N; ++i) {
            const size_t base = point_step * i;

            uint16_t r = 0;
            if (this->ring_is_uint8_) {
                r = static_cast<uint16_t>(msg_bytes[base + this->ring_offset_]);
            } else {
                std::memcpy(&r, &msg_bytes[base + this->ring_offset_], sizeof(uint16_t));
            }
            this->ring_idx_[i] = r;

            const float x = points[i].x();
            const float y = points[i].y();
            const float z = points[i].z();
            const float range_sq = x * x + y * y + z * z;
            if (range_sq < 1e-6f) continue;

            const float ref = this->en_ref_[i] = intensities[i] * range_sq;
            uint16_t raw_amb_val = 0;
            std::memcpy(&raw_amb_val, &msg_bytes[base + this->ambient_offset_], sizeof(uint16_t));
            const float raw_amb = static_cast<float>(raw_amb_val);
            const float amb = this->en_amb_[i] = raw_amb / range_sq;

            if (r < MAX_RINGS) {
                ring_sum_ref[r] += ref;
                ring_sum_amb[r] += amb;
                ring_count[r]++;
            }
        }

        // Update per-ring means with EMA for rings that have points in this scan.
        // First observation is initialized directly (no blending) tracked via bitset.
        // Rings with no points keep their previous mean unchanged.
        for (size_t r = 0; r < MAX_RINGS; ++r) {
            if (ring_count[r] > 0) {
                const float cnt = static_cast<float>(ring_count[r]);
                const float new_ref = ring_sum_ref[r] / cnt;
                const float new_amb = ring_sum_amb[r] / cnt;
                if (!this->ring_initialized_[r]) {
                    this->ring_mean_ref_[r] = new_ref;
                    this->ring_mean_amb_[r] = new_amb;
                    this->ring_initialized_[r] = true;
                } else {
                    this->ring_mean_ref_[r] =
                        this->ema_alpha_ * new_ref + (1.0f - this->ema_alpha_) * this->ring_mean_ref_[r];
                    this->ring_mean_amb_[r] =
                        this->ema_alpha_ * new_amb + (1.0f - this->ema_alpha_) * this->ring_mean_amb_[r];
                }
            }
        }

        // Pass 2: normalize and write back
        for (size_t i = 0; i < N; ++i) {
            const uint16_t r = this->ring_idx_[i];
            if (r >= MAX_RINGS) continue;
            float ref = this->en_ref_[i];
            float amb = this->en_amb_[i];
            if (this->ring_mean_ref_[r] > 0.0f) ref /= this->ring_mean_ref_[r];
            if (this->ring_mean_amb_[r] > 0.0f) amb /= this->ring_mean_amb_[r];
            intensities[i] = std::clamp(ref + amb, 0.0f, clip_max);
        }
    }

private:
    // Returns false if the message doesn't have the required fields.
    // Caches offsets/types after the first successful parse.
    bool parse_fields(const sensor_msgs::msg::PointCloud2& msg) {
        if (this->fields_parsed_) return true;

        int32_t ambient_offset = -1;
        int32_t ring_offset = -1;
        uint8_t ambient_type = 0;
        uint8_t ring_type = 0;

        for (const auto& field : msg.fields) {
            if (field.name == "ambient") {
                ambient_offset = static_cast<int32_t>(field.offset);
                ambient_type = field.datatype;
            } else if (field.name == "ring") {
                ring_offset = static_cast<int32_t>(field.offset);
                ring_type = field.datatype;
            }
        }

        if (ambient_offset < 0 || ring_offset < 0) return false;
        if (ambient_type != sensor_msgs::msg::PointField::UINT16) return false;

        const bool ring_is_uint8 = (ring_type == sensor_msgs::msg::PointField::UINT8);
        const bool ring_is_uint16 = (ring_type == sensor_msgs::msg::PointField::UINT16);
        if (!ring_is_uint8 && !ring_is_uint16) return false;

        this->ambient_offset_ = ambient_offset;
        this->ring_offset_ = ring_offset;
        this->ring_is_uint8_ = ring_is_uint8;
        this->fields_parsed_ = true;
        return true;
    }

    float ema_alpha_;

    // Cached field layout (parsed once on first valid scan)
    bool fields_parsed_ = false;
    int32_t ambient_offset_ = -1;
    int32_t ring_offset_ = -1;
    bool ring_is_uint8_ = false;

    // Per-ring state
    std::bitset<MAX_RINGS> ring_initialized_;
    std::array<float, MAX_RINGS> ring_mean_ref_;
    std::array<float, MAX_RINGS> ring_mean_amb_;

    // Per-scan working buffers (reused across calls to avoid repeated allocation)
    std::vector<float> en_ref_;
    std::vector<float> en_amb_;
    std::vector<uint16_t> ring_idx_;
};

}  // namespace ros2
}  // namespace sycl_points
