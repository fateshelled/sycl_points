#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vector>

#include "sycl_points/points/point_cloud.hpp"

namespace sycl_points {
namespace ros2 {

// Apply generate_enhanced_reflectivity2-equivalent correction to cloud.intensities.
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
// Requires intensity, ambient (UINT16), and ring fields in msg. Returns early if any is absent.
// When enabled, scan/intensity_correction is automatically skipped (pointcloud_processing.hpp).
inline void apply_enhanced_reflectivity(PointCloudShared& cloud, const sensor_msgs::msg::PointCloud2& msg,
                                        float clip_max = 5.0f) {
    if (!cloud.has_intensity()) return;

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

    if (ambient_offset < 0 || ring_offset < 0) return;
    if (ambient_type != sensor_msgs::msg::PointField::UINT16) return;

    const bool ring_is_uint8 = (ring_type == sensor_msgs::msg::PointField::UINT8);
    const bool ring_is_uint16 = (ring_type == sensor_msgs::msg::PointField::UINT16);
    if (!ring_is_uint8 && !ring_is_uint16) return;

    const size_t N = cloud.size();
    const size_t point_step = msg.point_step;
    const auto* msg_bytes = msg.data.data();

    const auto& points = *cloud.points;
    auto& intensities = *cloud.intensities;

    constexpr size_t MAX_RINGS = 256;
    std::vector<float> en_ref(N, 0.0f);
    std::vector<float> en_amb(N, 0.0f);
    std::vector<uint16_t> ring_idx(N, 0);

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
        if (ring_is_uint8) {
            r = static_cast<uint16_t>(msg_bytes[base + ring_offset]);
        } else {
            r = reinterpret_cast<const uint16_t*>(&msg_bytes[base + ring_offset])[0];
        }
        ring_idx[i] = r;

        const float x = points[i].x();
        const float y = points[i].y();
        const float z = points[i].z();
        const float range_sq = x * x + y * y + z * z;
        if (range_sq < 1e-6f) continue;

        const float ref = en_ref[i] = intensities[i] * range_sq;
        const float raw_amb =
            static_cast<float>(reinterpret_cast<const uint16_t*>(&msg_bytes[base + ambient_offset])[0]);
        const float amb = en_amb[i] = raw_amb / range_sq;

        if (r < MAX_RINGS) {
            ring_sum_ref[r] += ref;
            ring_sum_amb[r] += amb;
            ring_count[r]++;
        }
    }

    // Compute per-ring means
    std::array<float, MAX_RINGS> ring_mean_ref{};
    std::array<float, MAX_RINGS> ring_mean_amb{};
    for (size_t r = 0; r < MAX_RINGS; ++r) {
        const float cnt = static_cast<float>(ring_count[r]);
        ring_mean_ref[r] = (cnt > 0.0f) ? ring_sum_ref[r] / cnt : 0.0f;
        ring_mean_amb[r] = (cnt > 0.0f) ? ring_sum_amb[r] / cnt : 0.0f;
    }

    // Pass 2: normalize and write back
    for (size_t i = 0; i < N; ++i) {
        const uint16_t r = ring_idx[i];
        float ref = en_ref[i];
        float amb = en_amb[i];
        if (ring_mean_ref[r] > 0.0f) ref /= ring_mean_ref[r];
        if (ring_mean_amb[r] > 0.0f) amb /= ring_mean_amb[r];
        intensities[i] = std::clamp(ref + amb, 0.0f, clip_max);
    }
}

}  // namespace ros2
}  // namespace sycl_points
