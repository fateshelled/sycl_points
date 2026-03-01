#include <gtest/gtest.h>

#include "sycl_points/algorithms/filter/intensity_correction.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

TEST(IntensityCorrectionTest, AppliesDistanceCompensation) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3);
    cpu_cloud.intensities->resize(3);

    (*cpu_cloud.points)[0] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);  // distance 1
    (*cpu_cloud.points)[1] = sycl_points::PointType(0.0f, 3.0f, 4.0f, 1.0f);  // distance 5
    (*cpu_cloud.points)[2] = sycl_points::PointType(1.0f, 2.0f, 2.0f, 1.0f);  // distance 3

    (*cpu_cloud.intensities)[0] = 10.0f;
    (*cpu_cloud.intensities)[1] = 2.0f;
    (*cpu_cloud.intensities)[2] = 1.0f;

    const float scale = 1.0f;
    const float min_intensity = 0.0f;
    const float max_intensity = 10.0f;
    auto clamp = [min_intensity, max_intensity](float x) { return std::clamp(x, min_intensity, max_intensity); };

    {
        sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
        sycl_points::algorithms::intensity_correction::correct_intensity(shared_cloud, 2.0f, scale, min_intensity,
                                                                         max_intensity);

        ASSERT_TRUE(shared_cloud.has_intensity());
        EXPECT_NEAR((*shared_cloud.intensities)[0], clamp(10.0f * std::pow(1.0f, 2.0f) * scale), 1e-5f);
        EXPECT_NEAR((*shared_cloud.intensities)[1], clamp(2.0f * std::pow(5.0f, 2.0f) * scale), 1e-5f);
        EXPECT_NEAR((*shared_cloud.intensities)[2], clamp(1.0f * std::pow(3.0f, 2.0f) * scale), 1e-5f);
    }
    {
        sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
        sycl_points::algorithms::intensity_correction::correct_intensity(shared_cloud, 1.0f, scale, min_intensity,
                                                                         max_intensity);
        ASSERT_TRUE(shared_cloud.has_intensity());
        EXPECT_NEAR((*shared_cloud.intensities)[0], clamp(10.0f * 1.0f * scale), 1e-5f);
        EXPECT_NEAR((*shared_cloud.intensities)[1], clamp(2.0f * 5.0f * scale), 1e-5f);
        EXPECT_NEAR((*shared_cloud.intensities)[2], clamp(1.0f * 3.0f * scale), 1e-5f);
    }
}

TEST(IntensityCorrectionTest, ThrowsWithoutIntensityField) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(2);
    (*cpu_cloud.points)[0] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.points)[1] = sycl_points::PointType(0.0f, 1.0f, 0.0f, 1.0f);

    cpu_cloud.intensities->clear();

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);

    EXPECT_THROW(sycl_points::algorithms::intensity_correction::correct_intensity(shared_cloud), std::runtime_error);
}

TEST(IntensityCorrectionTest, ThrowsNegativeExponent) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3);
    cpu_cloud.intensities->resize(3);

    (*cpu_cloud.points)[0] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);  // distance 1
    (*cpu_cloud.points)[1] = sycl_points::PointType(0.0f, 3.0f, 4.0f, 1.0f);  // distance 5
    (*cpu_cloud.points)[2] = sycl_points::PointType(1.0f, 2.0f, 2.0f, 1.0f);  // distance 3

    (*cpu_cloud.intensities)[0] = 10.0f;
    (*cpu_cloud.intensities)[1] = 2.0f;
    (*cpu_cloud.intensities)[2] = 1.0f;

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);

    EXPECT_THROW(sycl_points::algorithms::intensity_correction::correct_intensity(shared_cloud, -1.0f),
                 std::runtime_error);
}

TEST(IntensityCorrectionTest, AppliesReferenceDistanceNormalization) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    cpu_cloud.intensities->resize(1);
    (*cpu_cloud.points)[0] = sycl_points::PointType(2.0f, 0.0f, 0.0f, 1.0f);  // distance 2
    (*cpu_cloud.intensities)[0] = 3.0f;

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
    sycl_points::algorithms::intensity_correction::correct_intensity(
        shared_cloud, 2.0f, 1.0f, 0.0f, 1000.0f, 2.0f);

    ASSERT_TRUE(shared_cloud.has_intensity());
    // (dist/reference_distance)^exp = (2/2)^2 = 1
    EXPECT_NEAR((*shared_cloud.intensities)[0], 3.0f, 1e-5f);
}

TEST(IntensityCorrectionTest, ThrowsNonPositiveReferenceDistance) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    cpu_cloud.intensities->resize(1);
    (*cpu_cloud.points)[0] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.intensities)[0] = 1.0f;

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
    EXPECT_THROW(
        sycl_points::algorithms::intensity_correction::correct_intensity(shared_cloud, 2.0f, 1.0f, 0.0f, 1.0f, 0.0f),
        std::runtime_error);
}

// --- Normal-aware intensity correction tests ---

TEST(IntensityCorrectionWithNormalTest, NormalPerpendicular_CosThetaEqualsOne) {
    // Normal aligned with beam direction → cos(theta) = 1 → same as distance-only correction
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(2);
    cpu_cloud.intensities->resize(2);
    cpu_cloud.normals->resize(2);

    // Point at (1,0,0), distance=1. Normal points away from origin along x → |n.p|/dist = 1
    (*cpu_cloud.points)[0] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.normals)[0] = sycl_points::Normal(1.0f, 0.0f, 0.0f, 0.0f);
    (*cpu_cloud.intensities)[0] = 4.0f;

    // Point at (0,3,4), distance=5. Normal = (0, 0.6, 0.8) → |n.p|/dist = |0*0+0.6*3+0.8*4|/5 = |1.8+3.2|/5 = 1.0
    (*cpu_cloud.points)[1] = sycl_points::PointType(0.0f, 3.0f, 4.0f, 1.0f);
    (*cpu_cloud.normals)[1] = sycl_points::Normal(0.0f, 0.6f, 0.8f, 0.0f);
    (*cpu_cloud.intensities)[1] = 2.0f;

    const float scale = 1.0f;
    const float min_intensity = 0.0f;
    const float max_intensity = 1000.0f;
    const float min_cos_theta = 0.1f;
    const float exponent = 2.0f;

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
    sycl_points::algorithms::intensity_correction::correct_intensity_with_normal(
        shared_cloud, exponent, scale, min_intensity, max_intensity, min_cos_theta);

    ASSERT_TRUE(shared_cloud.has_intensity());
    // cos(theta)=1 → same as distance-only: I * dist^exp / 1.0 * scale
    EXPECT_NEAR((*shared_cloud.intensities)[0], std::clamp(4.0f * std::pow(1.0f, 2.0f) / 1.0f * scale,
                                                           min_intensity, max_intensity), 1e-4f);
    EXPECT_NEAR((*shared_cloud.intensities)[1], std::clamp(2.0f * std::pow(5.0f, 2.0f) / 1.0f * scale,
                                                           min_intensity, max_intensity), 1e-4f);
}

TEST(IntensityCorrectionWithNormalTest, IncidenceAngle45Degrees) {
    // Normal at 45° to beam direction → cos(theta) = 1/sqrt(2) → intensity amplified by sqrt(2)
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    cpu_cloud.intensities->resize(1);
    cpu_cloud.normals->resize(1);

    // Point at (1,0,0), distance=1. Normal = (1/√2, 1/√2, 0) → |n.p|/dist = |1/√2|/1 = 1/√2
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    (*cpu_cloud.points)[0] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.normals)[0] = sycl_points::Normal(inv_sqrt2, inv_sqrt2, 0.0f, 0.0f);
    (*cpu_cloud.intensities)[0] = 4.0f;

    const float scale = 1.0f;
    const float min_intensity = 0.0f;
    const float max_intensity = 1000.0f;
    const float min_cos_theta = 0.1f;
    const float exponent = 2.0f;

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
    sycl_points::algorithms::intensity_correction::correct_intensity_with_normal(
        shared_cloud, exponent, scale, min_intensity, max_intensity, min_cos_theta);

    ASSERT_TRUE(shared_cloud.has_intensity());
    // I' = I * dist^exp / cos(theta) = 4.0 * 1.0 / (1/√2) = 4 * √2
    const float expected = std::clamp(4.0f * std::pow(1.0f, 2.0f) / inv_sqrt2 * scale, min_intensity, max_intensity);
    EXPECT_NEAR((*shared_cloud.intensities)[0], expected, 1e-4f);
}

TEST(IntensityCorrectionWithNormalTest, GrazingAngleClampsToMinCosTheta) {
    // Normal perpendicular to beam → cos(theta)=0, clamped to min_cos_theta
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    cpu_cloud.intensities->resize(1);
    cpu_cloud.normals->resize(1);

    // Point at (0,0,1), distance=1. Normal = (1,0,0) → |n.p|/dist = |0|/1 = 0 → clamped to min_cos_theta
    (*cpu_cloud.points)[0] = sycl_points::PointType(0.0f, 0.0f, 1.0f, 1.0f);
    (*cpu_cloud.normals)[0] = sycl_points::Normal(1.0f, 0.0f, 0.0f, 0.0f);
    (*cpu_cloud.intensities)[0] = 2.0f;

    const float scale = 1.0f;
    const float min_intensity = 0.0f;
    const float max_intensity = 1000.0f;
    const float min_cos_theta = 0.5f;
    const float exponent = 2.0f;

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
    sycl_points::algorithms::intensity_correction::correct_intensity_with_normal(
        shared_cloud, exponent, scale, min_intensity, max_intensity, min_cos_theta);

    ASSERT_TRUE(shared_cloud.has_intensity());
    // cos(theta) clamped to 0.5 → I' = 2.0 * 1^2 / 0.5 = 4.0
    const float expected = std::clamp(2.0f * std::pow(1.0f, 2.0f) / min_cos_theta * scale, min_intensity, max_intensity);
    EXPECT_NEAR((*shared_cloud.intensities)[0], expected, 1e-4f);
}

TEST(IntensityCorrectionWithNormalTest, ThrowsWithoutNormalField) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    cpu_cloud.intensities->resize(1);
    (*cpu_cloud.points)[0] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.intensities)[0] = 1.0f;
    // normals NOT set

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
    EXPECT_THROW(sycl_points::algorithms::intensity_correction::correct_intensity_with_normal(shared_cloud),
                 std::runtime_error);
}

TEST(IntensityCorrectionWithNormalTest, ThrowsWithNonPositiveMinCosTheta) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    cpu_cloud.intensities->resize(1);
    cpu_cloud.normals->resize(1);
    (*cpu_cloud.points)[0] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.intensities)[0] = 1.0f;
    (*cpu_cloud.normals)[0] = sycl_points::Normal(1.0f, 0.0f, 0.0f, 0.0f);

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
    EXPECT_THROW(
        sycl_points::algorithms::intensity_correction::correct_intensity_with_normal(
            shared_cloud, 2.0f, 1.0f, 0.0f, 1.0f, 0.0f),  // min_cos_theta = 0 → invalid
        std::runtime_error);
}

TEST(IntensityCorrectionWithNormalTest, ThrowsWithNonPositiveReferenceDistance) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    cpu_cloud.intensities->resize(1);
    cpu_cloud.normals->resize(1);
    (*cpu_cloud.points)[0] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.intensities)[0] = 1.0f;
    (*cpu_cloud.normals)[0] = sycl_points::Normal(1.0f, 0.0f, 0.0f, 0.0f);

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
    EXPECT_THROW(
        sycl_points::algorithms::intensity_correction::correct_intensity_with_normal(
            shared_cloud, 2.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.0f),
        std::runtime_error);
}
