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

TEST(IntensityCorrectionTest, ThrowsNonPositiveRefDistance) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    cpu_cloud.intensities->resize(1);
    (*cpu_cloud.points)[0] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.intensities)[0] = 1.0f;

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);

    EXPECT_THROW(
        sycl_points::algorithms::intensity_correction::correct_intensity(shared_cloud, 2.0f, 1.0f, 0.0f, 100.0f, 0.0f),
        std::runtime_error);
    EXPECT_THROW(
        sycl_points::algorithms::intensity_correction::correct_intensity(shared_cloud, 2.0f, 1.0f, 0.0f, 100.0f, -1.0f),
        std::runtime_error);
}

TEST(IntensityCorrectionTest, RefDistanceNormalization) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(2);
    cpu_cloud.intensities->resize(2);

    // distance 5.0, reference distance 5.0 → dist_factor = (5/5)^2 = 1.0
    (*cpu_cloud.points)[0] = sycl_points::PointType(0.0f, 3.0f, 4.0f, 1.0f);  // distance 5
    // distance 1.0, reference distance 5.0 → dist_factor = (1/5)^2 = 0.04
    (*cpu_cloud.points)[1] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);  // distance 1

    (*cpu_cloud.intensities)[0] = 10.0f;
    (*cpu_cloud.intensities)[1] = 10.0f;

    const float scale = 1.0f;
    const float ref_dist = 5.0f;
    const float min_i = 0.0f;
    const float max_i = 1000.0f;

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
    sycl_points::algorithms::intensity_correction::correct_intensity(shared_cloud, 2.0f, scale, min_i, max_i, ref_dist);

    ASSERT_TRUE(shared_cloud.has_intensity());
    EXPECT_NEAR((*shared_cloud.intensities)[0], 10.0f * 1.0f * scale, 1e-4f);   // (5/5)^2 = 1
    EXPECT_NEAR((*shared_cloud.intensities)[1], 10.0f * 0.04f * scale, 1e-4f);  // (1/5)^2 = 0.04
}

TEST(IntensityCorrectionTest, AngleCorrectionWithNormals) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(2);
    cpu_cloud.intensities->resize(2);
    cpu_cloud.normals->resize(2);

    // Point along z-axis, normal also along z: cos(theta) = 1 → angle_factor = 1
    (*cpu_cloud.points)[0] = sycl_points::PointType(0.0f, 0.0f, 5.0f, 1.0f);
    (*cpu_cloud.normals)[0] = sycl_points::Normal(0.0f, 0.0f, 1.0f, 0.0f);

    // Point along z-axis, normal at 60° → cos(theta) = 0.5 → angle_factor = 1/0.5 = 2
    (*cpu_cloud.points)[1] = sycl_points::PointType(0.0f, 0.0f, 5.0f, 1.0f);
    (*cpu_cloud.normals)[1] = sycl_points::Normal(0.0f, std::sin(M_PIf / 3.0f), std::cos(M_PIf / 3.0f), 0.0f);

    (*cpu_cloud.intensities)[0] = 1.0f;
    (*cpu_cloud.intensities)[1] = 1.0f;

    const float scale = 1.0f;
    const float ref_dist = 5.0f;  // dist/ref = 1 → dist_factor = 1
    const float min_i = 0.0f;
    const float max_i = 1000.0f;
    const float angle_exp = 1.0f;

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);
    sycl_points::algorithms::intensity_correction::correct_intensity(shared_cloud, 2.0f, scale, min_i, max_i, ref_dist,
                                                                     angle_exp);

    ASSERT_TRUE(shared_cloud.has_intensity());
    EXPECT_NEAR((*shared_cloud.intensities)[0], 1.0f, 1e-4f);  // angle_factor = 1/cos(0) = 1
    EXPECT_NEAR((*shared_cloud.intensities)[1], 2.0f, 1e-4f);  // angle_factor = 1/cos(60°) = 2
}

TEST(IntensityCorrectionTest, AngleCorrectionSkippedWithoutNormals) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(1);
    cpu_cloud.intensities->resize(1);
    (*cpu_cloud.points)[0] = sycl_points::PointType(1.0f, 0.0f, 0.0f, 1.0f);
    (*cpu_cloud.intensities)[0] = 5.0f;

    sycl_points::PointCloudShared shared_cloud(queue, cpu_cloud);

    // angle_exponent > 0 but no normals: should not throw, angle correction is silently skipped
    EXPECT_NO_THROW(sycl_points::algorithms::intensity_correction::correct_intensity(shared_cloud, 2.0f, 1.0f, 0.0f,
                                                                                     1000.0f, 1.0f, 1.0f));

    // Result should be distance-only correction: 5 * (1/1)^2 = 5
    EXPECT_NEAR((*shared_cloud.intensities)[0], 5.0f, 1e-4f);
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
