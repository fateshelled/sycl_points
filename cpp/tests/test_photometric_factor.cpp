#include <gtest/gtest.h>

#include "sycl_points/algorithms/feature/photometric_gradient.hpp"
#include "sycl_points/algorithms/knn/result.hpp"
#include "sycl_points/algorithms/registration/photometric_factor.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"
#include "sycl_points/utils/eigen_utils.hpp"

namespace sycl_points::algorithms::registration::kernel {
namespace {

TEST(PhotometricFactorTest, IntensityJacobianRespectsTangentProjection) {
    const Eigen::Matrix4f T_eigen = Eigen::Matrix4f::Identity();
    const auto T = eigen_utils::to_sycl_vec(T_eigen);

    PointType source_pt;
    source_pt << 1.0f, -2.0f, 3.0f, 1.0f;
    PointType target_pt = source_pt;

    Normal target_normal;
    target_normal << 0.0f, 0.0f, 1.0f, 0.0f;

    // Pure normal-direction intensity gradient should be removed by tangent-plane projection.
    IntensityGradient target_intensity_grad;
    target_intensity_grad << 0.0f, 0.0f, 5.0f;

    const auto linearized =
        linearize_intensity(T, source_pt, target_pt, 10.0f, 10.0f, target_normal, target_intensity_grad);

    EXPECT_NEAR(linearized.H.norm(), 0.0f, 1e-6f);
    EXPECT_NEAR(linearized.b.norm(), 0.0f, 1e-6f);
    EXPECT_NEAR(linearized.squared_error, 0.0f, 1e-6f);
}

TEST(PhotometricFactorTest, ColorJacobianRespectsTangentProjection) {
    const Eigen::Matrix4f T_eigen = Eigen::Matrix4f::Identity();
    const auto T = eigen_utils::to_sycl_vec(T_eigen);

    PointType source_pt;
    source_pt << -0.5f, 0.2f, 1.3f, 1.0f;
    PointType target_pt = source_pt;

    Normal target_normal;
    target_normal << 0.0f, 0.0f, 1.0f, 0.0f;

    RGBType source_rgb;
    source_rgb << 10.0f, 20.0f, 30.0f, 0.0f;
    RGBType target_rgb = source_rgb;

    // Color gradient aligned with the normal direction for all channels.
    ColorGradient target_rgb_grad = ColorGradient::Zero();
    target_rgb_grad(0, 2) = 1.0f;
    target_rgb_grad(1, 2) = -2.0f;
    target_rgb_grad(2, 2) = 0.5f;

    const auto linearized =
        linearize_color(T, source_pt, target_pt, source_rgb, target_rgb, target_normal, target_rgb_grad);

    EXPECT_NEAR(linearized.H.norm(), 0.0f, 1e-6f);
    EXPECT_NEAR(linearized.b.norm(), 0.0f, 1e-6f);
    EXPECT_NEAR(linearized.squared_error, 0.0f, 1e-6f);
}

TEST(PhotometricFactorTest, ColorErrorMatchesLinearizedSquaredError) {
    const Eigen::Matrix4f T_eigen = Eigen::Matrix4f::Identity();
    const auto T = eigen_utils::to_sycl_vec(T_eigen);

    PointType source_pt;
    source_pt << 1.0f, 0.5f, 0.25f, 1.0f;
    PointType target_pt;
    target_pt << 0.8f, 0.6f, 0.20f, 1.0f;

    Normal target_normal;
    target_normal << 0.0f, 0.0f, 1.0f, 0.0f;

    RGBType source_rgb;
    source_rgb << 80.0f, 60.0f, 40.0f, 0.0f;
    RGBType target_rgb;
    target_rgb << 78.0f, 58.0f, 39.0f, 0.0f;

    ColorGradient target_rgb_grad = ColorGradient::Zero();
    target_rgb_grad(0, 0) = 0.2f;
    target_rgb_grad(0, 1) = -0.1f;
    target_rgb_grad(1, 0) = -0.05f;
    target_rgb_grad(1, 1) = 0.15f;
    target_rgb_grad(2, 0) = 0.1f;
    target_rgb_grad(2, 1) = 0.05f;

    const auto linearized =
        linearize_color(T, source_pt, target_pt, source_rgb, target_rgb, target_normal, target_rgb_grad);
    const float error = calculate_color_error(T, source_pt, target_pt, source_rgb, target_rgb, target_normal, target_rgb_grad);

    EXPECT_NEAR(error, linearized.squared_error, 1e-6f);
}

TEST(PhotometricFactorTest, ColorResidualZeroWhenSourceMatchesWarpedTarget) {
    const Eigen::Matrix4f T_eigen = Eigen::Matrix4f::Identity();
    const auto T = eigen_utils::to_sycl_vec(T_eigen);

    PointType source_pt;
    source_pt << 1.0f, 0.0f, 0.0f, 1.0f;
    PointType target_pt;
    target_pt << 0.0f, 0.0f, 0.0f, 1.0f;

    Normal target_normal;
    target_normal << 0.0f, 0.0f, 1.0f, 0.0f;

    ColorGradient target_rgb_grad = ColorGradient::Zero();
    target_rgb_grad(0, 0) = 2.0f;
    target_rgb_grad(1, 0) = -1.0f;
    target_rgb_grad(2, 0) = 0.5f;

    RGBType target_rgb;
    target_rgb << 10.0f, 20.0f, 30.0f, 0.0f;
    RGBType source_rgb;
    source_rgb << 12.0f, 19.0f, 30.5f, 0.0f;  // target_rgb + target_rgb_grad * [1, 0, 0]^T

    const auto linearized =
        linearize_color(T, source_pt, target_pt, source_rgb, target_rgb, target_normal, target_rgb_grad);
    const float error = calculate_color_error(T, source_pt, target_pt, source_rgb, target_rgb, target_normal, target_rgb_grad);

    EXPECT_NEAR(linearized.squared_error, 0.0f, 1e-6f);
    EXPECT_NEAR(error, 0.0f, 1e-6f);
    EXPECT_NEAR(linearized.b.norm(), 0.0f, 1e-6f);
}

TEST(PhotometricFactorTest, IntensityResidualZeroWhenSourceMatchesWarpedTarget) {
    const Eigen::Matrix4f T_eigen = Eigen::Matrix4f::Identity();
    const auto T = eigen_utils::to_sycl_vec(T_eigen);

    PointType source_pt;
    source_pt << 0.0f, 1.0f, 0.0f, 1.0f;
    PointType target_pt;
    target_pt << 0.0f, 0.0f, 0.0f, 1.0f;

    Normal target_normal;
    target_normal << 0.0f, 0.0f, 1.0f, 0.0f;

    IntensityGradient target_intensity_grad;
    target_intensity_grad << 0.0f, 3.0f, 0.0f;

    const float target_intensity = 5.0f;
    const float source_intensity = 8.0f;  // target + grad dot [0, 1, 0]

    const auto linearized = linearize_intensity(T, source_pt, target_pt, source_intensity, target_intensity,
                                                 target_normal, target_intensity_grad);
    const float error = calculate_intensity_error(T, source_pt, target_pt, source_intensity, target_intensity,
                                                  target_normal, target_intensity_grad);

    EXPECT_NEAR(linearized.squared_error, 0.0f, 1e-6f);
    EXPECT_NEAR(error, 0.0f, 1e-6f);
    EXPECT_NEAR(linearized.b.norm(), 0.0f, 1e-6f);
}

}  // namespace
}  // namespace sycl_points::algorithms::registration::kernel

namespace {

TEST(IntensityZScoreCalculatorTest, ComputesNeighborhoodZScore) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(3, sycl_points::PointType::Zero());
    cpu_cloud.intensities->resize(3);
    (*cpu_cloud.intensities)[0] = 10.0f;
    (*cpu_cloud.intensities)[1] = 4.0f;
    (*cpu_cloud.intensities)[2] = 16.0f;

    sycl_points::PointCloudShared cloud(queue, cpu_cloud);

    sycl_points::algorithms::knn::KNNResult neighbors;
    neighbors.allocate(queue, cloud.size(), 2);
    // Each point uses itself and one neighbor.
    (*neighbors.indices)[0] = 0;
    (*neighbors.indices)[1] = 1;
    (*neighbors.indices)[2] = 1;
    (*neighbors.indices)[3] = 2;
    (*neighbors.indices)[4] = 2;
    (*neighbors.indices)[5] = 0;

    sycl_points::algorithms::intensity_z_score::IntensityZScoreCalculator calculator(queue);
    calculator.compute_async(cloud, neighbors).wait_and_throw();

    const auto z_scores = calculator.intensity_z_scores();
    ASSERT_EQ(z_scores->size(), 3);
    EXPECT_NEAR((*z_scores)[0], 1.0f, 1e-5f);   // (10 - mean(10,4)) / std(10,4)
    EXPECT_NEAR((*z_scores)[1], -1.0f, 1e-5f);  // (4 - mean(4,16)) / std(4,16)
    EXPECT_NEAR((*z_scores)[2], 1.0f, 1e-5f);   // (16 - mean(16,10)) / std(16,10)
}

TEST(IntensityZScoreCalculatorTest, IgnoresInvalidNeighborIndices) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(2, sycl_points::PointType::Zero());
    cpu_cloud.intensities->resize(2);
    (*cpu_cloud.intensities)[0] = 8.0f;
    (*cpu_cloud.intensities)[1] = 2.0f;

    sycl_points::PointCloudShared cloud(queue, cpu_cloud);

    sycl_points::algorithms::knn::KNNResult neighbors;
    neighbors.allocate(queue, cloud.size(), 3);
    (*neighbors.indices)[0] = -1;
    (*neighbors.indices)[1] = 1;
    (*neighbors.indices)[2] = 0;
    (*neighbors.indices)[3] = -1;
    (*neighbors.indices)[4] = -1;
    (*neighbors.indices)[5] = -1;

    sycl_points::algorithms::intensity_z_score::IntensityZScoreCalculator calculator(queue);
    calculator.compute_async(cloud, neighbors).wait_and_throw();

    const auto z_scores = calculator.intensity_z_scores();
    ASSERT_EQ(z_scores->size(), 2);
    EXPECT_NEAR((*z_scores)[0], 1.0f, 1e-5f);  // (8 - mean(2,8)) / std(2,8)
    EXPECT_NEAR((*z_scores)[1], 0.0f, 1e-5f);  // no valid neighbors
}

TEST(IntensityZScoreCalculatorTest, ThrowsWhenNeighborQuerySizeMismatch) {
    sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
    sycl_points::sycl_utils::DeviceQueue queue(device);

    sycl_points::PointCloudCPU cpu_cloud;
    cpu_cloud.points->resize(2, sycl_points::PointType::Zero());
    cpu_cloud.intensities->resize(2, 1.0f);

    sycl_points::PointCloudShared cloud(queue, cpu_cloud);

    sycl_points::algorithms::knn::KNNResult neighbors;
    neighbors.allocate(queue, 1, 1);
    (*neighbors.indices)[0] = 0;

    sycl_points::algorithms::intensity_z_score::IntensityZScoreCalculator calculator(queue);
    EXPECT_THROW(calculator.compute_async(cloud, neighbors), std::runtime_error);
}

}  // namespace
