#include <gtest/gtest.h>

#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/algorithms/registration/registration_pipeline.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {
namespace {

class DummyKNN : public knn::KNNBase {
public:
    sycl_utils::events knn_search_async(const PointCloudShared&, const size_t, knn::KNNResult&,
                                        const std::vector<sycl::event>& = std::vector<sycl::event>(),
                                        const TransformMatrix& = TransformMatrix::Identity()) const override {
        return sycl_utils::events();
    }
};

PointCloudShared make_cloud(const sycl_utils::DeviceQueue& queue, size_t size) {
    PointCloudShared cloud(queue);
    cloud.points->resize(size);
    cloud.intensities->resize(size);
    cloud.timestamp_offsets->resize(size);
    for (size_t i = 0; i < size; ++i) {
        cloud.points->data()[i] =
            PointType(static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2), 1.0f);
        cloud.intensities->data()[i] = static_cast<float>(i);
        cloud.timestamp_offsets->data()[i] = static_cast<float>(i) * 0.1f;
    }
    cloud.start_time_ms = 1.0;
    cloud.end_time_ms = 2.0;
    return cloud;
}

TEST(RegistrationPipelineTest, RandomSamplingLimitsRegistrationInputSize) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);
    RegistrationPipelineParams params;
    params.random_sampling.enable = true;
    params.random_sampling.num = 3;

    size_t aligned_source_size = 0;
    bool source_has_intensity = false;
    bool source_has_timestamps = false;
    auto aligner = [&](const PointCloudShared& source, const PointCloudShared&, const knn::KNNBase&,
                       const TransformMatrix&, const Registration::ExecutionOptions&) {
        aligned_source_size = source.size();
        source_has_intensity = source.has_intensity();
        source_has_timestamps = source.has_timestamps();
        RegistrationResult result;
        result.inlier = static_cast<uint32_t>(source.size());
        return result;
    };

    RegistrationPipeline pipeline(aligner, params);
    const auto source = make_cloud(queue, 6);
    const auto target = make_cloud(queue, 4);
    DummyKNN knn;

    const auto result = pipeline.align(source, target, knn);

    EXPECT_EQ(result.inlier, 3U);
    EXPECT_EQ(aligned_source_size, 3U);
    ASSERT_NE(pipeline.get_registration_input_point_cloud(), nullptr);
    EXPECT_EQ(pipeline.get_registration_input_point_cloud()->size(), 3U);
    EXPECT_TRUE(source_has_intensity);
    EXPECT_TRUE(source_has_timestamps);
    EXPECT_TRUE(pipeline.get_registration_input_point_cloud()->has_intensity());
    EXPECT_TRUE(pipeline.get_registration_input_point_cloud()->has_timestamps());
}

TEST(RegistrationPipelineTest, RandomSamplingCanBeDisabled) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);
    RegistrationPipelineParams params;
    params.random_sampling.enable = false;
    params.random_sampling.num = 2;

    size_t aligned_source_size = 0;
    auto aligner = [&](const PointCloudShared& source, const PointCloudShared&, const knn::KNNBase&,
                       const TransformMatrix&, const Registration::ExecutionOptions&) {
        aligned_source_size = source.size();
        RegistrationResult result;
        result.inlier = static_cast<uint32_t>(source.size());
        return result;
    };

    RegistrationPipeline pipeline(aligner, params);
    const auto source = make_cloud(queue, 5);
    const auto target = make_cloud(queue, 4);
    DummyKNN knn;

    pipeline.align(source, target, knn);

    EXPECT_EQ(aligned_source_size, 5U);
    ASSERT_NE(pipeline.get_registration_input_point_cloud(), nullptr);
    EXPECT_EQ(pipeline.get_registration_input_point_cloud()->size(), 5U);
}

TEST(RegistrationPipelineTest, RandomSamplingDoesNotShrinkSmallCloud) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);
    RegistrationPipelineParams params;
    params.random_sampling.enable = true;
    params.random_sampling.num = 8;

    size_t aligned_source_size = 0;
    auto aligner = [&](const PointCloudShared& source, const PointCloudShared&, const knn::KNNBase&,
                       const TransformMatrix&, const Registration::ExecutionOptions&) {
        aligned_source_size = source.size();
        RegistrationResult result;
        result.inlier = static_cast<uint32_t>(source.size());
        return result;
    };

    RegistrationPipeline pipeline(aligner, params);
    const auto source = make_cloud(queue, 5);
    const auto target = make_cloud(queue, 4);
    DummyKNN knn;

    pipeline.align(source, target, knn);

    EXPECT_EQ(aligned_source_size, 5U);
    ASSERT_NE(pipeline.get_registration_input_point_cloud(), nullptr);
    EXPECT_EQ(pipeline.get_registration_input_point_cloud()->size(), 5U);
}

TEST(RegistrationPipelineTest, AccessorTracksMostRecentAlignInput) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);
    RegistrationPipelineParams params;
    params.random_sampling.enable = true;
    params.random_sampling.num = 2;

    auto aligner = [&](const PointCloudShared& source, const PointCloudShared&, const knn::KNNBase&,
                       const TransformMatrix&, const Registration::ExecutionOptions&) {
        RegistrationResult result;
        result.inlier = static_cast<uint32_t>(source.size());
        return result;
    };

    RegistrationPipeline pipeline(aligner, params);
    DummyKNN knn;

    pipeline.align(make_cloud(queue, 5), make_cloud(queue, 3), knn);
    ASSERT_NE(pipeline.get_registration_input_point_cloud(), nullptr);
    EXPECT_EQ(pipeline.get_registration_input_point_cloud()->size(), 2U);

    pipeline.align(make_cloud(queue, 1), make_cloud(queue, 3), knn);
    ASSERT_NE(pipeline.get_registration_input_point_cloud(), nullptr);
    EXPECT_EQ(pipeline.get_registration_input_point_cloud()->size(), 1U);
}

TEST(RegistrationPipelineTest, AccessorsReturnNullBeforeAlign) {
    RegistrationPipeline pipeline([](const PointCloudShared&, const PointCloudShared&, const knn::KNNBase&,
                                     const TransformMatrix&,
                                     const Registration::ExecutionOptions&) { return RegistrationResult{}; });

    EXPECT_EQ(pipeline.get_registration_input_point_cloud(), nullptr);
    EXPECT_EQ(pipeline.get_deskewed_point_cloud(), nullptr);

    VelocityUpdatePipeline velocity_pipeline(
        [](const PointCloudShared&, const PointCloudShared&, const knn::KNNBase&, const TransformMatrix&,
           const Registration::ExecutionOptions&) { return RegistrationResult{}; },
        1, false);
    EXPECT_EQ(velocity_pipeline.get_deskewed_point_cloud(), nullptr);
}

TEST(RegistrationPipelineTest, VelocityUpdatePipelineExposesMostRecentDeskewedPointCloud) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);
    const auto source = make_cloud(queue, 4);
    const auto target = make_cloud(queue, 3);
    DummyKNN knn;

    auto aligner = [&](const PointCloudShared& source_arg, const PointCloudShared&, const knn::KNNBase&,
                       const TransformMatrix&, const Registration::ExecutionOptions&) {
        RegistrationResult result;
        result.T = Eigen::Translation3f(1.0f, 0.0f, 0.0f) * Eigen::Isometry3f::Identity();
        result.inlier = static_cast<uint32_t>(source_arg.size());
        return result;
    };

    VelocityUpdatePipeline pipeline(aligner, 1, false);
    Registration::ExecutionOptions options;
    options.dt = 1.0f;
    options.prev_pose = TransformMatrix::Identity();

    pipeline.align(source, target, knn, TransformMatrix::Identity(), options);

    const auto* deskewed = pipeline.get_deskewed_point_cloud();
    ASSERT_NE(deskewed, nullptr);
    EXPECT_EQ(deskewed->size(), source.size());
    EXPECT_TRUE(deskewed->has_timestamps());
    EXPECT_NE(deskewed->points.get(), source.points.get());
    EXPECT_NE(deskewed->intensities.get(), source.intensities.get());
    EXPECT_NE(deskewed->timestamp_offsets.get(), source.timestamp_offsets.get());
}

TEST(RegistrationPipelineTest, RegistrationPipelineExposesDeskewedPointCloud) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);
    RegistrationPipelineParams params;
    params.velocity_update.enable = true;
    params.velocity_update.iter = 1;

    auto aligner = [&](const PointCloudShared& source_arg, const PointCloudShared&, const knn::KNNBase&,
                       const TransformMatrix&, const Registration::ExecutionOptions&) {
        RegistrationResult result;
        result.T = Eigen::Translation3f(0.5f, 0.0f, 0.0f) * Eigen::Isometry3f::Identity();
        result.inlier = static_cast<uint32_t>(source_arg.size());
        return result;
    };

    RegistrationPipeline pipeline(aligner, params);
    Registration::ExecutionOptions options;
    options.dt = 1.0f;
    options.prev_pose = TransformMatrix::Identity();
    DummyKNN knn;
    const auto source = make_cloud(queue, 5);

    pipeline.align(source, make_cloud(queue, 3), knn, TransformMatrix::Identity(), options);

    const auto* deskewed = pipeline.get_deskewed_point_cloud();
    ASSERT_NE(deskewed, nullptr);
    EXPECT_EQ(deskewed->size(), 5U);
    EXPECT_NE(deskewed->points.get(), source.points.get());
    EXPECT_NE(deskewed->intensities.get(), source.intensities.get());
    EXPECT_NE(deskewed->timestamp_offsets.get(), source.timestamp_offsets.get());
}

TEST(RegistrationPipelineTest, DeskewedAccessorFallsBackToRegistrationInputWithoutVelocityUpdate) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);
    RegistrationPipelineParams params;
    params.random_sampling.enable = true;
    params.random_sampling.num = 2;

    auto aligner = [&](const PointCloudShared& source_arg, const PointCloudShared&, const knn::KNNBase&,
                       const TransformMatrix&, const Registration::ExecutionOptions&) {
        RegistrationResult result;
        result.inlier = static_cast<uint32_t>(source_arg.size());
        return result;
    };

    RegistrationPipeline pipeline(aligner, params);
    DummyKNN knn;

    pipeline.align(make_cloud(queue, 5), make_cloud(queue, 3), knn);

    EXPECT_EQ(pipeline.get_deskewed_point_cloud(), pipeline.get_registration_input_point_cloud());
    ASSERT_NE(pipeline.get_deskewed_point_cloud(), nullptr);
    EXPECT_EQ(pipeline.get_deskewed_point_cloud()->size(), 2U);
}

}  // namespace
}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
