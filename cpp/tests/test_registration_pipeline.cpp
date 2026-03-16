#include <gtest/gtest.h>

#include <cmath>

#include "sycl_points/algorithms/knn/knn.hpp"
#include "sycl_points/algorithms/registration/registration.hpp"
#include "sycl_points/algorithms/registration/registration_pipeline.hpp"
#include "sycl_points/algorithms/robust/robust.hpp"
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

class CountingNearestKNN : public knn::KNNBase {
public:
    mutable size_t call_count = 0;

    sycl_utils::events knn_search_async(const PointCloudShared& queries, const size_t k, knn::KNNResult& result,
                                        const std::vector<sycl::event>& = std::vector<sycl::event>(),
                                        const TransformMatrix& transT = TransformMatrix::Identity()) const override {
        ++call_count;
        result.allocate(queries.queue, queries.size(), k);

        const auto T = transT;
        for (size_t i = 0; i < queries.size(); ++i) {
            PointType transformed = T * queries.points->at(i);
            float best_distance = std::numeric_limits<float>::max();
            int32_t best_index = -1;
            for (size_t j = 0; j < target_->size(); ++j) {
                const auto& target_point = target_->points->at(j);
                const float dx = transformed.x() - target_point.x();
                const float dy = transformed.y() - target_point.y();
                const float dz = transformed.z() - target_point.z();
                const float squared_distance = dx * dx + dy * dy + dz * dz;
                if (squared_distance < best_distance) {
                    best_distance = squared_distance;
                    best_index = static_cast<int32_t>(j);
                }
            }
            result.indices->at(i) = best_index;
            result.distances->at(i) = best_distance;
        }
        return sycl_utils::events();
    }

    void set_target(const PointCloudShared& target) { target_ = &target; }

private:
    const PointCloudShared* target_ = nullptr;
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

PointCloudShared make_registration_source(const sycl_utils::DeviceQueue& queue) {
    PointCloudShared cloud(queue);
    cloud.points->resize(3);
    cloud.timestamp_offsets->resize(3);
    cloud.points->at(0) = PointType(0.0f, 0.0f, 0.0f, 1.0f);
    cloud.points->at(1) = PointType(1.0f, 0.0f, 0.0f, 1.0f);
    cloud.points->at(2) = PointType(5.0f, 0.0f, 0.0f, 1.0f);
    cloud.timestamp_offsets->at(0) = 0.0f;
    cloud.timestamp_offsets->at(1) = 0.5f;
    cloud.timestamp_offsets->at(2) = 1.0f;
    cloud.start_time_ms = 0.0;
    cloud.end_time_ms = 1.0;
    return cloud;
}

PointCloudShared make_registration_target(const sycl_utils::DeviceQueue& queue) {
    PointCloudShared cloud(queue);
    cloud.points->resize(2);
    cloud.points->at(0) = PointType(0.0f, 0.0f, 0.0f, 1.0f);
    cloud.points->at(1) = PointType(1.0f, 0.0f, 0.0f, 1.0f);
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

    pipeline::VelocityUpdateAligner velocity_pipeline(
        [](const PointCloudShared&, const PointCloudShared&, const knn::KNNBase&, const TransformMatrix&,
           const Registration::ExecutionOptions&) { return RegistrationResult{}; },
        1, false);
    EXPECT_EQ(velocity_pipeline.get_deskewed_point_cloud(), nullptr);
}

TEST(RegistrationPipelineTest, VelocityUpdateAlignerExposesMostRecentDeskewedPointCloud) {
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

    pipeline::VelocityUpdateAligner pipeline(aligner, 1, false);
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

TEST(RegistrationPipelineTest, RobustAlignerUsesDefaultScaleAsFixedAndInitialScale) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);

    RegistrationPipelineParams params;
    params.registration.robust.type = robust::RobustLossType::HUBER;
    params.registration.robust.default_scale = 8.0f;

    std::vector<float> fixed_scales;
    auto fixed_aligner = [&](const PointCloudShared&, const PointCloudShared&, const knn::KNNBase&,
                             const TransformMatrix&, const Registration::ExecutionOptions& options) {
        fixed_scales.push_back(options.robust_scale);
        return RegistrationResult{};
    };

    RegistrationPipeline fixed_pipeline(fixed_aligner, params);
    DummyKNN knn;
    fixed_pipeline.align(make_cloud(queue, 3), make_cloud(queue, 3), knn);

    ASSERT_EQ(fixed_scales.size(), 1U);
    EXPECT_FLOAT_EQ(fixed_scales.front(), 8.0f);

    params.robust.auto_scale = true;
    params.robust.init_scale = 6.0f;
    params.robust.min_scale = 2.0f;
    params.robust.rotation_init_scale = 9.0f;
    params.robust.rotation_min_scale = 3.0f;
    params.robust.auto_scaling_iter = 3;

    std::vector<float> annealed_scales;
    std::vector<float> annealed_rotation_scales;
    auto annealed_aligner = [&](const PointCloudShared&, const PointCloudShared&, const knn::KNNBase&,
                                const TransformMatrix&, const Registration::ExecutionOptions& options) {
        annealed_scales.push_back(options.robust_scale);
        annealed_rotation_scales.push_back(options.rotation_robust_scale);
        return RegistrationResult{};
    };

    RegistrationPipeline annealed_pipeline(annealed_aligner, params);
    annealed_pipeline.align(make_cloud(queue, 3), make_cloud(queue, 3), knn);

    ASSERT_EQ(annealed_scales.size(), 3U);
    EXPECT_FLOAT_EQ(annealed_scales[0], 6.0f);
    EXPECT_NEAR(annealed_scales[1], std::sqrt(12.0f), 1e-5f);
    EXPECT_NEAR(annealed_scales[2], 2.0f, 1e-5f);
    ASSERT_EQ(annealed_rotation_scales.size(), 3U);
    EXPECT_FLOAT_EQ(annealed_rotation_scales[0], 9.0f);
    EXPECT_NEAR(annealed_rotation_scales[1], std::sqrt(27.0f), 1e-5f);
    EXPECT_NEAR(annealed_rotation_scales[2], 3.0f, 1e-5f);
}

TEST(RegistrationPipelineTest, RegistrationComputeWeightsUseZeroOneForNoneLoss) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);

    RegistrationParams params;
    params.reg_type = RegType::POINT_TO_POINT;
    params.robust.type = robust::RobustLossType::NONE;
    params.max_iterations = 1;
    params.max_correspondence_distance = 1.5f;

    Registration registration(queue, params);
    const auto source = make_registration_source(queue);
    const auto target = make_registration_target(queue);
    CountingNearestKNN knn;
    knn.set_target(target);
    shared_vector<float> weights(*queue.ptr);

    registration.align(source, target, knn);
    const size_t calls_after_align = knn.call_count;
    registration.compute_icp_robust_weights(source, target, knn, TransformMatrix::Identity(),
                                            params.robust.default_scale, weights);
    EXPECT_EQ(weights.size(), source.size());
    EXPECT_FLOAT_EQ(weights[0], 1.0f);
    EXPECT_FLOAT_EQ(weights[1], 1.0f);
    EXPECT_FLOAT_EQ(weights[2], 0.0f);
    EXPECT_EQ(knn.call_count, calls_after_align + 1);
}

TEST(RegistrationPipelineTest, RegistrationComputeWeightsFollowsProvidedSource) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);

    RegistrationParams params;
    params.reg_type = RegType::POINT_TO_POINT;
    params.robust.type = robust::RobustLossType::NONE;
    params.max_iterations = 1;
    params.max_correspondence_distance = 1.5f;

    Registration registration(queue, params);
    const auto target = make_registration_target(queue);
    CountingNearestKNN knn;
    knn.set_target(target);
    shared_vector<float> first_weights(*queue.ptr);
    shared_vector<float> refreshed_weights(*queue.ptr);

    auto source = make_registration_source(queue);
    registration.align(source, target, knn);
    registration.compute_icp_robust_weights(source, target, knn, TransformMatrix::Identity(),
                                            params.robust.default_scale, first_weights);
    ASSERT_EQ(first_weights.size(), 3U);
    EXPECT_FLOAT_EQ(first_weights[2], 0.0f);

    PointCloudShared updated_source(queue);
    updated_source.points->resize(2);
    updated_source.points->at(0) = PointType(0.0f, 0.0f, 0.0f, 1.0f);
    updated_source.points->at(1) = PointType(1.0f, 0.0f, 0.0f, 1.0f);
    registration.align(updated_source, target, knn);

    const size_t calls_before_refresh = knn.call_count;
    registration.compute_icp_robust_weights(updated_source, target, knn, TransformMatrix::Identity(),
                                            params.robust.default_scale, refreshed_weights);
    EXPECT_EQ(knn.call_count, calls_before_refresh + 1);
    ASSERT_EQ(refreshed_weights.size(), 2U);
    EXPECT_FLOAT_EQ(refreshed_weights[0], 1.0f);
    EXPECT_FLOAT_EQ(refreshed_weights[1], 1.0f);
}

TEST(RegistrationPipelineTest, RegistrationComputeWeightsUsesProvidedRobustScale) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);

    RegistrationParams params;
    params.reg_type = RegType::POINT_TO_POINT;
    params.robust.type = robust::RobustLossType::HUBER;
    params.robust.default_scale = 10.0f;
    params.max_iterations = 1;
    params.max_correspondence_distance = 10.0f;

    Registration registration(queue, params);
    PointCloudShared source(queue);
    source.points->resize(1);
    source.points->at(0) = PointType(3.0f, 0.0f, 0.0f, 1.0f);
    PointCloudShared target(queue);
    target.points->resize(1);
    target.points->at(0) = PointType(0.0f, 0.0f, 0.0f, 1.0f);
    CountingNearestKNN knn;
    knn.set_target(target);

    shared_vector<float> weights(*queue.ptr);
    registration.compute_icp_robust_weights(source, target, knn, TransformMatrix::Identity(), 1.0f, weights);
    ASSERT_EQ(weights.size(), 1U);
    EXPECT_NEAR(weights[0], 1.0f / 3.0f, 1e-5f);

    registration.align(source, target, knn);
    registration.compute_icp_robust_weights(source, target, knn, TransformMatrix::Identity(), 2.0f, weights);
    ASSERT_EQ(weights.size(), 1U);
    EXPECT_NEAR(weights[0], 2.0f / 3.0f, 1e-5f);
}

TEST(RegistrationPipelineTest, PipelineLazyWeightsMatchDeskewedPointCloudSize) {
    sycl::device device(sycl_utils::device_selector::default_selector_v);
    sycl_utils::DeviceQueue queue(device);

    RegistrationPipelineParams params;
    params.registration.reg_type = RegType::POINT_TO_POINT;
    params.registration.robust.type = robust::RobustLossType::NONE;
    params.registration.max_iterations = 1;
    params.registration.max_correspondence_distance = 1.5f;
    params.velocity_update.enable = true;
    params.velocity_update.iter = 1;

    RegistrationPipeline pipeline(queue, params);
    const auto source = make_registration_source(queue);
    const auto target = make_registration_target(queue);
    CountingNearestKNN knn;
    knn.set_target(target);

    Registration::ExecutionOptions options;
    options.dt = 1.0f;
    options.prev_pose = TransformMatrix::Identity();

    pipeline.align(source, target, knn, TransformMatrix::Identity(), options);

    const auto* deskewed = pipeline.get_deskewed_point_cloud();
    ASSERT_NE(deskewed, nullptr);
    shared_vector<float> weights(*queue.ptr);
    pipeline.compute_icp_robust_weights(target, knn, TransformMatrix::Identity(),
                                        params.registration.robust.default_scale, weights);
    EXPECT_EQ(weights.size(), deskewed->size());
}

}  // namespace
}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
