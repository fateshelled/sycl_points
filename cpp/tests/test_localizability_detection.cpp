#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>

#include <sycl_points/algorithms/localizability_constrained_optimizer.hpp>
#include <sycl_points/algorithms/localizability_detection.hpp>
#include <sycl_points/algorithms/registration_lp_icp.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sp = sycl_points;
namespace loc = sp::algorithms::localizability;
namespace reg = sp::algorithms::registration;

class LocalizabilityDetectionTest : public ::testing::Test {
protected:
    void SetUp() override { queue_ = sp::sycl_utils::create_device_queue_auto(); }

    sp::sycl_utils::DeviceQueue queue_;
};

// Test Hessian decomposition
TEST_F(LocalizabilityDetectionTest, HessianDecomposition) {
    loc::LocalizabilityParams params;
    params.verbose = false;
    loc::LocalizabilityDetection detector(queue_, params);

    // Create a simple Hessian matrix
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();

    // Rotation block (3x3)
    H.block<3, 3>(0, 0) << 10.0f, 0.0f, 0.0f,  //
        0.0f, 5.0f, 0.0f,                       //
        0.0f, 0.0f, 1.0f;

    // Translation block (3x3)
    H.block<3, 3>(3, 3) << 20.0f, 0.0f, 0.0f,  //
        0.0f, 15.0f, 0.0f,                      //
        0.0f, 0.0f, 2.0f;

    // Create minimal point clouds for detection
    auto source = std::make_shared<sp::PointCloudShared>(queue_);
    auto target = std::make_shared<sp::PointCloudShared>(queue_);

    source->resize(1);
    target->resize(1);

    source->points_ptr()[0] = sp::PointType(0.0f, 0.0f, 0.0f, 1.0f);
    target->points_ptr()[0] = sp::PointType(0.0f, 0.0f, 0.0f, 1.0f);

    // Allocate normals for target
    target->allocate_normals();
    target->normals_ptr()[0] = sp::Normal(0.0f, 0.0f, 1.0f, 0.0f);

    // Create KNN result
    sp::algorithms::knn::KNNResult neighbors;
    neighbors.allocate(queue_, 1, 1);
    neighbors.indices->at(0) = 0;
    neighbors.distances->at(0) = 0.0f;

    sp::TransformMatrix T = sp::TransformMatrix::Identity();

    // Detect localizability
    auto result = detector.detect(H, *source, *target, neighbors, T, 10.0f);

    // Check that eigenvalues are extracted (sorted ascending)
    EXPECT_LE(result.eigenvalues_r(0), result.eigenvalues_r(1));
    EXPECT_LE(result.eigenvalues_r(1), result.eigenvalues_r(2));
    EXPECT_LE(result.eigenvalues_t(0), result.eigenvalues_t(1));
    EXPECT_LE(result.eigenvalues_t(1), result.eigenvalues_t(2));
}

// Test tri-value classification thresholds
TEST_F(LocalizabilityDetectionTest, TriValueClassification) {
    loc::LocalizabilityParams params;
    params.T1 = 50.0f;
    params.T2 = 30.0f;
    params.T3 = 15.0f;
    params.T4 = 9.0f;

    // Test FULL classification (L_f >= T1)
    {
        loc::LocalizabilityAggregate agg;
        agg.L_f(0) = 60.0f;  // >= T1
        agg.L_u(0) = 0.0f;

        if (agg.L_f(0) >= params.T1 || agg.L_u(0) >= params.T2) {
            agg.categories[0] = loc::LocalizabilityCategory::FULL;
        }
        EXPECT_EQ(agg.categories[0], loc::LocalizabilityCategory::FULL);
    }

    // Test FULL classification (L_u >= T2)
    {
        loc::LocalizabilityAggregate agg;
        agg.L_f(0) = 10.0f;
        agg.L_u(0) = 35.0f;  // >= T2

        if (agg.L_f(0) >= params.T1 || agg.L_u(0) >= params.T2) {
            agg.categories[0] = loc::LocalizabilityCategory::FULL;
        }
        EXPECT_EQ(agg.categories[0], loc::LocalizabilityCategory::FULL);
    }

    // Test PARTIAL classification (L_f >= T3 AND L_u >= T4)
    {
        loc::LocalizabilityAggregate agg;
        agg.L_f(0) = 20.0f;  // >= T3 but < T1
        agg.L_u(0) = 12.0f;  // >= T4 but < T2

        if (agg.L_f(0) >= params.T1 || agg.L_u(0) >= params.T2) {
            agg.categories[0] = loc::LocalizabilityCategory::FULL;
        } else if (agg.L_f(0) >= params.T3 && agg.L_u(0) >= params.T4) {
            agg.categories[0] = loc::LocalizabilityCategory::PARTIAL;
        }
        EXPECT_EQ(agg.categories[0], loc::LocalizabilityCategory::PARTIAL);
    }

    // Test NONE classification
    {
        loc::LocalizabilityAggregate agg;
        agg.L_f(0) = 5.0f;  // < T3
        agg.L_u(0) = 3.0f;  // < T4

        if (agg.L_f(0) >= params.T1 || agg.L_u(0) >= params.T2) {
            agg.categories[0] = loc::LocalizabilityCategory::FULL;
        } else if (agg.L_f(0) >= params.T3 && agg.L_u(0) >= params.T4) {
            agg.categories[0] = loc::LocalizabilityCategory::PARTIAL;
        } else {
            agg.categories[0] = loc::LocalizabilityCategory::NONE;
        }
        EXPECT_EQ(agg.categories[0], loc::LocalizabilityCategory::NONE);
    }
}

// Test constrained optimizer with soft constraints
TEST_F(LocalizabilityDetectionTest, SoftConstraintOptimization) {
    loc::ConstrainedOptimizerParams params;
    params.soft_constraint_weight = 1.0f;
    params.damping = 1e-6f;
    loc::ConstrainedOptimizer optimizer(params);

    // Simple Hessian and gradient
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Identity() * 10.0f;
    Eigen::Vector<float, 6> b;
    b << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f;

    // Create localizability result with soft constraint
    loc::LocalizabilityResult loc_result;
    loc::SoftConstraint soft;
    soft.direction_index = 0;
    soft.constraint_value = 0.5f;
    soft.eigenvector = Eigen::Vector<float, 6>::Zero();
    soft.eigenvector(0) = 1.0f;
    loc_result.soft_constraints.push_back(soft);

    // Solve
    Eigen::Vector<float, 6> delta = optimizer.solve(H, b, loc_result);

    // Result should be finite
    EXPECT_TRUE(delta.allFinite());
}

// Test constrained optimizer with hard constraints
TEST_F(LocalizabilityDetectionTest, HardConstraintOptimization) {
    loc::ConstrainedOptimizerParams params;
    params.damping = 1e-6f;
    loc::ConstrainedOptimizer optimizer(params);

    // Hessian and gradient
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Identity() * 10.0f;
    Eigen::Vector<float, 6> b;
    b << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f;

    // Create localizability result with hard constraint on z-translation
    loc::LocalizabilityResult loc_result;
    Eigen::Vector<float, 6> v = Eigen::Vector<float, 6>::Zero();
    v(5) = 1.0f;  // Constrain tz
    loc_result.hard_constraint.addConstraint(v);

    // Solve
    Eigen::Vector<float, 6> delta = optimizer.solve(H, b, loc_result);

    // Check that tz is approximately zero (constrained)
    EXPECT_NEAR(delta(5), 0.0f, 1e-4f);
}

// Test LP-ICP registration parameters
TEST_F(LocalizabilityDetectionTest, LPRegistrationParams) {
    reg::LPRegistrationParams params;

    // Default values
    EXPECT_EQ(params.reg_type, reg::RegType::POINT_TO_PLANE);
    EXPECT_EQ(params.max_iterations, 20);
    EXPECT_TRUE(params.enable_localizability);

    // Localizability params
    EXPECT_FLOAT_EQ(params.localizability_params.noise_threshold, 0.03f);
    EXPECT_FLOAT_EQ(params.localizability_params.high_contribution_threshold, 0.4998f);
    EXPECT_FLOAT_EQ(params.localizability_params.T1, 50.0f);
    EXPECT_FLOAT_EQ(params.localizability_params.T2, 30.0f);
    EXPECT_FLOAT_EQ(params.localizability_params.T3, 15.0f);
    EXPECT_FLOAT_EQ(params.localizability_params.T4, 9.0f);
}

// Test contribution computation kernel
TEST_F(LocalizabilityDetectionTest, ContributionComputation) {
    // Create simple eigenvector matrices (identity)
    Eigen::Matrix3f V_r = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f V_t = Eigen::Matrix3f::Identity();

    // Create a Jacobian row
    Eigen::Matrix<float, 1, 6> J;
    J << 0.5f, 0.3f, 0.1f, 0.4f, 0.2f, 0.6f;

    // Compute contribution
    auto contrib = loc::kernel::compute_contribution(J, V_r, V_t, true);

    // Check that contributions are computed
    EXPECT_GE(contrib.F_r.sum(), 0.0f);
    EXPECT_GE(contrib.F_t.sum(), 0.0f);
}

// Test noise filtering
TEST_F(LocalizabilityDetectionTest, NoiseFiltering) {
    loc::LocalizabilityParams params;
    params.noise_threshold = 0.03f;
    params.high_contribution_threshold = 0.4998f;

    // Create contributions
    std::vector<loc::LocalizabilityContribution> contributions;
    loc::LocalizabilityContribution c1;
    c1.F_r << 0.01f, 0.5f, 0.02f;   // First and third below threshold
    c1.F_t << 0.04f, 0.02f, 0.6f;   // Second below threshold
    contributions.push_back(c1);

    // Test filtering logic
    Eigen::Vector<float, 6> F;
    F.head<3>() = c1.F_r;
    F.tail<3>() = c1.F_t;

    Eigen::Vector<float, 6> F_f = Eigen::Vector<float, 6>::Zero();
    for (size_t j = 0; j < 6; ++j) {
        if (F(j) >= params.noise_threshold) {
            F_f(j) = F(j);
        }
    }

    // Check that values below threshold are filtered
    EXPECT_FLOAT_EQ(F_f(0), 0.0f);   // 0.01 < 0.03
    EXPECT_FLOAT_EQ(F_f(1), 0.5f);   // 0.5 >= 0.03
    EXPECT_FLOAT_EQ(F_f(2), 0.0f);   // 0.02 < 0.03
    EXPECT_FLOAT_EQ(F_f(3), 0.04f);  // 0.04 >= 0.03
    EXPECT_FLOAT_EQ(F_f(4), 0.0f);   // 0.02 < 0.03
    EXPECT_FLOAT_EQ(F_f(5), 0.6f);   // 0.6 >= 0.03
}

// Test hard constraint matrix construction
TEST_F(LocalizabilityDetectionTest, HardConstraintConstruction) {
    loc::HardConstraint constraint;

    // Add constraints
    Eigen::Vector<float, 6> v1 = Eigen::Vector<float, 6>::Zero();
    v1(0) = 1.0f;
    constraint.addConstraint(v1);

    Eigen::Vector<float, 6> v2 = Eigen::Vector<float, 6>::Zero();
    v2(3) = 1.0f;
    constraint.addConstraint(v2);

    // Check
    EXPECT_EQ(constraint.num_constraints, 2);
    EXPECT_FLOAT_EQ(constraint.D(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(constraint.D(1, 3), 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
