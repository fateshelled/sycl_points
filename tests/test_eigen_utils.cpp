#include <gtest/gtest.h>

#include <cmath>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {
namespace eigen_utils {
namespace test {

// 浮動小数点の比較のための許容誤差
constexpr float EPSILON = 1e-5f;

// マトリックス比較用のヘルパー関数 - C++17のテンプレート型推論を活用
template <typename Derived1, typename Derived2>
void expectMatrixNear(const Eigen::MatrixBase<Derived1>& expected, const Eigen::MatrixBase<Derived2>& actual) {
    // 行列のサイズが一致することを確認
    ASSERT_EQ(expected.rows(), actual.rows()) << "Matrix row count mismatch";
    ASSERT_EQ(expected.cols(), actual.cols()) << "Matrix column count mismatch";

    for (int i = 0; i < expected.rows(); ++i) {
        for (int j = 0; j < expected.cols(); ++j) {
            EXPECT_NEAR(expected(i, j), actual(i, j), EPSILON) << "Matrices differ at (" << i << "," << j << ")";
        }
    }
}

// ベクトル比較用のヘルパー関数 - C++17のテンプレート型推論を活用
template <typename Derived1, typename Derived2>
void expectVectorNear(const Eigen::MatrixBase<Derived1>& expected, const Eigen::MatrixBase<Derived2>& actual) {
    // ベクトルのサイズが一致することを確認
    ASSERT_EQ(expected.size(), actual.size()) << "Vector size mismatch";

    for (int i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(expected(i), actual(i), EPSILON) << "Vectors differ at index " << i;
    }
}

class EigenUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // テスト用の一般的なマトリックスとベクトルを初期化
        A3x3 << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f;

        B3x3 << 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f;

        A4x4 << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f;

        B4x4 << 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f;

        vec3 << 1.0f, 2.0f, 3.0f;
        vec4 << 1.0f, 2.0f, 3.0f, 4.0f;
        vec4_2 << 5.0f, 6.0f, 7.0f, 8.0f;

        A6x4 << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
            17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f;

        B4x6 << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
            17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f;
    }

    Eigen::Matrix3f A3x3;
    Eigen::Matrix3f B3x3;
    Eigen::Matrix4f A4x4;
    Eigen::Matrix4f B4x4;
    Eigen::Vector3f vec3;
    Eigen::Vector4f vec4;
    Eigen::Vector4f vec4_2;
    Eigen::Matrix<float, 6, 4> A6x4;
    Eigen::Matrix<float, 4, 6> B4x6;
};

TEST_F(EigenUtilsTest, add) {
    // 3x3
    Eigen::Matrix3f expected3x3;
    expected3x3 << 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f;

    const Eigen::Matrix3f result3x3 = add<3, 3>(A3x3, B3x3);
    expectMatrixNear(expected3x3, result3x3);

    // 4x4
    Eigen::Matrix4f expected4x4;
    expected4x4 << 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f, 17.0f,
        17.0f, 17.0f;

    const Eigen::Matrix4f result4x4 = add<4, 4>(A4x4, B4x4);
    expectMatrixNear(expected4x4, result4x4);

    // vector4
    Eigen::Vector4f expected4;
    expected4 << 6.0f, 8.0f, 10.0f, 12.0f;

    const Eigen::Vector4f result4 = add<4, 1>(vec4, vec4_2);
    expectVectorNear(expected4, result4);
}

TEST_F(EigenUtilsTest, add_inplace) {
    // 3x3
    Eigen::Matrix3f expected3x3;
    expected3x3 << 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f;

    Eigen::Matrix3f result3x3 = A3x3;
    add_inplace<3, 3>(result3x3, B3x3);
    expectMatrixNear(expected3x3, result3x3);
}

TEST_F(EigenUtilsTest, subtract) {
    Eigen::Matrix3f expected;
    expected << -8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 2.0f, 4.0f, 6.0f, 8.0f;

    const Eigen::Matrix3f result = subtract<3, 3>(A3x3, B3x3);
    expectMatrixNear(expected, result);
}

TEST_F(EigenUtilsTest, multiply) {
    // 3x3 * 3x3
    const Eigen::Matrix3f C3x3 = A3x3 * B3x3;
    const Eigen::Matrix3f result3x3 = multiply<3, 3, 3>(A3x3, B3x3);
    expectMatrixNear(C3x3, result3x3);

    // 4x4 * 4x4
    const Eigen::Matrix4f C4x4 = A4x4 * B4x4;
    const Eigen::Matrix4f result4x4 = multiply<4, 4, 4>(A4x4, B4x4);
    expectMatrixNear(C4x4, result4x4);

    // 6x4 * 4x6
    const Eigen::Matrix<float, 6, 6> C6x6 = A6x4 * B4x6;
    const Eigen::Matrix<float, 6, 6> result6x6 = multiply<6, 4, 6>(A6x4, B4x6);
    expectMatrixNear(C6x6, result6x6);

    // 6x4 * 4x4
    const Eigen::Matrix<float, 6, 4> C6x4 = A6x4 * B4x4.block<4, 4>(0, 0);
    const Eigen::Matrix<float, 6, 4> result6x4 = multiply<6, 4, 4>(A6x4, B4x4);
    expectMatrixNear(C6x4, result6x4);
}

TEST_F(EigenUtilsTest, multiply_inplace) {
    const float scalar = 2.5f;
    const Eigen::Matrix3f expected = A3x3 * scalar;
    Eigen::Matrix3f result = A3x3;
    multiply_inplace<3, 3>(result, scalar);
    expectMatrixNear(expected, result);
}

TEST_F(EigenUtilsTest, MatrixVectorMultiply) {
    // 4x4 * 4x1
    const Eigen::Vector4f expected4 = A4x4 * vec4;
    const Eigen::Vector4f result4 = multiply<4, 4>(A4x4, vec4);
    expectVectorNear(expected4, result4);

    // 3x3 * 3x1
    const Eigen::Vector3f expected3 = A3x3 * vec3;
    const Eigen::Vector3f result3 = multiply<3, 3>(A3x3, vec3);
    expectVectorNear(expected3, result3);
}

TEST_F(EigenUtilsTest, MatrixScalarMultiply) {
    const float scalar = 2.5f;
    const Eigen::Matrix3f expected = A3x3 * scalar;
    const Eigen::Matrix3f result = multiply<3, 3>(A3x3, scalar);
    expectMatrixNear(expected, result);
}

TEST_F(EigenUtilsTest, VectorScalarMultiply) {
    const float scalar = 3.0f;
    const Eigen::Vector3f expected = vec3 * scalar;
    const Eigen::Vector3f result = multiply<3>(vec3, scalar);
    expectVectorNear(expected, result);
}

TEST_F(EigenUtilsTest, transpose) {
    // 3x3
    const Eigen::Matrix3f expected3 = A3x3.transpose();
    const Eigen::Matrix3f result3 = transpose<3, 3>(A3x3);
    expectMatrixNear(expected3, result3);

    // 4x4
    const Eigen::Matrix4f expected4 = A4x4.transpose();
    const Eigen::Matrix4f result4 = transpose<4, 4>(A4x4);
    expectMatrixNear(expected4, result4);

    // 4x6
    const Eigen::Matrix<float, 6, 4> expected6x4 = B4x6.transpose();
    const Eigen::Matrix<float, 6, 4> result6x4 = transpose<4, 6>(B4x6);
    expectMatrixNear(expected6x4, result6x4);
}

TEST_F(EigenUtilsTest, dot) {
    const float expected3 = vec3.dot(vec3);
    const float result3 = dot<3>(vec3, vec3);
    EXPECT_NEAR(expected3, result3, EPSILON);

    const float expected4 = vec4.dot(vec4);
    const float result4 = dot<4>(vec4, vec4);
    EXPECT_NEAR(expected4, result4, EPSILON);
}

TEST_F(EigenUtilsTest, cross) {
    const Eigen::Vector3f v1(1.0f, 2.0f, 3.0f);
    const Eigen::Vector3f v2(4.0f, 5.0f, 6.0f);

    const Eigen::Vector3f expected = v1.cross(v2);
    const Eigen::Vector3f result = cross(v1, v2);
    expectVectorNear(expected, result);
}

TEST_F(EigenUtilsTest, outer) {
    const Eigen::Vector4f v1(1.0f, 2.0f, 3.0f, 4.0f);
    const Eigen::Vector4f v2(5.0f, 6.0f, 7.0f, 8.0f);

    const Eigen::Matrix4f expected = v1 * v2.transpose();
    const Eigen::Matrix4f result = outer<4>(v1, v2);
    expectMatrixNear(expected, result);
}

TEST_F(EigenUtilsTest, block3x3) {
    const Eigen::Matrix3f expected = A4x4.block<3, 3>(0, 0);
    const Eigen::Matrix3f result = block3x3(A4x4);
    expectMatrixNear(expected, result);
}

TEST_F(EigenUtilsTest, inverse) {
    // regular matrix
    Eigen::Matrix3f invertible;
    invertible << 4.0f, 7.0f, 2.0f, 9.0f, 6.0f, 1.0f, 8.0f, 5.0f, 3.0f;

    const Eigen::Matrix3f expectedInv = invertible.inverse();
    const Eigen::Matrix3f resultInv = inverse(invertible);
    expectMatrixNear(expectedInv, resultInv);

    // Irregular matrix (det==0)
    Eigen::Matrix3f singular;
    singular << 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f, 7.0f, 8.0f, 9.0f;

    const Eigen::Matrix3f zeroMatrix = Eigen::Matrix3f::Zero();
    const Eigen::Matrix3f resultSingular = inverse(singular);
    expectMatrixNear(zeroMatrix, resultSingular);
}

TEST_F(EigenUtilsTest, ensure_symmetric) {
    // Create a non-symmetric matrix
    Eigen::Matrix3f non_symmetric;
    non_symmetric << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f;

    // Expected symmetric result (average of corresponding elements)
    Eigen::Matrix3f expected;
    expected << 1.0f, 3.0f, 5.0f, 3.0f, 5.0f, 7.0f, 5.0f, 7.0f, 9.0f;

    const Eigen::Matrix3f result = ensure_symmetric<3>(non_symmetric);
    expectMatrixNear(expected, result);

    // Verify the result is actually symmetric
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(result(i, j), result(j, i), EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, frobenius_norm) {
    // Matrix Frobenius norm
    const float expected_norm_A3x3 = A3x3.norm();
    const float result_norm_A3x3 = frobenius_norm<3, 3>(A3x3);
    EXPECT_NEAR(expected_norm_A3x3, result_norm_A3x3, EPSILON);

    // Vector Frobenius norm (L2 norm)
    const float expected_norm_vec3 = vec3.norm();
    const float result_norm_vec3 = frobenius_norm<3>(vec3);
    EXPECT_NEAR(expected_norm_vec3, result_norm_vec3, EPSILON);
}

TEST_F(EigenUtilsTest, determinant) {
    Eigen::Matrix3f mat;
    mat << 4.0f, 3.0f, 2.0f, 1.0f, 5.0f, 9.0f, 7.0f, 6.0f, 8.0f;

    const float expected_det = mat.determinant();
    const float result_det = determinant(mat);
    EXPECT_NEAR(expected_det, result_det, EPSILON);
}

TEST_F(EigenUtilsTest, as_diagonal) {
    const Eigen::Vector3f diag_elements(2.0f, 4.0f, 6.0f);

    Eigen::Matrix3f expected = Eigen::Matrix3f::Zero();
    expected(0, 0) = 2.0f;
    expected(1, 1) = 4.0f;
    expected(2, 2) = 6.0f;

    const Eigen::Matrix3f result = as_diagonal<3>(diag_elements);
    expectMatrixNear(expected, result);
}

TEST_F(EigenUtilsTest, symmetric_eigen_decomposition_3x3) {
    // Create a symmetric matrix
    Eigen::Matrix3f symmetric;
    symmetric << 2.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 2.0f;

    Eigen::Vector3f computed_eigenvalues;
    Eigen::Matrix3f computed_eigenvectors;

    // Use our implementation to compute eigenvalues and eigenvectors
    symmetric_eigen_decomposition_3x3(symmetric, computed_eigenvalues, computed_eigenvectors);

    // Use Eigen's implementation as reference
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(symmetric);
    Eigen::Vector3f expected_eigenvalues = eigen_solver.eigenvalues();
    Eigen::Matrix3f expected_eigenvectors = eigen_solver.eigenvectors();

    // Sort eigenvalues for comparison (our implementation should already be sorted)
    std::sort(expected_eigenvalues.data(), expected_eigenvalues.data() + 3);

    // Check eigenvalues
    expectVectorNear(expected_eigenvalues, computed_eigenvalues);

    // Check that eigenvectors are valid
    for (int i = 0; i < 3; ++i) {
        // Apply matrix to eigenvector
        const Eigen::Vector3f Av = symmetric * computed_eigenvectors.col(i);
        // Should be close to lambda * v
        const Eigen::Vector3f lambda_v = computed_eigenvalues(i) * computed_eigenvectors.col(i);
        expectVectorNear(Av, lambda_v);
    }
}

TEST_F(EigenUtilsTest, solve_system_6x6) {
    // Create a 6x6 SPD matrix
    Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Random();
    A = A.transpose() * A + Eigen::Matrix<float, 6, 6>::Identity();  // Ensure positive definiteness

    const Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Random();

    // Use Eigen's solver as reference
    const Eigen::Matrix<float, 6, 1> expected_x = A.ldlt().solve(b);

    // Use our implementation
    const Eigen::Matrix<float, 6, 1> computed_x = solve_system_6x6(A, b);

    // Compare results
    expectVectorNear(expected_x, computed_x);

    // Also verify solution by checking A*x ≈ b
    const Eigen::Matrix<float, 6, 1> computed_b = A * computed_x;
    expectVectorNear(b, computed_b);
}

// trace test
TEST_F(EigenUtilsTest, trace) {
    const float expected_trace = A3x3(0, 0) + A3x3(1, 1) + A3x3(2, 2);
    const float computed_trace = trace<3>(A3x3);
    EXPECT_NEAR(expected_trace, computed_trace, EPSILON);
}

// SYCL vector conversion tests
TEST_F(EigenUtilsTest, to_sycl_vec_from_sycl_vec) {
    // Test vector conversion
    const Eigen::Vector4f input_vec(1.0f, 2.0f, 3.0f, 4.0f);
    const sycl::float4 sycl_vec = to_sycl_vec(input_vec);
    const Eigen::Vector4f output_vec = from_sycl_vec(sycl_vec);
    expectVectorNear(input_vec, output_vec);

    // Test Matrix4f conversion
    const Eigen::Matrix4f input_mat = A4x4;
    const auto sycl_mat = to_sycl_vec(input_mat);
    const Eigen::Matrix4f output_mat = from_sycl_vec(sycl_mat);
    expectMatrixNear(input_mat, output_mat);

    // Test Matrix<float, 6, 6> conversion
    const Eigen::Matrix<float, 6, 6> input_mat6 = Eigen::Matrix<float, 6, 6>::Random();
    const auto sycl_mat6 = to_sycl_vec(input_mat6);
    const Eigen::Matrix<float, 6, 6> output_mat6 = from_sycl_vec(sycl_mat6);
    expectMatrixNear(input_mat6, output_mat6);

    // Test Vector<float, 6> conversion
    const Eigen::Vector<float, 6> input_vec6 = Eigen::Vector<float, 6>::Random();
    const auto sycl_vec6 = to_sycl_vec(input_vec6);
    const Eigen::Vector<float, 6> output_vec6 = from_sycl_vec(sycl_vec6);
    expectVectorNear(input_vec6, output_vec6);
}

TEST_F(EigenUtilsTest, skew) {
    const Eigen::Vector3f v(1.0f, 2.0f, 3.0f);

    Eigen::Matrix3f expected;
    expected << 0.0f, -3.0f, 2.0f, 3.0f, 0.0f, -1.0f, -2.0f, 1.0f, 0.0f;

    const Eigen::Matrix3f result3 = lie::skew(v);
    expectMatrixNear(expected, result3);

    const Eigen::Vector4f v4(1.0f, 2.0f, 3.0f, 4.0f);
    const Eigen::Matrix3f result4 = lie::skew(v4);
    expectMatrixNear(expected, result4);
}

TEST_F(EigenUtilsTest, SO3Exp) {
    // small rotation
    const Eigen::Vector3f small_rotation(0.1f, 0.2f, 0.3f);
    const Eigen::Quaternionf quat1 = lie::so3_exp(small_rotation);

    const float angle = small_rotation.norm();
    const Eigen::Vector3f axis = small_rotation / angle;
    const Eigen::AngleAxisf aa(angle, axis);
    const Eigen::Quaternionf expected = Eigen::Quaternionf(aa);

    EXPECT_NEAR(expected.w(), quat1.w(), EPSILON);
    EXPECT_NEAR(expected.x(), quat1.x(), EPSILON);
    EXPECT_NEAR(expected.y(), quat1.y(), EPSILON);
    EXPECT_NEAR(expected.z(), quat1.z(), EPSILON);

    // nearly zero
    const Eigen::Vector3f zero_rotation(1e-10f, 2e-10f, 3e-10f);
    const Eigen::Quaternionf quat2 = lie::so3_exp(zero_rotation);

    EXPECT_NEAR(1.0f, quat2.w(), EPSILON);
    EXPECT_NEAR(0.0f, quat2.x(), EPSILON);
    EXPECT_NEAR(0.0f, quat2.y(), EPSILON);
    EXPECT_NEAR(0.0f, quat2.z(), EPSILON);
}

TEST_F(EigenUtilsTest, SE3Exp) {
    Eigen::Matrix<float, 6, 1> twist;
    twist << 0.1f, 0.2f, 0.3f, 1.0f, 2.0f, 3.0f;  // [rx, ry, rz, tx, ty, tz]

    const Eigen::Isometry3f transform = lie::se3_exp(twist);

    const Eigen::Vector3f rotation_part = twist.head<3>();
    const Eigen::Quaternionf quat = lie::so3_exp(rotation_part);
    const Eigen::Matrix3f expected_rotation = quat.toRotationMatrix();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(expected_rotation(i, j), transform.linear()(i, j), EPSILON)
                << "Rotation matrices differ at (" << i << "," << j << ")";
        }
    }

    // transform * point1 = point2 -> transform.inverse() * point2 = point1
    const Eigen::Vector3f point1(5.0f, 6.0f, 7.0f);
    const Eigen::Vector3f point2 = transform * point1;
    const Eigen::Vector3f point1_back = transform.inverse() * point2;

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(point1(i), point1_back(i), EPSILON) << "Points differ after transform and inverse at index " << i;
    }

    // nearly zero rotation
    Eigen::Matrix<float, 6, 1> small_twist;
    small_twist << 1e-10f, 2e-10f, 3e-10f, 1.0f, 2.0f, 3.0f;

    const Eigen::Isometry3f small_transform = lie::se3_exp(small_twist);

    const Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(identity(i, j), small_transform.linear()(i, j), EPSILON)
                << "Small rotation matrix differs from identity at (" << i << "," << j << ")";
        }
    }

    const Eigen::Vector3f trans = small_transform.translation();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(small_twist(i + 3), trans(i), EPSILON) << "Small translation differs at index " << i;
    }
}

}  // namespace test
}  // namespace eigen_utils
}  // namespace sycl_points

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
