#include <gtest/gtest.h>

#include <cmath>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {
namespace eigen_utils {
namespace test {

constexpr float EPSILON = 1e-5f;
constexpr size_t TEST_ITERAIONS = 1000;

::testing::AssertionResult AssertMatrixNear(const char* expr1, const char* expr2, const char* expr3,
                                            const Eigen::MatrixXf& m1, const Eigen::MatrixXf& m2, double threshold) {
    if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
        return ::testing::AssertionFailure() << "Matrix size mismatch:\n"
                                             << expr1 << " is " << m1.rows() << "x" << m1.cols() << ",\n"
                                             << expr2 << " is " << m2.rows() << "x" << m2.cols();
    }

    Eigen::MatrixXf diff = (m1 - m2).cwiseAbs();
    for (int i = 0; i < diff.rows(); ++i) {
        for (int j = 0; j < diff.cols(); ++j) {
            if (diff(i, j) > threshold) {
                return ::testing::AssertionFailure()
                       << "Matrix differ at (" << i << "," << j << "): threshold = " << threshold << "\n"
                       << expr1 << ":\n"
                       << m1 << "\n"
                       << expr2 << ":\n"
                       << m2 << std::endl;
            }
        }
    }

    return ::testing::AssertionSuccess();
}

::testing::AssertionResult AssertVectorNear(const char* expr1, const char* expr2, const char* expr3,
                                            const Eigen::VectorXf& m1, const Eigen::VectorXf& m2, double threshold) {
    if (m1.size() != m2.size()) {
        return ::testing::AssertionFailure() << "Vector size mismatch:\n"
                                             << expr1 << " is " << m1.size() << ",\n"
                                             << expr2 << " is " << m2.size();
    }

    Eigen::VectorXf diff = m1 - m2;
    for (int i = 0; i < diff.size(); ++i) {
        if (diff(i) > threshold) {
            return ::testing::AssertionFailure() << "Vector differ at (" << i << "): threshold = " << threshold << "\n"
                                                 << expr1 << ":\n"
                                                 << m1.transpose() << "\n"
                                                 << expr2 << ":\n"
                                                 << m2.transpose() << std::endl;
        }
    }

    return ::testing::AssertionSuccess();
}

#define EXPECT_MATRIX_NEAR(m1, m2, threshold) EXPECT_PRED_FORMAT3(AssertMatrixNear, m1, m2, threshold)
#define ASSERT_MATRIX_NEAR(m1, m2, threshold) ASSERT_PRED_FORMAT3(AssertMatrixNear, m1, m2, threshold)

#define EXPECT_VECTOR_NEAR(m1, m2, threshold) EXPECT_PRED_FORMAT3(AssertVectorNear, m1, m2, threshold)
#define ASSERT_VECTOR_NEAR(m1, m2, threshold) ASSERT_PRED_FORMAT3(AssertVectorNear, m1, m2, threshold)

class EigenUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed the random number generator
        srand(1234);

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
    for (size_t iter = 0; iter<TEST_ITERAIONS; ++iter){
        // 3x3
        {
            constexpr size_t M = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, N> A = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> B = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> expected = A + B;
            const Eigen::Matrix<float, M, N> result = add<M, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, EPSILON);
        }
        // 4x4
        {
            constexpr size_t M = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, N> A = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> B = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> expected = A + B;
            const Eigen::Matrix<float, M, N> result = add<M, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, EPSILON);
        }
        // vector3
        {
            constexpr size_t M = 3;
            const Eigen::Vector<float, M> A = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> B = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> expected = A + B;
            const Eigen::Vector<float, M> result = add<M, 1>(A, B);

            EXPECT_VECTOR_NEAR(expected, result, EPSILON);
        }
        // vector4
        {
            constexpr size_t M = 4;
            const Eigen::Vector<float, M> A = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> B = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> expected = A + B;
            const Eigen::Vector<float, M> result = add<M, 1>(A, B);

            EXPECT_VECTOR_NEAR(expected, result, EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, add_inplace) {
    for (size_t iter = 0; iter<TEST_ITERAIONS; ++iter){
        // 3x3
        {
            constexpr size_t M = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, N> A = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> B = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> expected = A + B;
            Eigen::Matrix<float, M, N> actual = A;
            add_inplace<M, N>(actual, B);

            EXPECT_MATRIX_NEAR(expected, actual, EPSILON);
        }
        // 4x4
        {
            constexpr size_t M = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, N> A = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> B = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> expected = A + B;
            Eigen::Matrix<float, M, N> actual = A;
            add_inplace<M, N>(actual, B);

            EXPECT_MATRIX_NEAR(expected, actual, EPSILON);
        }
        // vector3
        {
            constexpr size_t M = 3;
            constexpr size_t N = 1;
            const Eigen::Vector<float, M> A = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> B = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> expected = A + B;
            Eigen::Vector<float, M> actual = A;
            add_inplace<M, 1>(actual, B);

            EXPECT_VECTOR_NEAR(expected, actual, EPSILON);
        }
        // vector4
        {
            constexpr size_t M = 4;
            const Eigen::Vector<float, M> A = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> B = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> expected = A + B;
            Eigen::Vector<float, M> actual = A;
            add_inplace<M, 1>(actual, B);

            EXPECT_VECTOR_NEAR(expected, actual, EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, subtract) {
    for (size_t iter = 0; iter<TEST_ITERAIONS; ++iter){
        // 3x3
        {
            constexpr size_t M = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, N> A = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> B = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> expected = A - B;
            const Eigen::Matrix<float, M, N> result = subtract<M, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, EPSILON);
        }
        // 4x4
        {
            constexpr size_t M = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, N> A = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> B = Eigen::Matrix<float, M, N>::Random();
            const Eigen::Matrix<float, M, N> expected = A - B;
            const Eigen::Matrix<float, M, N> result = subtract<M, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, EPSILON);
        }
        // vector3
        {
            constexpr size_t M = 3;
            const Eigen::Vector<float, M> A = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> B = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> expected = A - B;
            const Eigen::Vector<float, M> result = subtract<M, 1>(A, B);

            EXPECT_VECTOR_NEAR(expected, result, EPSILON);
        }
        // vector4
        {
            constexpr size_t M = 4;
            const Eigen::Vector<float, M> A = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> B = Eigen::Vector<float, M>::Random();
            const Eigen::Vector<float, M> expected = A - B;
            const Eigen::Vector<float, M> result = subtract<M, 1>(A, B);

            EXPECT_VECTOR_NEAR(expected, result, EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, multiply) {
    for (size_t iter = 0; iter<TEST_ITERAIONS; ++iter){
        // 3x3 * 3x3
        {
            constexpr size_t M = 3;
            constexpr size_t K = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, K> A = Eigen::Matrix<float, M, K>::Random();
            const Eigen::Matrix<float, K, N> B = Eigen::Matrix<float, K, N>::Random();
            const Eigen::Matrix<float, M, N> expected = A * B;
            const Eigen::Matrix<float, M, N> result = multiply<M, K, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, EPSILON);
        }
        // 4x4 * 4x4
        {
            constexpr size_t M = 4;
            constexpr size_t K = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, K> A = Eigen::Matrix<float, M, K>::Random();
            const Eigen::Matrix<float, K, N> B = Eigen::Matrix<float, K, N>::Random();
            const Eigen::Matrix<float, M, N> expected = A * B;
            const Eigen::Matrix<float, M, N> result = multiply<M, K, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, EPSILON);
        }
        // 6x4 * 4x6
        {
            constexpr size_t M = 6;
            constexpr size_t K = 4;
            constexpr size_t N = 6;
            const Eigen::Matrix<float, M, K> A = Eigen::Matrix<float, M, K>::Random();
            const Eigen::Matrix<float, K, N> B = Eigen::Matrix<float, K, N>::Random();
            const Eigen::Matrix<float, M, N> expected = A * B;
            const Eigen::Matrix<float, M, N> result = multiply<M, K, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, EPSILON);
        }
        // 6x4 * 4x4
        {
            constexpr size_t M = 6;
            constexpr size_t K = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, K> A = Eigen::Matrix<float, M, K>::Random();
            const Eigen::Matrix<float, K, N> B = Eigen::Matrix<float, K, N>::Random();
            const Eigen::Matrix<float, M, N> expected = A * B;
            const Eigen::Matrix<float, M, N> result = multiply<M, K, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, EPSILON);
        }
        // 6x4 * vector4
        {
            constexpr size_t M = 6;
            constexpr size_t K = 4;
            constexpr size_t N = 1;
            const Eigen::Matrix<float, M, K> A = Eigen::Matrix<float, M, K>::Random();
            const Eigen::Matrix<float, K, N> B = Eigen::Matrix<float, K, N>::Random();
            const Eigen::Vector<float, M> expected = A * B;
            const Eigen::Vector<float, M> result = multiply<M, K, N>(A, B);

            EXPECT_VECTOR_NEAR(expected, result, EPSILON);
        }
        // 10x20 * 20x30 large matrix
        {
            constexpr size_t M = 10;
            constexpr size_t K = 20;
            constexpr size_t N = 30;
            const Eigen::Matrix<float, M, K> A = Eigen::Matrix<float, M, K>::Random();
            const Eigen::Matrix<float, K, N> B = Eigen::Matrix<float, K, N>::Random();
            const Eigen::Matrix<float, M, N> expected = A * B;
            const Eigen::Matrix<float, M, N> result = multiply<M, K, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, EPSILON);
        }
        // 4x4 * scalar
        {
            constexpr size_t M = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, N> A = Eigen::Matrix<float, M, N>::Random();
            const float b = Eigen::Matrix<float, 1, 1>::Random()(0, 0);
            const Eigen::Matrix<float, M, N> expected = A * b;
            const Eigen::Matrix<float, M, N> result = multiply<M, N>(A, b);

            EXPECT_MATRIX_NEAR(expected, result, EPSILON);
        }
        // vector4 * scalar
        {
            constexpr size_t M = 4;
            const Eigen::Vector<float, M> A = Eigen::Vector<float, M>::Random();
            const float b = Eigen::Matrix<float, 1, 1>::Random()(0, 0);
            const Eigen::Vector<float, M> expected = A * b;
            const Eigen::Vector<float, M> result = multiply<M>(A, b);

            EXPECT_VECTOR_NEAR(expected, result, EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, multiply_inplace) {
    const float scalar = 2.5f;
    const Eigen::Matrix3f expected = A3x3 * scalar;
    Eigen::Matrix3f result = A3x3;
    multiply_inplace<3, 3>(result, scalar);
    EXPECT_MATRIX_NEAR(expected, result, EPSILON);
}

TEST_F(EigenUtilsTest, MatrixVectorMultiply) {
    // 4x4 * 4x1
    const Eigen::Vector4f expected4 = A4x4 * vec4;
    const Eigen::Vector4f result4 = multiply<4, 4>(A4x4, vec4);
    EXPECT_VECTOR_NEAR(expected4, result4, EPSILON);

    // 3x3 * 3x1
    const Eigen::Vector3f expected3 = A3x3 * vec3;
    const Eigen::Vector3f result3 = multiply<3, 3>(A3x3, vec3);
    EXPECT_VECTOR_NEAR(expected3, result3, EPSILON);
}

TEST_F(EigenUtilsTest, MatrixScalarMultiply) {
    const float scalar = 2.5f;
    const Eigen::Matrix3f expected = A3x3 * scalar;
    const Eigen::Matrix3f result = multiply<3, 3>(A3x3, scalar);
    EXPECT_MATRIX_NEAR(expected, result, EPSILON);
}

TEST_F(EigenUtilsTest, VectorScalarMultiply) {
    const float scalar = 3.0f;
    const Eigen::Vector3f expected = vec3 * scalar;
    const Eigen::Vector3f result = multiply<3>(vec3, scalar);
    EXPECT_VECTOR_NEAR(expected, result, EPSILON);
}

TEST_F(EigenUtilsTest, transpose) {
    // 3x3
    const Eigen::Matrix3f expected3 = A3x3.transpose();
    const Eigen::Matrix3f result3 = transpose<3, 3>(A3x3);
    EXPECT_MATRIX_NEAR(expected3, result3, EPSILON);

    // 4x4
    const Eigen::Matrix4f expected4 = A4x4.transpose();
    const Eigen::Matrix4f result4 = transpose<4, 4>(A4x4);
    EXPECT_MATRIX_NEAR(expected4, result4, EPSILON);

    // 4x6
    const Eigen::Matrix<float, 6, 4> expected6x4 = B4x6.transpose();
    const Eigen::Matrix<float, 6, 4> result6x4 = transpose<4, 6>(B4x6);
    EXPECT_MATRIX_NEAR(expected6x4, result6x4, EPSILON);
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
    EXPECT_VECTOR_NEAR(expected, result, EPSILON);
}

TEST_F(EigenUtilsTest, outer) {
    const Eigen::Vector4f v1(1.0f, 2.0f, 3.0f, 4.0f);
    const Eigen::Vector4f v2(5.0f, 6.0f, 7.0f, 8.0f);

    const Eigen::Matrix4f expected = v1 * v2.transpose();
    const Eigen::Matrix4f result = outer<4>(v1, v2);
    EXPECT_MATRIX_NEAR(expected, result, EPSILON);
}

TEST_F(EigenUtilsTest, block3x3) {
    const Eigen::Matrix3f expected = A4x4.block<3, 3>(0, 0);
    const Eigen::Matrix3f result = block3x3(A4x4);
    EXPECT_MATRIX_NEAR(expected, result, EPSILON);
}

TEST_F(EigenUtilsTest, inverse) {
    // regular matrix
    Eigen::Matrix3f invertible;
    invertible << 4.0f, 7.0f, 2.0f, 9.0f, 6.0f, 1.0f, 8.0f, 5.0f, 3.0f;

    const Eigen::Matrix3f expectedInv = invertible.inverse();
    const Eigen::Matrix3f resultInv = inverse(invertible);
    EXPECT_MATRIX_NEAR(expectedInv, resultInv, EPSILON);

    // Irregular matrix (det==0)
    Eigen::Matrix3f singular;
    singular << 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f, 7.0f, 8.0f, 9.0f;

    const Eigen::Matrix3f zeroMatrix = Eigen::Matrix3f::Zero();
    const Eigen::Matrix3f resultSingular = inverse(singular);
    EXPECT_MATRIX_NEAR(zeroMatrix, resultSingular, EPSILON);
}

TEST_F(EigenUtilsTest, ensure_symmetric) {
    // Create a non-symmetric matrix
    Eigen::Matrix3f non_symmetric;
    non_symmetric << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f;

    // Expected symmetric result (average of corresponding elements)
    Eigen::Matrix3f expected;
    expected << 1.0f, 3.0f, 5.0f, 3.0f, 5.0f, 7.0f, 5.0f, 7.0f, 9.0f;

    const Eigen::Matrix3f result = ensure_symmetric<3>(non_symmetric);
    EXPECT_MATRIX_NEAR(expected, result, EPSILON);

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
    EXPECT_MATRIX_NEAR(expected, result, EPSILON);
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
    EXPECT_VECTOR_NEAR(expected_eigenvalues, computed_eigenvalues, EPSILON);

    // Check that eigenvectors are valid
    for (int i = 0; i < 3; ++i) {
        // Apply matrix to eigenvector
        const Eigen::Vector3f Av = symmetric * computed_eigenvectors.col(i);
        // Should be close to lambda * v
        const Eigen::Vector3f lambda_v = computed_eigenvalues(i) * computed_eigenvectors.col(i);
        EXPECT_VECTOR_NEAR(Av, lambda_v, EPSILON);
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
    EXPECT_VECTOR_NEAR(expected_x, computed_x, EPSILON);

    // Also verify solution by checking A*x ≈ b
    const Eigen::Matrix<float, 6, 1> computed_b = A * computed_x;
    EXPECT_VECTOR_NEAR(b, computed_b, EPSILON);
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
    EXPECT_VECTOR_NEAR(input_vec, output_vec, EPSILON);

    // Test Matrix4f conversion
    const Eigen::Matrix4f input_mat = A4x4;
    const auto sycl_mat = to_sycl_vec(input_mat);
    const Eigen::Matrix4f output_mat = from_sycl_vec(sycl_mat);
    EXPECT_MATRIX_NEAR(input_mat, output_mat, EPSILON);

    // Test Matrix<float, 6, 6> conversion
    const Eigen::Matrix<float, 6, 6> input_mat6 = Eigen::Matrix<float, 6, 6>::Random();
    const auto sycl_mat6 = to_sycl_vec(input_mat6);
    const Eigen::Matrix<float, 6, 6> output_mat6 = from_sycl_vec(sycl_mat6);
    EXPECT_MATRIX_NEAR(input_mat6, output_mat6, EPSILON);

    // Test Vector<float, 6> conversion
    const Eigen::Vector<float, 6> input_vec6 = Eigen::Vector<float, 6>::Random();
    const auto sycl_vec6 = to_sycl_vec(input_vec6);
    const Eigen::Vector<float, 6> output_vec6 = from_sycl_vec(sycl_vec6);
    EXPECT_VECTOR_NEAR(input_vec6, output_vec6, EPSILON);
}

TEST_F(EigenUtilsTest, skew) {
    const Eigen::Vector3f v(1.0f, 2.0f, 3.0f);

    Eigen::Matrix3f expected;
    expected << 0.0f, -3.0f, 2.0f, 3.0f, 0.0f, -1.0f, -2.0f, 1.0f, 0.0f;

    const Eigen::Matrix3f result3 = lie::skew(v);
    EXPECT_MATRIX_NEAR(expected, result3, EPSILON);

    const Eigen::Vector4f v4(1.0f, 2.0f, 3.0f, 4.0f);
    const Eigen::Matrix3f result4 = lie::skew(v4);
    EXPECT_MATRIX_NEAR(expected, result4, EPSILON);
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
