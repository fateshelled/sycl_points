#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <sycl_points/utils/eigen_utils.hpp>

namespace sycl_points {
namespace eigen_utils {
namespace test {

constexpr float BASE_EPSILON = 1e-5f;
constexpr float DET_EPSILON = 1e-4f;
constexpr float MATMUL_EPSILON = 1e-4f;
constexpr float INVERSE_EPSILON = 1e-3f;
constexpr size_t TEST_ITERATIONS = 1000;
constexpr float RANDOM_SCALE = 10.0f;  // Range scale for random values

// Generate a random floating-point number in [-RANDOM_SCALE, RANDOM_SCALE]
inline float random_float() {
    // rand()/RAND_MAX -> [0,1], scale to [-RANDOM_SCALE, RANDOM_SCALE]
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f * RANDOM_SCALE - RANDOM_SCALE;
}

// Generate a random matrix of size MxN with elements in [-RANDOM_SCALE, RANDOM_SCALE]
template <int M, int N>
Eigen::Matrix<float, M, N> random_matrix() {
    return Eigen::Matrix<float, M, N>::NullaryExpr([]() { return random_float(); });
}

// Generate a random vector of size M with elements in [-RANDOM_SCALE, RANDOM_SCALE]
template <int M>
Eigen::Matrix<float, M, 1> random_vector() {
    return Eigen::Matrix<float, M, 1>::NullaryExpr([]() { return random_float(); });
}

// Generate a random scalar in [-RANDOM_SCALE, RANDOM_SCALE]
inline float random_scalar() {
    return random_float();
}

::testing::AssertionResult AssertMatrixExactEqual(const char* expr1, const char* expr2, const Eigen::MatrixXf& m1,
                                                  const Eigen::MatrixXf& m2) {
    if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
        return ::testing::AssertionFailure() << "Matrix size mismatch:\n"
                                             << expr1 << " is " << m1.rows() << "x" << m1.cols() << ",\n"
                                             << expr2 << " is " << m2.rows() << "x" << m2.cols();
    }

    for (int i = 0; i < m1.rows(); ++i) {
        for (int j = 0; j < m1.cols(); ++j) {
            if (m1(i, j) != m2(i, j)) {
                return ::testing::AssertionFailure() << "Matrix not equal at (" << i << "," << j << ")\n"
                                                     << expr1 << ":\n"
                                                     << m1 << "\n"
                                                     << expr2 << ":\n"
                                                     << m2 << std::endl;
            }
        }
    }

    return ::testing::AssertionSuccess();
}

::testing::AssertionResult AssertVectorExactEqual(const char* expr1, const char* expr2, const Eigen::VectorXf& m1,
                                                  const Eigen::VectorXf& m2) {
    if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
        return ::testing::AssertionFailure() << "Matrix size mismatch:\n"
                                             << expr1 << " is " << m1.rows() << "x" << m1.cols() << ",\n"
                                             << expr2 << " is " << m2.rows() << "x" << m2.cols();
    }

    for (int i = 0; i < m1.size(); ++i) {
        if (m1(i) != m2(i)) {
            return ::testing::AssertionFailure() << "Vector not equal at (" << i << "):\n"
                                                 << expr1 << ":\n"
                                                 << m1.transpose() << "\n"
                                                 << expr2 << ":\n"
                                                 << m2.transpose() << std::endl;
        }
    }

    return ::testing::AssertionSuccess();
}

::testing::AssertionResult AssertMatrixNear(const char* expr1, const char* expr2, const char* expr3,
                                            const Eigen::MatrixXf& m1, const Eigen::MatrixXf& m2, double threshold) {
    if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
        return ::testing::AssertionFailure() << "Matrix size mismatch:\n"
                                             << expr1 << " is " << m1.rows() << "x" << m1.cols() << ",\n"
                                             << expr2 << " is " << m2.rows() << "x" << m2.cols();
    }

    const Eigen::MatrixXf abs_diff = (m1 - m2).cwiseAbs();
    const Eigen::MatrixXf rel_diff = abs_diff.cwiseQuotient(m1.cwiseAbs().cwiseMax(m2.cwiseAbs()));

    for (int i = 0; i < abs_diff.rows(); ++i) {
        for (int j = 0; j < abs_diff.cols(); ++j) {
            if (abs_diff(i, j) > threshold && rel_diff(i, j) > threshold) {
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
                                            const Eigen::VectorXf& v1, const Eigen::VectorXf& v2, double threshold) {
    if (v1.size() != v2.size()) {
        return ::testing::AssertionFailure() << "Vector size mismatch:\n"
                                             << expr1 << " is " << v1.size() << ",\n"
                                             << expr2 << " is " << v2.size();
    }

    const Eigen::VectorXf abs_diff = (v1 - v2).cwiseAbs();
    const Eigen::VectorXf rel_diff = abs_diff.cwiseQuotient(v1.cwiseAbs().cwiseMax(v2.cwiseAbs()));

    for (int i = 0; i < abs_diff.size(); ++i) {
        if (abs_diff(i) > threshold && rel_diff(i) > threshold) {
            return ::testing::AssertionFailure() << "Vector differ at (" << i << "): threshold = " << threshold << "\n"
                                                 << expr1 << ":\n"
                                                 << v1.transpose() << "\n"
                                                 << expr2 << ":\n"
                                                 << v2.transpose() << std::endl;
        }
    }

    return ::testing::AssertionSuccess();
}

template <typename T>
::testing::AssertionResult AssertScalarNear(const char* expr1, const char* expr2, const char* expr3, const T& v1,
                                            const T& v2, double threshold) {
    const T abs_diff = std::fabs(v1 - v2);
    const T rel_diff = abs_diff / (std::max(std::fabs(v1), std::fabs(v2)));

    if (abs_diff > threshold && rel_diff > threshold) {
        return ::testing::AssertionFailure() << "Scalr differ: threshold = " << threshold << "\n"
                                             << expr1 << ": " << v1 << "\n"
                                             << expr2 << ": " << v2 << "\n"
                                             << std::endl;
    }
    return ::testing::AssertionSuccess();
}

#define EXPECT_MATRIX_NEAR(m1, m2, threshold) EXPECT_PRED_FORMAT3(AssertMatrixNear, m1, m2, threshold)
#define ASSERT_MATRIX_NEAR(m1, m2, threshold) ASSERT_PRED_FORMAT3(AssertMatrixNear, m1, m2, threshold)

#define EXPECT_VECTOR_NEAR(m1, m2, threshold) EXPECT_PRED_FORMAT3(AssertVectorNear, m1, m2, threshold)
#define ASSERT_VECTOR_NEAR(m1, m2, threshold) ASSERT_PRED_FORMAT3(AssertVectorNear, m1, m2, threshold)

#define EXPECT_SCALAR_NEAR(m1, m2, threshold) EXPECT_PRED_FORMAT3(AssertScalarNear, m1, m2, threshold)
#define ASSERT_SCALAR_NEAR(m1, m2, threshold) ASSERT_PRED_FORMAT3(AssertScalarNear, m1, m2, threshold)

#define EXPECT_MATRIX_EXACT_EQ(m1, m2) EXPECT_PRED_FORMAT2(AssertMatrixExactEqual, m1, m2)
#define ASSERT_MATRIX_EXACT_EQ(m1, m2) ASSERT_PRED_FORMAT2(AssertMatrixExactEqual, m1, m2)

#define EXPECT_VECTOR_EXACT_EQ(m1, m2) EXPECT_PRED_FORMAT2(AssertVectorExactEqual, m1, m2)
#define ASSERT_VECTOR_EXACT_EQ(m1, m2) ASSERT_PRED_FORMAT2(AssertVectorExactEqual, m1, m2)

class EigenUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed the random number generator
        srand(this->seed);
    }
    const uint32_t seed = 1234;
};

TEST_F(EigenUtilsTest, add) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        // 3x3
        {
            constexpr size_t M = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> B = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> expected = A + B;
            const Eigen::Matrix<float, M, N> result = add<M, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, BASE_EPSILON);
        }
        // 4x4
        {
            constexpr size_t M = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> B = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> expected = A + B;
            const Eigen::Matrix<float, M, N> result = add<M, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, BASE_EPSILON);
        }
        // vector3
        {
            constexpr size_t M = 3;
            const Eigen::Vector<float, M> A = random_vector<M>();
            const Eigen::Vector<float, M> B = random_vector<M>();
            const Eigen::Vector<float, M> expected = A + B;
            const Eigen::Vector<float, M> result = add<M, 1>(A, B);

            EXPECT_VECTOR_NEAR(expected, result, BASE_EPSILON);
        }
        // vector4
        {
            constexpr size_t M = 4;
            const Eigen::Vector<float, M> A = random_vector<M>();
            const Eigen::Vector<float, M> B = random_vector<M>();
            const Eigen::Vector<float, M> expected = A + B;
            const Eigen::Vector<float, M> result = add<M, 1>(A, B);

            EXPECT_VECTOR_NEAR(expected, result, BASE_EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, add_inplace) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        // 3x3
        {
            constexpr size_t M = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> B = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> expected = A + B;
            Eigen::Matrix<float, M, N> actual = A;
            add_inplace<M, N>(actual, B);

            EXPECT_MATRIX_NEAR(expected, actual, BASE_EPSILON);
        }
        // 4x4
        {
            constexpr size_t M = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> B = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> expected = A + B;
            Eigen::Matrix<float, M, N> actual = A;
            add_inplace<M, N>(actual, B);

            EXPECT_MATRIX_NEAR(expected, actual, BASE_EPSILON);
        }
        // vector3
        {
            constexpr size_t M = 3;
            constexpr size_t N = 1;
            const Eigen::Vector<float, M> A = random_vector<M>();
            const Eigen::Vector<float, M> B = random_vector<M>();
            const Eigen::Vector<float, M> expected = A + B;
            Eigen::Vector<float, M> actual = A;
            add_inplace<M, 1>(actual, B);

            EXPECT_VECTOR_NEAR(expected, actual, BASE_EPSILON);
        }
        // vector4
        {
            constexpr size_t M = 4;
            const Eigen::Vector<float, M> A = random_vector<M>();
            const Eigen::Vector<float, M> B = random_vector<M>();
            const Eigen::Vector<float, M> expected = A + B;
            Eigen::Vector<float, M> actual = A;
            add_inplace<M, 1>(actual, B);

            EXPECT_VECTOR_NEAR(expected, actual, BASE_EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, subtract) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        // 3x3
        {
            constexpr size_t M = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> B = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> expected = A - B;
            const Eigen::Matrix<float, M, N> result = subtract<M, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, BASE_EPSILON);
        }
        // 4x4
        {
            constexpr size_t M = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> B = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> expected = A - B;
            const Eigen::Matrix<float, M, N> result = subtract<M, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, BASE_EPSILON);
        }
        // vector3
        {
            constexpr size_t M = 3;
            const Eigen::Vector<float, M> A = random_vector<M>();
            const Eigen::Vector<float, M> B = random_vector<M>();
            const Eigen::Vector<float, M> expected = A - B;
            const Eigen::Vector<float, M> result = subtract<M, 1>(A, B);

            EXPECT_VECTOR_NEAR(expected, result, BASE_EPSILON);
        }
        // vector4
        {
            constexpr size_t M = 4;
            const Eigen::Vector<float, M> A = random_vector<M>();
            const Eigen::Vector<float, M> B = random_vector<M>();
            const Eigen::Vector<float, M> expected = A - B;
            const Eigen::Vector<float, M> result = subtract<M, 1>(A, B);

            EXPECT_VECTOR_NEAR(expected, result, BASE_EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, multiply) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        // 3x3 * 3x3
        {
            constexpr size_t M = 3;
            constexpr size_t K = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, K> A = random_matrix<M, K>();
            const Eigen::Matrix<float, K, N> B = random_matrix<K, N>();
            const Eigen::Matrix<float, M, N> expected = A * B;
            const Eigen::Matrix<float, M, N> result = multiply<M, K, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, MATMUL_EPSILON);
        }
        // 4x4 * 4x4
        {
            constexpr size_t M = 4;
            constexpr size_t K = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, K> A = random_matrix<M, K>();
            const Eigen::Matrix<float, K, N> B = random_matrix<K, N>();
            const Eigen::Matrix<float, M, N> expected = A * B;
            const Eigen::Matrix<float, M, N> result = multiply<M, K, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, MATMUL_EPSILON);
        }
        // 6x4 * 4x6
        {
            constexpr size_t M = 6;
            constexpr size_t K = 4;
            constexpr size_t N = 6;
            const Eigen::Matrix<float, M, K> A = random_matrix<M, K>();
            const Eigen::Matrix<float, K, N> B = random_matrix<K, N>();
            const Eigen::Matrix<float, M, N> expected = A * B;
            const Eigen::Matrix<float, M, N> result = multiply<M, K, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, MATMUL_EPSILON);
        }
        // 6x4 * 4x4
        {
            constexpr size_t M = 6;
            constexpr size_t K = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, K> A = random_matrix<M, K>();
            const Eigen::Matrix<float, K, N> B = random_matrix<K, N>();
            const Eigen::Matrix<float, M, N> expected = A * B;
            const Eigen::Matrix<float, M, N> result = multiply<M, K, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, MATMUL_EPSILON);
        }
        // 6x4 * vector4
        {
            constexpr size_t M = 6;
            constexpr size_t K = 4;
            constexpr size_t N = 1;
            const Eigen::Matrix<float, M, K> A = random_matrix<M, K>();
            const Eigen::Matrix<float, K, N> B = random_matrix<K, N>();
            const Eigen::Vector<float, M> expected = A * B;
            const Eigen::Vector<float, M> result = multiply<M, K, N>(A, B);

            EXPECT_VECTOR_NEAR(expected, result, BASE_EPSILON);
        }
        // 10x20 * 20x30 large matrix
        {
            constexpr size_t M = 10;
            constexpr size_t K = 20;
            constexpr size_t N = 30;
            const Eigen::Matrix<float, M, K> A = random_matrix<M, K>();
            const Eigen::Matrix<float, K, N> B = random_matrix<K, N>();
            const Eigen::Matrix<float, M, N> expected = A * B;
            const Eigen::Matrix<float, M, N> result = multiply<M, K, N>(A, B);

            EXPECT_MATRIX_NEAR(expected, result, MATMUL_EPSILON);
        }
        // 4x4 * scalar
        {
            constexpr size_t M = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const float b = random_scalar();
            const Eigen::Matrix<float, M, N> expected = A * b;
            const Eigen::Matrix<float, M, N> result = multiply<M, N>(A, b);

            EXPECT_MATRIX_NEAR(expected, result, BASE_EPSILON);
        }
        // vector4 * scalar
        {
            constexpr size_t M = 4;
            const Eigen::Vector<float, M> A = random_vector<M>();
            const float b = random_scalar();
            const Eigen::Vector<float, M> expected = A * b;
            const Eigen::Vector<float, M> result = multiply<M>(A, b);

            EXPECT_VECTOR_NEAR(expected, result, BASE_EPSILON);
        }
    }
}

// Additional random-loop tests
TEST_F(EigenUtilsTest, multiply_inplace) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        // 3x3
        {
            constexpr size_t M = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const float scalar = random_scalar();
            const Eigen::Matrix<float, M, N> expected = A * scalar;
            Eigen::Matrix<float, M, N> actual = A;
            multiply_inplace<M, N>(actual, scalar);
            EXPECT_MATRIX_NEAR(expected, actual, BASE_EPSILON);
        }
        // 4x4
        {
            constexpr size_t M = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const float scalar = random_scalar();
            const Eigen::Matrix<float, M, N> expected = A * scalar;
            Eigen::Matrix<float, M, N> actual = A;
            multiply_inplace<M, N>(actual, scalar);
            EXPECT_MATRIX_NEAR(expected, actual, BASE_EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, element_wise_multiply) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        // 3x3
        {
            constexpr size_t M = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> B = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> expected = A.cwiseProduct(B);
            const Eigen::Matrix<float, M, N> result = element_wise_multiply<M, N>(A, B);
            EXPECT_MATRIX_NEAR(expected, result, BASE_EPSILON);
        }
        // 4x4
        {
            constexpr size_t M = 4;
            constexpr size_t N = 4;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> B = random_matrix<M, N>();
            const Eigen::Matrix<float, M, N> expected = A.cwiseProduct(B);
            const Eigen::Matrix<float, M, N> result = element_wise_multiply<M, N>(A, B);
            EXPECT_MATRIX_NEAR(expected, result, BASE_EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, transpose) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        // 3x3
        {
            constexpr size_t M = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const Eigen::Matrix<float, N, M> expected = A.transpose();
            const Eigen::Matrix<float, N, M> result = transpose<M, N>(A);
            EXPECT_MATRIX_EXACT_EQ(expected, result);
        }
        // 4x6
        {
            constexpr size_t M = 4;
            constexpr size_t N = 6;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const Eigen::Matrix<float, N, M> expected = A.transpose();
            const Eigen::Matrix<float, N, M> result = transpose<M, N>(A);
            EXPECT_MATRIX_EXACT_EQ(expected, result);
        }
    }
}

TEST_F(EigenUtilsTest, dot) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        // size 3
        {
            constexpr size_t M = 3;
            const Eigen::Vector<float, M> a = random_vector<M>();
            const Eigen::Vector<float, M> b = random_vector<M>();
            const float expected = a.dot(b);
            const float result = dot<M>(a, b);
            EXPECT_SCALAR_NEAR(expected, result, BASE_EPSILON);
        }
        // size 4
        {
            constexpr size_t M = 4;
            const Eigen::Vector<float, M> a = random_vector<M>();
            const Eigen::Vector<float, M> b = random_vector<M>();
            const float expected = a.dot(b);
            const float result = dot<M>(a, b);
            EXPECT_SCALAR_NEAR(expected, result, BASE_EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, cross) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        const Eigen::Vector3f a = random_vector<3>();
        const Eigen::Vector3f b = random_vector<3>();
        const Eigen::Vector3f expected = a.cross(b);
        const Eigen::Vector3f result = cross(a, b);
        EXPECT_VECTOR_NEAR(expected, result, BASE_EPSILON);
    }
}

TEST_F(EigenUtilsTest, outer) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        const Eigen::Vector4f a = random_vector<4>();
        const Eigen::Vector4f b = random_vector<4>();
        const Eigen::Matrix4f expected = a * b.transpose();
        const Eigen::Matrix4f result = outer<4>(a, b);
        EXPECT_MATRIX_NEAR(expected, result, BASE_EPSILON);
    }
}

TEST_F(EigenUtilsTest, block3x3) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        const Eigen::Matrix4f A = random_matrix<4, 4>();
        const Eigen::Matrix3f expected = A.block<3, 3>(0, 0);
        const Eigen::Matrix3f result = block3x3(A);
        EXPECT_MATRIX_EXACT_EQ(expected, result);
    }
}

TEST_F(EigenUtilsTest, inverse) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        Eigen::Matrix3f A = random_matrix<3, 3>();

        const Eigen::Matrix3f expected = A.inverse();
        const Eigen::Matrix3f invA = inverse(A);
        EXPECT_MATRIX_NEAR(expected, invA, INVERSE_EPSILON);

        const Eigen::Matrix3f identity = A * invA;
        EXPECT_MATRIX_NEAR(Eigen::Matrix3f::Identity(), identity, INVERSE_EPSILON);
    }

    Eigen::Matrix3f singular;
    singular << 1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f, 7.0f, 8.0f, 9.0f;
    const Eigen::Matrix3f zero = Eigen::Matrix3f::Zero();
    const Eigen::Matrix3f result = inverse(singular);
    EXPECT_MATRIX_NEAR(zero, result, INVERSE_EPSILON);
}

TEST_F(EigenUtilsTest, ensure_symmetric) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        Eigen::Matrix3f A = random_matrix<3, 3>();
        const Eigen::Matrix3f expected = 0.5f * (A + A.transpose());
        const Eigen::Matrix3f result = ensure_symmetric<3>(A);
        EXPECT_MATRIX_NEAR(expected, result, BASE_EPSILON);
        EXPECT_MATRIX_NEAR(result, result.transpose(), BASE_EPSILON);
    }
}

TEST_F(EigenUtilsTest, frobenius_norm) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        {
            constexpr size_t M = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            const float expected = A.norm();
            const float result = frobenius_norm<M, N>(A);
            EXPECT_SCALAR_NEAR(expected, result, BASE_EPSILON);
        }
        {
            constexpr size_t M = 3;
            const Eigen::Vector<float, M> v = random_vector<M>();
            const float expected = v.norm();
            const float result = frobenius_norm<M>(v);
            EXPECT_SCALAR_NEAR(expected, result, BASE_EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, frobenius_norm_squared) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        constexpr size_t M = 3;
        const Eigen::Vector<float, M> v = random_vector<M>();
        const float expected = v.squaredNorm();
        const float result = frobenius_norm_squared<M>(v);
        EXPECT_SCALAR_NEAR(expected, result, BASE_EPSILON);
    }
}

TEST_F(EigenUtilsTest, determinant) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        Eigen::Matrix3f A = random_matrix<3, 3>();
        const float expected = A.determinant();
        const float result = determinant(A);
        EXPECT_SCALAR_NEAR(expected, result, DET_EPSILON);
    }
}

TEST_F(EigenUtilsTest, as_diagonal) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        const Eigen::Vector3f diag = random_vector<3>();
        Eigen::Matrix3f expected = Eigen::Matrix3f::Zero();
        expected.diagonal() = diag;
        const Eigen::Matrix3f result = as_diagonal<3>(diag);
        EXPECT_MATRIX_NEAR(expected, result, BASE_EPSILON);
    }
}

TEST_F(EigenUtilsTest, symmetric_eigen_decomposition_3x3) {
    Eigen::Matrix3f symmetric;
    symmetric << 2.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 2.0f;
    Eigen::Vector3f values;
    Eigen::Matrix3f vectors;
    symmetric_eigen_decomposition_3x3(symmetric, values, vectors);
    const Eigen::Matrix3f reconstructed = vectors * values.asDiagonal() * vectors.transpose();
    EXPECT_MATRIX_NEAR(symmetric, reconstructed, BASE_EPSILON);
}

TEST_F(EigenUtilsTest, trace) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        const Eigen::Matrix3f A = random_matrix<3, 3>();
        const float expected = A.trace();
        const float result = trace<3>(A);
        EXPECT_SCALAR_NEAR(expected, result, BASE_EPSILON);
    }
}

TEST_F(EigenUtilsTest, copy_swap) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        {
            constexpr size_t M = 3;
            constexpr size_t N = 3;
            const Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            Eigen::Matrix<float, M, N> B;
            copy<M, N>(A, B);
            EXPECT_MATRIX_EXACT_EQ(A, B);
        }
        {
            constexpr size_t M = 4;
            constexpr size_t N = 4;
            Eigen::Matrix<float, M, N> A = random_matrix<M, N>();
            Eigen::Matrix<float, M, N> B = random_matrix<M, N>();
            Eigen::Matrix<float, M, N> A_orig = A;
            Eigen::Matrix<float, M, N> B_orig = B;
            swap<M, N>(A, B);
            EXPECT_MATRIX_EXACT_EQ(A_orig, B);
            EXPECT_MATRIX_EXACT_EQ(B_orig, A);
        }
    }
}

TEST_F(EigenUtilsTest, normalize) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        Eigen::Vector3f v = random_vector<3>();
        if (v.norm() < 1e-3f) {
            v(0) += 1.0f;
        }
        const Eigen::Vector3f expected = v.normalized();
        const Eigen::Vector3f result = normalize<3>(v);
        EXPECT_VECTOR_NEAR(expected, result, BASE_EPSILON);
    }
    const Eigen::Vector3f zero = Eigen::Vector3f::Zero();
    const Eigen::Vector3f result = normalize<3>(zero);
    EXPECT_VECTOR_NEAR(zero, result, BASE_EPSILON);
}

TEST_F(EigenUtilsTest, to_from_sycl_vec) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        {
            const Eigen::Vector4f v = random_vector<4>();
            const sycl::float4 sv = to_sycl_vec(v);
            const Eigen::Vector4f back = from_sycl_vec(sv);
            EXPECT_VECTOR_NEAR(v, back, BASE_EPSILON);
        }
        {
            const Eigen::Matrix4f m = random_matrix<4, 4>();
            const auto sm = to_sycl_vec(m);
            const Eigen::Matrix4f back = from_sycl_vec(sm);
            EXPECT_MATRIX_NEAR(m, back, BASE_EPSILON);
        }
        {
            const Eigen::Matrix<float, 6, 6> m = random_matrix<6, 6>();
            const auto sm = to_sycl_vec(m);
            const Eigen::Matrix<float, 6, 6> back = from_sycl_vec(sm);
            EXPECT_MATRIX_NEAR(m, back, BASE_EPSILON);
        }
        {
            const Eigen::Vector<float, 6> v = random_vector<6>();
            const auto sv = to_sycl_vec(v);
            const Eigen::Vector<float, 6> back = from_sycl_vec(sv);
            EXPECT_VECTOR_NEAR(v, back, BASE_EPSILON);
        }
    }
}

TEST_F(EigenUtilsTest, so3_exp_log) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        const Eigen::Vector3f omega = Eigen::Vector3f::Random();
        const Eigen::Quaternionf q = lie::so3_exp(omega);
        const Eigen::Vector3f back = lie::so3_log(q);
        EXPECT_VECTOR_NEAR(omega, back, BASE_EPSILON);
    }
}

TEST_F(EigenUtilsTest, se3_exp_log) {
    for (size_t iter = 0; iter < TEST_ITERATIONS; ++iter) {
        Eigen::Matrix<float, 6, 1> twist = Eigen::Matrix<float, 6, 1>::Random();
        const Eigen::Isometry3f T = lie::se3_exp(twist);
        const Eigen::Matrix<float, 6, 1> back = lie::se3_log(T);
        EXPECT_VECTOR_NEAR(twist, back, BASE_EPSILON);
    }
}

}  // namespace test
}  // namespace eigen_utils
}  // namespace sycl_points

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
