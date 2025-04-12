#include <gtest/gtest.h>
#include <sycl_points/utils/eigen_utils.hpp>
#include <cmath>

namespace sycl_points {
namespace eigen_utils {
namespace test {

// 浮動小数点の比較のための許容誤差
constexpr float EPSILON = 1e-5f;


// マトリックス比較用のヘルパー関数 - C++17のテンプレート型推論を活用
template<typename Derived1, typename Derived2>
void expectMatrixNear(const Eigen::MatrixBase<Derived1>& expected,
                      const Eigen::MatrixBase<Derived2>& actual) {
  // 行列のサイズが一致することを確認
  ASSERT_EQ(expected.rows(), actual.rows()) << "Matrix row count mismatch";
  ASSERT_EQ(expected.cols(), actual.cols()) << "Matrix column count mismatch";

  for (int i = 0; i < expected.rows(); ++i) {
    for (int j = 0; j < expected.cols(); ++j) {
      EXPECT_NEAR(expected(i, j), actual(i, j), EPSILON)
        << "Matrices differ at (" << i << "," << j << ")";
    }
  }
}

// ベクトル比較用のヘルパー関数 - C++17のテンプレート型推論を活用
template<typename Derived1, typename Derived2>
void expectVectorNear(const Eigen::MatrixBase<Derived1>& expected,
                      const Eigen::MatrixBase<Derived2>& actual) {
  // ベクトルのサイズが一致することを確認
  ASSERT_EQ(expected.size(), actual.size()) << "Vector size mismatch";

  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected(i), actual(i), EPSILON)
      << "Vectors differ at index " << i;
  }
}

class EigenUtilsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // テスト用の一般的なマトリックスとベクトルを初期化
    A3x3 << 1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f;

    B3x3 << 9.0f, 8.0f, 7.0f,
            6.0f, 5.0f, 4.0f,
            3.0f, 2.0f, 1.0f;

    A4x4 << 1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f;

    B4x4 << 16.0f, 15.0f, 14.0f, 13.0f,
            12.0f, 11.0f, 10.0f, 9.0f,
            8.0f, 7.0f, 6.0f, 5.0f,
            4.0f, 3.0f, 2.0f, 1.0f;

    vec3 << 1.0f, 2.0f, 3.0f;
    vec4 << 1.0f, 2.0f, 3.0f, 4.0f;
    vec4_2 << 5.0f, 6.0f, 7.0f, 8.0f;

    A6x4 << 1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f,
            17.0f, 18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f, 24.0f;

    B4x6 << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
            19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f;
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

// add テスト
TEST_F(EigenUtilsTest, add) {
  // 3x3
  Eigen::Matrix3f expected3x3;
  expected3x3 << 10.0f, 10.0f, 10.0f,
                 10.0f, 10.0f, 10.0f,
                 10.0f, 10.0f, 10.0f;

  Eigen::Matrix3f result3x3 = add<3, 3>(A3x3, B3x3);
  expectMatrixNear(expected3x3, result3x3);

  // 4x4
  Eigen::Matrix4f expected4x4;
  expected4x4 << 17.0f, 17.0f, 17.0f, 17.0f,
                 17.0f, 17.0f, 17.0f, 17.0f,
                 17.0f, 17.0f, 17.0f, 17.0f,
                 17.0f, 17.0f, 17.0f, 17.0f;

  Eigen::Matrix4f result4x4 = add<4, 4>(A4x4, B4x4);
  expectMatrixNear(expected4x4, result4x4);

  // vector4
  Eigen::Vector4f expected4;
  expected4 << 6.0f, 8.0f, 10.0f, 12.0f;

  Eigen::Vector4f result4 = add<4, 1>(vec4, vec4_2);
  expectVectorNear(expected4, result4);
}

// add zerocopy テスト
TEST_F(EigenUtilsTest, add_zerocopy) {
  // 3x3
  Eigen::Matrix3f expected3x3;
  expected3x3 << 10.0f, 10.0f, 10.0f,
                 10.0f, 10.0f, 10.0f,
                 10.0f, 10.0f, 10.0f;

  Eigen::Matrix3f result3x3 = A3x3;
  add_zerocopy<3, 3>(result3x3, B3x3);
  expectMatrixNear(expected3x3, result3x3);
}

// subtract テスト
TEST_F(EigenUtilsTest, subtract) {
  Eigen::Matrix3f expected;
  expected << -8.0f, -6.0f, -4.0f,
               -2.0f,  0.0f,  2.0f,
                4.0f,  6.0f,  8.0f;

  Eigen::Matrix3f result = subtract<3, 3>(A3x3, B3x3);
  expectMatrixNear(expected, result);
}

// multiply (matrix x matrix) テスト
TEST_F(EigenUtilsTest, multiply) {
  // 3x3 * 3x3
  Eigen::Matrix3f C3x3 = A3x3 * B3x3;  // Eigenの組み込み乗算
  Eigen::Matrix3f result3x3 = multiply<3, 3, 3>(A3x3, B3x3);
  expectMatrixNear(C3x3, result3x3);

  // 4x4 * 4x4
  Eigen::Matrix4f C4x4 = A4x4 * B4x4;  // Eigenの組み込み乗算
  Eigen::Matrix4f result4x4 = multiply<4, 4, 4>(A4x4, B4x4);
  expectMatrixNear(C4x4, result4x4);

  // 6x4 * 4x6
  Eigen::Matrix<float, 6, 6> C6x6 = A6x4 * B4x6;  // Eigenの組み込み乗算
  Eigen::Matrix<float, 6, 6> result6x6 = multiply<6, 4, 6>(A6x4, B4x6);
  expectMatrixNear(C6x6, result6x6);

  // 6x4 * 4x4
  Eigen::Matrix<float, 6, 4> C6x4 = A6x4 * B4x4.block<4, 4>(0, 0);  // Eigenの組み込み乗算
  Eigen::Matrix<float, 6, 4> result6x4 = multiply<6, 4, 4>(A6x4, B4x4);
  expectMatrixNear(C6x4, result6x4);
}

// multiply (matrix x vector) テスト
TEST_F(EigenUtilsTest, MatrixVectorMultiply) {
  // 4x4 * 4x1
  Eigen::Vector4f expected4 = A4x4 * vec4;  // Eigenの組み込み乗算
  Eigen::Vector4f result4 = multiply<4, 4>(A4x4, vec4);
  expectVectorNear(expected4, result4);

  // 3x3 * 3x1
  Eigen::Vector3f expected3 = A3x3 * vec3;  // Eigenの組み込み乗算
  Eigen::Vector3f result3 = multiply<3, 3>(A3x3, vec3);
  expectVectorNear(expected3, result3);
}

// multiply (matrix x scalar) テスト
TEST_F(EigenUtilsTest, MatrixScalarMultiply) {
  float scalar = 2.5f;
  Eigen::Matrix3f expected = A3x3 * scalar;  // Eigenの組み込み乗算
  Eigen::Matrix3f result = multiply<3, 3>(A3x3, scalar);
  expectMatrixNear(expected, result);
}

// multiply (vector x scalar) テスト
TEST_F(EigenUtilsTest, VectorScalarMultiply) {
  float scalar = 3.0f;
  Eigen::Vector3f expected = vec3 * scalar;  // Eigenの組み込み乗算
  Eigen::Vector3f result = multiply<3>(vec3, scalar);
  expectVectorNear(expected, result);
}

// transpose テスト
TEST_F(EigenUtilsTest, transpose) {
  // 3x3
  Eigen::Matrix3f expected3 = A3x3.transpose();  // Eigenの組み込み関数
  Eigen::Matrix3f result3 = transpose<3, 3>(A3x3);
  expectMatrixNear(expected3, result3);

  // 4x4
  Eigen::Matrix4f expected4 = A4x4.transpose();  // Eigenの組み込み関数
  Eigen::Matrix4f result4 = transpose<4, 4>(A4x4);
  expectMatrixNear(expected4, result4);

  // 4x6
  Eigen::Matrix<float, 6, 4> expected6x4 = B4x6.transpose();  // Eigenの組み込み関数
  Eigen::Matrix<float, 6, 4> result6x4 = transpose<4, 6>(B4x6);
  expectMatrixNear(expected6x4, result6x4);
}

// dot テスト
TEST_F(EigenUtilsTest, dot) {
  // 3次元ベクトル
  float expected3 = vec3.dot(vec3);  // Eigenの組み込み関数
  float result3 = dot<3>(vec3, vec3);
  EXPECT_NEAR(expected3, result3, EPSILON);

  // 4次元ベクトル
  float expected4 = vec4.dot(vec4);  // Eigenの組み込み関数
  float result4 = dot<4>(vec4, vec4);
  EXPECT_NEAR(expected4, result4, EPSILON);
}

// cross テスト
TEST_F(EigenUtilsTest, cross) {
  Eigen::Vector3f v1(1.0f, 2.0f, 3.0f);
  Eigen::Vector3f v2(4.0f, 5.0f, 6.0f);

  Eigen::Vector3f expected = v1.cross(v2);  // Eigenの組み込み関数
  Eigen::Vector3f result = cross(v1, v2);
  expectVectorNear(expected, result);
}

// outer テスト
TEST_F(EigenUtilsTest, outer) {
  Eigen::Vector4f v1(1.0f, 2.0f, 3.0f, 4.0f);
  Eigen::Vector4f v2(5.0f, 6.0f, 7.0f, 8.0f);

  Eigen::Matrix4f expected = v1 * v2.transpose();
  Eigen::Matrix4f result = outer(v1, v2);
  expectMatrixNear(expected, result);
}

// block3x3 テスト
TEST_F(EigenUtilsTest, Block3x3) {
  Eigen::Matrix3f expected = A4x4.block<3, 3>(0, 0);  // Eigenの組み込み関数
  Eigen::Matrix3f result = block3x3(A4x4);
  expectMatrixNear(expected, result);
}

// inverse テスト
TEST_F(EigenUtilsTest, Inverse) {
  // 正則行列のケース
  Eigen::Matrix3f invertible;
  invertible << 4.0f, 7.0f, 2.0f,
                9.0f, 6.0f, 1.0f,
                8.0f, 5.0f, 3.0f;

  Eigen::Matrix3f expectedInv = invertible.inverse();  // Eigenの組み込み関数
  Eigen::Matrix3f resultInv = inverse(invertible);
  expectMatrixNear(expectedInv, resultInv);

  // 特異行列（非正則）のケース
  Eigen::Matrix3f singular;
  singular << 1.0f, 2.0f, 3.0f,
              2.0f, 4.0f, 6.0f,  // 行列式 = 0の行
              7.0f, 8.0f, 9.0f;

  Eigen::Matrix3f zeroMatrix = Eigen::Matrix3f::Zero();
  Eigen::Matrix3f resultSingular = inverse(singular);
  expectMatrixNear(zeroMatrix, resultSingular);
}

// skew テスト
TEST_F(EigenUtilsTest, Skew) {
  Eigen::Vector3f v(1.0f, 2.0f, 3.0f);

  Eigen::Matrix3f expected;
  expected << 0.0f, -3.0f, 2.0f,
              3.0f, 0.0f, -1.0f,
              -2.0f, 1.0f, 0.0f;

  // Vector3fバージョン
  Eigen::Matrix3f result3 = lie::skew(v);
  expectMatrixNear(expected, result3);

  // Vector4fバージョン（最初の3要素だけ使用）
  Eigen::Vector4f v4(1.0f, 2.0f, 3.0f, 4.0f);
  Eigen::Matrix3f result4 = lie::skew(v4);
  expectMatrixNear(expected, result4);
}

// so3_exp テスト
TEST_F(EigenUtilsTest, SO3Exp) {
  // 単位ベクトルと少し回転するケース
  Eigen::Vector3f small_rotation(0.1f, 0.2f, 0.3f);
  Eigen::Quaternionf quat1 = lie::so3_exp(small_rotation);

  // 回転ベクトル -> 回転行列 -> クォータニオンでの手動計算と比較
  float angle = small_rotation.norm();
  Eigen::Vector3f axis = small_rotation / angle;
  Eigen::AngleAxisf aa(angle, axis);
  Eigen::Quaternionf expected = Eigen::Quaternionf(aa);

  EXPECT_NEAR(expected.w(), quat1.w(), EPSILON);
  EXPECT_NEAR(expected.x(), quat1.x(), EPSILON);
  EXPECT_NEAR(expected.y(), quat1.y(), EPSILON);
  EXPECT_NEAR(expected.z(), quat1.z(), EPSILON);

  // ほぼゼロのケース
  Eigen::Vector3f zero_rotation(1e-10f, 2e-10f, 3e-10f);
  Eigen::Quaternionf quat2 = lie::so3_exp(zero_rotation);

  // ゼロ回転はクォータニオン (1,0,0,0) に近いはず
  EXPECT_NEAR(1.0f, quat2.w(), EPSILON);
  EXPECT_NEAR(0.0f, quat2.x(), EPSILON);
  EXPECT_NEAR(0.0f, quat2.y(), EPSILON);
  EXPECT_NEAR(0.0f, quat2.z(), EPSILON);
}

// se3_exp テスト
TEST_F(EigenUtilsTest, SE3Exp) {
  // 小さい回転と並進
  Eigen::Matrix<float, 6, 1> twist;
  twist << 0.1f, 0.2f, 0.3f, 1.0f, 2.0f, 3.0f;  // [rx, ry, rz, tx, ty, tz]

  Eigen::Isometry3f transform = lie::se3_exp(twist);

  // 回転部分は so3_exp と一致するはず
  Eigen::Vector3f rotation_part = twist.head<3>();
  Eigen::Quaternionf quat = lie::so3_exp(rotation_part);
  Eigen::Matrix3f expected_rotation = quat.toRotationMatrix();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(expected_rotation(i, j), transform.linear()(i, j), EPSILON)
        << "Rotation matrices differ at (" << i << "," << j << ")";
    }
  }

  // 並進部分の計算はより複雑なので、transform が互いに変換可能かどうかを確認
  // transform * point1 = point2 -> transform.inverse() * point2 = point1
  Eigen::Vector3f point1(5.0f, 6.0f, 7.0f);
  Eigen::Vector3f point2 = transform * point1;
  Eigen::Vector3f point1_back = transform.inverse() * point2;

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(point1(i), point1_back(i), EPSILON)
      << "Points differ after transform and inverse at index " << i;
  }

  // ゼロに近い回転の場合
  Eigen::Matrix<float, 6, 1> small_twist;
  small_twist << 1e-10f, 2e-10f, 3e-10f, 1.0f, 2.0f, 3.0f;

  Eigen::Isometry3f small_transform = lie::se3_exp(small_twist);

  // 回転部分は単位行列に近いはず
  Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(identity(i, j), small_transform.linear()(i, j), EPSILON)
        << "Small rotation matrix differs from identity at (" << i << "," << j << ")";
    }
  }

  // 並進部分はtwistの後半と同じはず
  Eigen::Vector3f trans = small_transform.translation();
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(small_twist(i + 3), trans(i), EPSILON)
      << "Small translation differs at index " << i;
  }
}

}  // namespace test
}  // namespace eigen_utils
}  // namespace sycl_points

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
