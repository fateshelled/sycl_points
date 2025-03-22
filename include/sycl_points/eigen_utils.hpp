#pragma once

#include <Eigen/Dense>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace sycl_points {
namespace eigen_utils {
// Eigen::Matrix is column major

// A + B = C
// row: M, col: N
template <size_t M, size_t N>
SYCL_EXTERNAL inline Eigen::Matrix<float, M, N> add(const Eigen::Matrix<float, M, N>& A, const Eigen::Matrix<float, M, N>& B) {
  Eigen::Matrix<float, M, N> ret;
#pragma unroll
  for (size_t i = 0; i < M; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      ret(i, j) = A(i, j) + B(i, j);
    }
  }
  return ret;
}

template <size_t M, size_t N>
SYCL_EXTERNAL inline void add_zerocopy(Eigen::Matrix<float, M, N>& A, const Eigen::Matrix<float, M, N>& B) {
#pragma unroll
  for (size_t i = 0; i < M; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      A(i, j) += B(i, j);
    }
  }
}

// A - B = C
// row: M, col: N
template <size_t M, size_t N>
SYCL_EXTERNAL inline Eigen::Matrix<float, M, N> subtract(const Eigen::Matrix<float, M, N>& A, const Eigen::Matrix<float, M, N>& B) {
  Eigen::Matrix<float, M, N> ret;
#pragma unroll
  for (size_t i = 0; i < M; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      ret(i, j) = A(i, j) - B(i, j);
    }
  }
  return ret;
}

// A * B = C
// row: M, col: N
template <size_t M, size_t K, size_t N>
SYCL_EXTERNAL inline Eigen::Matrix<float, M, N> multiply(const Eigen::Matrix<float, M, K>& A, const Eigen::Matrix<float, K, N>& B) {
  Eigen::Matrix<float, M, N> ret = Eigen::Matrix<float, M, N>::Zero();
#pragma unroll
  for (size_t i = 0; i < M; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
#pragma unroll
      for (size_t k = 0; k < K; ++k) {
        ret(i, j) += A(i, k) * B(k, j);
      }
    }
  }
  return ret;
}

// A * v = r
// row: M, col: N
template <size_t M, size_t N>
SYCL_EXTERNAL inline Eigen::Vector<float, M> multiply(const Eigen::Matrix<float, M, N>& A, const Eigen::Vector<float, N>& v) {
  Eigen::Vector<float, M> ret = Eigen::Vector<float, M>::Zero();
#pragma unroll
  for (size_t i = 0; i < M; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      ret(i) += A(i, j) * v(j);
    }
  }
  return ret;
}


// A * s = B
// row: M, col: N
template <size_t M, size_t N>
SYCL_EXTERNAL inline Eigen::Matrix<float, M, N> multiply(const Eigen::Matrix<float, M, N>& A, float scalar) {
  Eigen::Matrix<float, M, N> ret;
#pragma unroll
  for (size_t i = 0; i < M; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      ret(i, j) = A(i, j) * scalar;
    }
  }
  return ret;
}

// A * s = B
template <size_t N>
SYCL_EXTERNAL inline Eigen::Vector<float, N> multiply(const Eigen::Vector<float, N>& a, float scalar) {
  Eigen::Vector<float, N> ret;
#pragma unroll
  for (size_t i = 0; i < N; ++i) {
    ret(i) = a(i) * scalar;
  }
  return ret;
}

// trans(A) = B
// row: M, col: N
template <size_t M, size_t N>
SYCL_EXTERNAL inline Eigen::Matrix<float, N, M> transpose(const Eigen::Matrix<float, M, N>& A) {
  Eigen::Matrix<float, N, M> ret;
#pragma unroll
  for (size_t i = 0; i < M; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      ret(j, i) = A(i, j);
    }
  }
  return ret;
}

// dot: u·v
template <size_t N>
SYCL_EXTERNAL inline float dot(const Eigen::Vector<float, N>& u, const Eigen::Vector<float, N>& v) {
  float result = 0.0f;
#pragma unroll
  for (size_t i = 0; i < N; ++i) {
    result += u(i, 0) * v(i, 0);
  }
  return result;
}

// cross: u × v = w
SYCL_EXTERNAL inline Eigen::Vector3f cross(const Eigen::Vector3f& u, const Eigen::Vector3f& v) {
  Eigen::Vector3f ret;
  ret << u(1) * v(2) - u(2) * v(1),  // nolint
    u(2) * v(0) - u(0) * v(2),       // nolint
    u(0) * v(1) - u(1) * v(0);
  return ret;
}

// outer: u ⊗ v = w
SYCL_EXTERNAL inline Eigen::Matrix4f outer(const Eigen::Vector4f& u, const Eigen::Vector4f& v) {
  Eigen::Matrix4f ret;
  ret << u(0) * v(0), u(0) * v(1), u(0) * v(2), u(0) * v(3),  // nolint
    u(1) * v(0), u(1) * v(1), u(1) * v(2), u(1) * v(3),       // nolint
    u(2) * v(0), u(2) * v(1), u(2) * v(2), u(2) * v(3),       // nolint
    u(3) * v(0), u(3) * v(1), u(3) * v(2), u(3) * v(3);
  return ret;
}

SYCL_EXTERNAL inline Eigen::Matrix3f block3x3(const Eigen::Matrix4f& src) {
  Eigen::Matrix3f ret;
  ret << src(0, 0), src(0, 1), src(0, 2),  // nolint
    src(1, 0), src(1, 1), src(1, 2),       // nolint
    src(2, 0), src(2, 1), src(2, 2);
  return ret;
}

SYCL_EXTERNAL inline Eigen::Matrix3f inverse(const Eigen::Matrix3f& src) {
  const float det = src(0, 0) * (src(1, 1) * src(2, 2) - src(1, 2) * src(2, 1))    // nolint
                    - src(0, 1) * (src(1, 0) * src(2, 2) - src(1, 2) * src(2, 0))  // nolint
                    + src(0, 2) * (src(1, 0) * src(2, 1) - src(1, 1) * src(2, 0));

  if (sycl::fabs(det) < 1e-6f) {
    return Eigen::Matrix3f::Zero();
  }

  const float invDet = 1.0f / det;

  Eigen::Matrix3f ret;
  ret << (src(1, 1) * src(2, 2) - src(1, 2) * src(2, 1)) * invDet,  // nolint
    (src(0, 2) * src(2, 1) - src(0, 1) * src(2, 2)) * invDet,       // nolint
    (src(0, 1) * src(1, 2) - src(0, 2) * src(1, 1)) * invDet,       // nolint
    (src(1, 2) * src(2, 0) - src(1, 0) * src(2, 2)) * invDet,       // nolint
    (src(0, 0) * src(2, 2) - src(0, 2) * src(2, 0)) * invDet,       // nolint
    (src(0, 2) * src(1, 0) - src(0, 0) * src(1, 2)) * invDet,       // nolint
    (src(1, 0) * src(2, 1) - src(1, 1) * src(2, 0)) * invDet,       // nolint
    (src(0, 1) * src(2, 0) - src(0, 0) * src(2, 1)) * invDet,       // nolint
    (src(0, 0) * src(1, 1) - src(0, 1) * src(1, 0)) * invDet;
  return ret;
}

// (x, y, z) -> | 0, -z,  y |
//              | z,  0, -x |
//              |-y,  x,  0 |
SYCL_EXTERNAL inline Eigen::Matrix3f skew(const Eigen::Vector3f& x) {
  Eigen::Matrix3f ret;
  ret << 0.0f, -x[2], x[1],  // nolint
    x[2], 0.0f, -x[0],       // nolint
    -x[1], x[0], 0.0f;
  return ret;
}

SYCL_EXTERNAL inline Eigen::Matrix3f skew(const Eigen::Vector4f& x) {
  Eigen::Matrix3f ret;
  ret << 0.0f, -x[2], x[1],  // nolint
    x[2], 0.0f, -x[0],       // nolint
    -x[1], x[0], 0.0f;
  return ret;
}

/// @brief SO3 expmap.
/// @param omega  [rx, ry, rz]
/// @return       Quaternion
/// https://github.com/koide3/small_gicp/blob/master/include/small_gicp/util/lie.hpp
inline Eigen::Quaternionf so3_exp(const Eigen::Vector3f& omega) {
  const float theta_sq = omega.dot(omega);

  float imag_factor;
  float real_factor;
  if (theta_sq < 1e-10) {
    const float theta_quad = theta_sq * theta_sq;
    imag_factor = 0.5f - 1.0f / 48.0f * theta_sq + 1.0f / 3840.0f * theta_quad;
    real_factor = 1.0f - 1.0f / 8.0f * theta_sq + 1.0f / 384.0f * theta_quad;
  } else {
    const float theta = std::sqrt(theta_sq);
    const float half_theta = 0.5 * theta;
    imag_factor = std::sin(half_theta) / theta;
    real_factor = std::cos(half_theta);
  }

  return Eigen::Quaternionf(real_factor, imag_factor * omega.x(), imag_factor * omega.y(), imag_factor * omega.z());
}

// Rotation-first
/// @brief SE3 expmap (Rotation-first).
/// @param a  Twist vector [rx, ry, rz, tx, ty, tz]
/// @return   SE3 matrix
/// https://github.com/koide3/small_gicp/blob/master/include/small_gicp/util/lie.hpp
inline Eigen::Isometry3f se3_exp(const Eigen::Matrix<float, 6, 1>& a) {
  const Eigen::Vector3f omega = a.head<3>();

  const float theta_sq = omega.dot(omega);
  const float theta = std::sqrt(theta_sq);

  Eigen::Isometry3f se3 = Eigen::Isometry3f::Identity();
  se3.linear() = so3_exp(omega).toRotationMatrix();

  if (theta < 1e-10f) {
    se3.translation() = se3.linear() * a.tail<3>();
    /// Note: That is an accurate expansion!
  } else {
    const Eigen::Matrix3f Omega = skew(omega);
    const Eigen::Matrix3f V = (Eigen::Matrix3f::Identity() + (1.0f - std::cos(theta)) / theta_sq * Omega + (theta - std::sin(theta)) / (theta_sq * theta) * Omega * Omega);
    se3.translation() = V * a.tail<3>();
  }

  return se3;
}

}  // namespace eigen_utils
}  // namespace sycl_points
