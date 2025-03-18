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
template <typename T = float, int M, int N>
SYCL_EXTERNAL inline void matrixAdd(const Eigen::Matrix<T, M, N>& A, const Eigen::Matrix<T, M, N>& B, Eigen::Matrix<T, M, N>& C) {
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      C(i, j) = A(i, j) + B(i, j);
    }
  }
}

// A - B = C
// row: M, col: N
template <typename T = float, int M, int N>
SYCL_EXTERNAL inline void matrixSubtract(const Eigen::Matrix<T, M, N>& A, const Eigen::Matrix<T, M, N>& B, Eigen::Matrix<T, M, N>& C) {
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      C(i, j) = A(i, j) - B(i, j);
    }
  }
}

// A * B = C
// row: M, col: N
template <typename T = float, int M, int K, int N>
SYCL_EXTERNAL inline void matrixMultiply(const Eigen::Matrix<T, M, K>& A, const Eigen::Matrix<T, K, N>& B, Eigen::Matrix<T, M, N>& C) {
  C = Eigen::Matrix<T, M, N>::Zero();
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

// A * v = r
// row: M, col: N
template <typename T = float, int M, int N>
SYCL_EXTERNAL inline void matrixVectorMultiply(const Eigen::Matrix<T, M, N>& A, const Eigen::Vector<T, N>& v, Eigen::Vector<T, M>& r) {
  r = Eigen::Vector<T, M>::Zero();
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      r(i) += A(i, j) * v(j);
    }
  }
}

// A * s = B
// row: M, col: N
template <typename T = float, int M, int N>
SYCL_EXTERNAL inline void matrixScalarMultiply(const Eigen::Matrix<T, M, N>& A, float scalar, Eigen::Matrix<T, M, N>& B) {
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      B(i, j) = A(i, j) * scalar;
    }
  }
}

// A * s = B
template <typename T = float, int N>
SYCL_EXTERNAL inline void vectorScalarMultiply(const Eigen::Vector<T, N>& a, float scalar, Eigen::Vector<T, N>& b) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    b(i) = a(i) * scalar;
  }
}

// trans(A) = B
// row: M, col: N
template <typename T = float, int M, int N>
SYCL_EXTERNAL inline void matrixTranspose(const Eigen::Matrix<T, M, N>& A, Eigen::Matrix<T, N, M>& B) {
#pragma unroll
  for (int i = 0; i < M; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      B(j, i) = A(i, j);
    }
  }
}

// dot: u·v
template <typename T = float, int N>
SYCL_EXTERNAL inline float vectorDot(const Eigen::Vector<T, N>& u, const Eigen::Vector<T, N>& v) {
  float result = 0.0f;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    result += u(i, 0) * v(i, 0);
  }
  return result;
}

// cross: u × v = w
template <typename T = float>
SYCL_EXTERNAL inline void vector3Cross(const Eigen::Vector3<T>& u, const Eigen::Vector3<T>& v, Eigen::Vector3<T>& w) {
  w(0) = u(1) * v(2) - u(2) * v(1);
  w(1) = u(2) * v(0) - u(0) * v(2);
  w(2) = u(0) * v(1) - u(1) * v(0);
}

// (x, y, z) -> | 0, -z,  y |
//              | z,  0, -x |
//              |-y,  x,  0 |
template <typename T = float>
SYCL_EXTERNAL inline void vector3Skew(const Eigen::Vector3<T>& x, Eigen::Matrix3d& skewd) {
  skewd(0, 0) = (T)0.0;
  skewd(0, 1) = -x[2];
  skewd(0, 2) = x[1];
  skewd(1, 0) = x[2];
  skewd(1, 1) = (T)0.0;
  skewd(1, 2) = -x[0];
  skewd(2, 0) = -x[1];
  skewd(2, 1) = x[0];
  skewd(2, 2) = (T)0.0;
}
}  // namespace eigen_utils
}  // namespace sycl_points
