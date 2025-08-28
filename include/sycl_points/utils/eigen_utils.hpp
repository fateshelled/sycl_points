/**
 * @file eigen_utils.hpp
 * @brief Utility functions for Eigen matrix operations with SYCL compatibility
 *
 * This header provides SYCL-compatible implementations of common Eigen matrix operations,
 * including basic arithmetic, linear algebra operations, and special matrix utilities.
 * All functions marked with SYCL_EXTERNAL can be used in SYCL device code.
 */

#pragma once

#include <Eigen/Dense>
#include <sycl_points/utils/sycl_utils.hpp>

namespace sycl_points {

/// @brief Utility functions for Eigen matrix operations with SYCL compatibility
/// @note Eigen::Matrix is column major
namespace eigen_utils {

/// @brief PI
constexpr float PI = 3.14159265358979323846f;

/// @brief Computes the element-wise addition of two matrices.
/// @tparam M number of rows
/// @tparam N number of cols
/// @param A 1st matrix
/// @param B 2nd matrix
/// @return Result of A + B
template <size_t M, size_t N>
SYCL_EXTERNAL Eigen::Matrix<float, M, N> add(const Eigen::Matrix<float, M, N>& A, const Eigen::Matrix<float, M, N>& B) {
    Eigen::Matrix<float, M, N> ret;
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
#pragma unroll M
        for (size_t i = 0; i < M; ++i) {
            ret(i, j) = A(i, j) + B(i, j);
        }
    }
    return ret;
}

template <size_t M, size_t N>
/// @brief In-place Matrix Addition
/// @tparam M number of rows
/// @tparam N number of cols
/// @param A Matrix to be modified (A += B)
/// @param B Matrix to add
SYCL_EXTERNAL void add_inplace(Eigen::Matrix<float, M, N>& A, const Eigen::Matrix<float, M, N>& B) {
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
#pragma unroll M
        for (size_t i = 0; i < M; ++i) {
            A(i, j) += B(i, j);
        }
    }
}

/// @brief Matrix Subtraction
/// @tparam M number of rows
/// @tparam N number of cols
/// @param A 1st matrix
/// @param B 2nd matrix
/// @return Result of A - B
template <size_t M, size_t N>
SYCL_EXTERNAL Eigen::Matrix<float, M, N> subtract(const Eigen::Matrix<float, M, N>& A,
                                                  const Eigen::Matrix<float, M, N>& B) {
    Eigen::Matrix<float, M, N> ret;
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
#pragma unroll M
        for (size_t i = 0; i < M; ++i) {
            ret(i, j) = A(i, j) - B(i, j);
        }
    }
    return ret;
}

/// @brief Matrix-Matrix Multiplication (dot product)
/// @tparam M Number of rows in result matrix
/// @tparam K Shared dimension (columns of A, rows of B)
/// @tparam N Number of columns in result matrix
/// @param A  First matrix (M×K)
/// @param B Second matrix (K×N)
/// @return Result of A × B
template <size_t M, size_t K, size_t N>
SYCL_EXTERNAL Eigen::Matrix<float, M, N> multiply(const Eigen::Matrix<float, M, K>& A,
                                                  const Eigen::Matrix<float, K, N>& B) {
    Eigen::Matrix<float, M, N> ret = Eigen::Matrix<float, M, N>::Zero();
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
#pragma unroll K
        for (size_t k = 0; k < K; ++k) {
            const auto b_kj = B(k, j);
#pragma unroll M
            for (size_t i = 0; i < M; ++i) {
                // ret(i, j) += A(i, k) * B(k, j);
                ret(i, j) = sycl::fma(A(i, k), b_kj, ret(i, j));
            }
        }
    }
    return ret;
}

/// @brief Matrix-Vector Multiplication
/// @tparam M Number of rows in matrix
/// @tparam N Number of columns in matrix
/// @param A Matrix (MxN)
/// @param v Vector (Nx1)
/// @return Result of A × v
template <size_t M, size_t N>
SYCL_EXTERNAL Eigen::Vector<float, M> multiply(const Eigen::Matrix<float, M, N>& A, const Eigen::Vector<float, N>& v) {
    Eigen::Vector<float, M> ret = Eigen::Vector<float, M>::Zero();
#pragma unroll M
    for (size_t i = 0; i < M; ++i) {
        float sum = 0.0f;
#pragma unroll N
        for (size_t j = 0; j < N; ++j) {
            // ret(i) += A(i, j) * v(j);
            sum = sycl::fma(A(i, j), v(j), sum);
        }
        ret(i) = sum;
    }
    return ret;
}

/// @brief Scalar Matrix Multiplication
/// @tparam M Number of rows
/// @tparam N Number of columns
/// @param A Matrix
/// @param scalar Scalar valur
/// @return Result of A x scalar
template <size_t M, size_t N>
SYCL_EXTERNAL Eigen::Matrix<float, M, N> multiply(const Eigen::Matrix<float, M, N>& A, float scalar) {
    Eigen::Matrix<float, M, N> ret;
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
#pragma unroll M
        for (size_t i = 0; i < M; ++i) {
            ret(i, j) = A(i, j) * scalar;
        }
    }
    return ret;
}

/// @brief Scalar Vector Multiplication
/// @tparam N Vector size
/// @param a Vector
/// @param scalar Scalar value
/// @return Result of a x scalar
template <size_t N>
SYCL_EXTERNAL Eigen::Vector<float, N> multiply(const Eigen::Vector<float, N>& a, float scalar) {
    Eigen::Vector<float, N> ret;
#pragma unroll N
    for (size_t i = 0; i < N; ++i) {
        ret(i) = a(i) * scalar;
    }
    return ret;
}

template <size_t M, size_t N>
/// @brief In-place Scalar Matrix Multiplication
/// @tparam M Number of rows
/// @tparam N Number of columns
/// @param A Matrix to be modified  (A *= scalar)
/// @param scalar Scalar value
SYCL_EXTERNAL void multiply_inplace(Eigen::Matrix<float, M, N>& A, float scalar) {
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
#pragma unroll M
        for (size_t i = 0; i < M; ++i) {
            A(i, j) *= scalar;
        }
    }
}

/// @brief Element-wise Matrix Multiplication
/// @tparam M Number of rows
/// @tparam N Number of columns
/// @param A Matrix A
/// @param B Matrix B
/// @return Matrix A x B
template <size_t M, size_t N>
SYCL_EXTERNAL Eigen::Matrix<float, M, N> element_wise_multiply(const Eigen::Matrix<float, M, N>& A,
                                                               const Eigen::Matrix<float, M, N>& B) {
    Eigen::Matrix<float, M, N> ret;
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
#pragma unroll M
        for (size_t i = 0; i < M; ++i) {
            ret(i, j) = A(i, j) * B(i, j);
        }
    }
    return ret;
}

/// @brief Ensure Matrix Symmetry
///
/// Makes a matrix symmetric by averaging corresponding off-diagonal elements.
/// Useful for numerical stability when the matrix should be symmetric but
/// rounding errors have introduced asymmetry.
///
/// @tparam M Matrix dimension (must be square)
/// @param A Square matrix to symmetrize
/// @return Symmetric version of A
template <size_t M>
SYCL_EXTERNAL Eigen::Matrix<float, M, M> ensure_symmetric(const Eigen::Matrix<float, M, M>& A) {
    Eigen::Matrix<float, M, M> ret;
#pragma unroll M
    for (size_t j = 0; j < M; ++j) {
#pragma unroll M
        for (size_t i = 0; i < M; ++i) {
            ret(i, j) = (i == j) ? A(i, j) : (A(i, j) + A(j, i)) * 0.5f;
        }
    }
    return ret;
}

/// @brief Matrix Transpose
/// @tparam M M Number of rows in original matrix
/// @tparam N Number of columns in original matrix
/// @param A Matrix to transpose (MxN)
/// @return Transposed matrix (NxM)
template <size_t M, size_t N>
SYCL_EXTERNAL Eigen::Matrix<float, N, M> transpose(const Eigen::Matrix<float, M, N>& A) {
    Eigen::Matrix<float, N, M> ret;
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
#pragma unroll M
        for (size_t i = 0; i < M; ++i) {
            ret(j, i) = A(i, j);
        }
    }
    return ret;
}

template <size_t N>
/// @brief Vector Dot Product
/// @tparam N Vector size
/// @param u 1st vector
/// @param v 2nd vector
/// @return Result of dot product
SYCL_EXTERNAL float dot(const Eigen::Vector<float, N>& u, const Eigen::Vector<float, N>& v) {
    float result = 0.0f;
#pragma unroll N
    for (size_t i = 0; i < N; ++i) {
        // result += u(i) * v(i);
        result = sycl::fma(u(i), v(i), result);
    }
    return result;
}

/// @brief Vector Cross Product
/// @param u 1st vector
/// @param v 2nd vector
/// @return Cross product result u × v
SYCL_EXTERNAL inline Eigen::Vector3f cross(const Eigen::Vector3f& u, const Eigen::Vector3f& v) {
    Eigen::Vector3f ret;
    ret(0) = u(1) * v(2) - u(2) * v(1);
    ret(1) = u(2) * v(0) - u(0) * v(2);
    ret(2) = u(0) * v(1) - u(1) * v(0);
    return ret;
}

/// @brief Vector Outer Product
/// @param u 1st vector
/// @param v 2nd vector
/// @return Outer product result u ⊗ v
SYCL_EXTERNAL inline Eigen::Matrix4f outer(const Eigen::Vector4f& u, const Eigen::Vector4f& v) {
    Eigen::Matrix4f ret;
#pragma unroll 4
    for (size_t j = 0; j < 4; ++j) {
        const float v_j = v(j);
#pragma unroll 4
        for (size_t i = 0; i < 4; ++i) {
            ret(i, j) = u(i) * v_j;
        }
    }
    return ret;
}

/// @brief Extract 3×3 Block from 4×4 Matrix
/// @param src Source 4×4 matrix
/// @return Upper-left 3×3 submatrix
SYCL_EXTERNAL inline Eigen::Matrix3f block3x3(const Eigen::Matrix4f& src) {
    Eigen::Matrix3f ret;
#pragma unroll 3
    for (size_t i = 0; i < 3; ++i) {
#pragma unroll 3
        for (size_t j = 0; j < 3; ++j) {
            ret(i, j) = src(i, j);
        }
    }
    return ret;
}

template <size_t M>
/// @brief Matrix Trace
/// @tparam M Matrix dimension
/// @param A Square matrix
/// @return Trace of A
SYCL_EXTERNAL float trace(const Eigen::Matrix<float, M, M>& A) {
    float ret = 0.0f;
#pragma unroll M
    for (size_t i = 0; i < M; ++i) {
        ret += A(i, i);
    }
    return ret;
}

/// @brief 3×3 Matrix Determinant
/// @param A 3×3 matrix
/// @return Determinant of A
SYCL_EXTERNAL inline float determinant(const Eigen::Matrix3f& A) {
    return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
           A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
}

template <size_t M, size_t N>
/// @brief Matrix Frobenius Norm
/// @tparam M Number of rows
/// @tparam N Number of columns
/// @param A Matrix
/// @return Frobenius norm of A
SYCL_EXTERNAL float frobenius_norm(const Eigen::Matrix<float, M, N>& A) {
    float ret = 0.0f;
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
#pragma unroll M
        for (size_t i = 0; i < M; ++i) {
            // ret += A(i, j) * A(i, j);
            ret = sycl::fma(A(i, j), A(i, j), ret);
        }
    }
    return sycl::sqrt(ret);
}

template <size_t M>
/// @brief Vector Frobenius Norm squared
/// @tparam M Vector size
/// @param a vector
/// @return Frobenius norm squared of a
SYCL_EXTERNAL float frobenius_norm_squared(const Eigen::Vector<float, M>& a) {
    float ret = 0.0f;
#pragma unroll M
    for (size_t i = 0; i < M; ++i) {
        // ret += a(i) * a(i);
        ret = sycl::fma(a(i), a(i), ret);
    }
    return ret;
}

template <size_t M>
/// @brief Vector Frobenius Norm
/// @tparam M Vector size
/// @param a vector
/// @return Frobenius norm of a
SYCL_EXTERNAL float frobenius_norm(const Eigen::Vector<float, M>& a) {
    return sycl::sqrt(frobenius_norm_squared<M>(a));
}

/// @brief 3×3 Matrix Inverse
/// @param src 3×3 matrix to invert
/// @return Inverse of src
SYCL_EXTERNAL inline Eigen::Matrix3f inverse(const Eigen::Matrix3f& src) {
    const float det = determinant(src);

    if (sycl::fabs(det) < 1e-6f) {
        return Eigen::Matrix3f::Zero();
    }

    const float invDet = 1.0f / det;

    Eigen::Matrix3f ret;
    ret(0, 0) = (src(1, 1) * src(2, 2) - src(1, 2) * src(2, 1)) * invDet;
    ret(1, 0) = (src(1, 2) * src(2, 0) - src(1, 0) * src(2, 2)) * invDet;
    ret(2, 0) = (src(1, 0) * src(2, 1) - src(1, 1) * src(2, 0)) * invDet;
    ret(0, 1) = (src(0, 2) * src(2, 1) - src(0, 1) * src(2, 2)) * invDet;
    ret(1, 1) = (src(0, 0) * src(2, 2) - src(0, 2) * src(2, 0)) * invDet;
    ret(2, 1) = (src(0, 1) * src(2, 0) - src(0, 0) * src(2, 1)) * invDet;
    ret(0, 2) = (src(0, 1) * src(1, 2) - src(0, 2) * src(1, 1)) * invDet;
    ret(1, 2) = (src(0, 2) * src(1, 0) - src(0, 0) * src(1, 2)) * invDet;
    ret(2, 2) = (src(0, 0) * src(1, 1) - src(0, 1) * src(1, 0)) * invDet;
    return ret;
}

/// @brief Create Diagonal Matrix
/// @tparam M Dimension of the diagonal matrix
/// @param diag Vector of diagonal elements
/// @return Diagonal matrix
template <size_t M>
SYCL_EXTERNAL Eigen::Matrix3f as_diagonal(const Eigen::Vector<float, M>& diag) {
    Eigen::Matrix<float, M, M> ret = Eigen::Matrix<float, M, M>::Zero();
#pragma unroll M
    for (size_t i = 0; i < M; ++i) {
        ret(i, i) = diag(i);
    }
    return ret;
}

/// @brief Symmetric Eigen Decomposition for 3×3 Matrix
/// @param A Symmetric 3×3 matrix
/// @param eigenvalues eigen values (sorted in ascending order)
/// @param eigenvectors eigen vectors
SYCL_EXTERNAL inline void symmetric_eigen_decomposition_3x3(const Eigen::Matrix3f& A, Eigen::Vector3f& eigenvalues,
                                                            Eigen::Matrix3f& eigenvectors) {
    constexpr float EPSILON = 1e-7f;

    // Characteristic polynomial
    // det(A - λI) = -λ^3 + c2*λ^2 + c1*λ + c0
    const float c2 = -trace<3>(A);
    const float c1 = (A(0, 0) * A(1, 1) + A(0, 0) * A(2, 2) + A(1, 1) * A(2, 2)) -
                     (A(0, 1) * A(1, 0) + A(0, 2) * A(2, 0) + A(1, 2) * A(2, 1));
    const float c0 = -determinant(A);

    const float p = c1 - c2 * c2 / 3.0f;
    const float q = 2.0f * c2 * c2 * c2 / 27.0f - c2 * c1 / 3.0f + c0;
    const float discriminant = 4.0f * p * p * p + 27.0f * q * q;

    // compute eigenvalues
    if (sycl::fabs(discriminant) <= EPSILON) {
        const float u = q >= 0 ? -sycl::cbrt(q / 2.0f) : sycl::cbrt(-q / 2.0f);
        eigenvalues(0) = 2.0f * u - c2 / 3.0f;
        eigenvalues(1) = eigenvalues(2) = -u - c2 / 3.0f;
    } else {
        // Cardano's formula
        const float sqrt_neg_p_over_3 = sycl::sqrt(-p / 3.0f);
        const float cos =
            sycl::max(-1.0f, sycl::min(1.0f, -q / (2.0f * sqrt_neg_p_over_3 * sqrt_neg_p_over_3 * sqrt_neg_p_over_3)));
        float phi = sycl::fabs(p) < EPSILON ? 0.0f : sycl::acos(cos);
        if (phi < 0.0f) phi += PI;

        // compute
        eigenvalues(0) = 2.0f * sqrt_neg_p_over_3 * sycl::cos(phi / 3.0f) - c2 / 3.0f;
        eigenvalues(2) = 2.0f * sqrt_neg_p_over_3 * sycl::cos((phi + 4.0f * PI) / 3.0f) - c2 / 3.0f;
        eigenvalues(1) = 2.0f * sqrt_neg_p_over_3 * sycl::cos((phi + 2.0f * PI) / 3.0f) - c2 / 3.0f;
    }
    // sort
    if (eigenvalues(0) > eigenvalues(1)) {
        std::swap(eigenvalues(0), eigenvalues(1));
    }
    if (eigenvalues(1) > eigenvalues(2)) {
        std::swap(eigenvalues(1), eigenvalues(2));
    }
    if (eigenvalues(0) > eigenvalues(1)) {
        std::swap(eigenvalues(1), eigenvalues(0));
    }

    // compute eigenvectors
    eigenvectors = Eigen::Matrix3f::Zero();

#pragma unroll 3
    for (size_t k = 0; k < 3; ++k) {
        // solve (A - λI)x = 0
        const Eigen::Matrix3f M = subtract<3, 3>(A, multiply<3, 3>(Eigen::Matrix3f::Identity(), eigenvalues(k)));

        // compute det
        const float m00 = M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1);
        const float m01 = M(1, 2) * M(2, 0) - M(1, 0) * M(2, 2);
        const float m02 = M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0);

        const float m10 = M(0, 2) * M(2, 1) - M(0, 1) * M(2, 2);
        const float m11 = M(0, 0) * M(2, 2) - M(0, 2) * M(2, 0);
        const float m12 = M(0, 1) * M(2, 0) - M(0, 0) * M(2, 1);

        const float m20 = M(0, 1) * M(1, 2) - M(0, 2) * M(1, 1);
        const float m21 = M(0, 2) * M(1, 0) - M(0, 0) * M(1, 2);
        const float m22 = M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);

        const float s0 = m00 * m00 + m10 * m10 + m20 * m20;
        const float s1 = m01 * m01 + m11 * m11 + m21 * m21;
        const float s2 = m02 * m02 + m12 * m12 + m22 * m22;

        sycl::float3 v;
        if (s0 >= s1 && s0 >= s2) {
            v[0] = m00;
            v[1] = m10;
            v[2] = m20;
        } else if (s1 >= s0 && s1 >= s2) {
            v[0] = m01;
            v[1] = m11;
            v[2] = m21;
        } else {
            v[0] = m02;
            v[1] = m12;
            v[2] = m22;
        }
        // normalize
        const float inv_length = 1.0f / sycl::sqrt(sycl::dot(v, v));
        eigenvectors(0, k) = v[0] * inv_length;
        eigenvectors(1, k) = v[1] * inv_length;
        eigenvectors(2, k) = v[2] * inv_length;
    }
}

/// @brief Solve 6×6 Linear System
///
/// Solves the linear system Ax = b for a 6×6 matrix using Cholesky decomposition.
///
/// @param A 6×6 coefficient matrix
/// @param b 6×1 right-hand side vector
/// @return Solution vector x
SYCL_EXTERNAL inline Eigen::Matrix<float, 6, 1> solve_system_6x6(const Eigen::Matrix<float, 6, 6>& A,
                                                                 const Eigen::Matrix<float, 6, 1>& b) {
    // Cholesky decomposition
    Eigen::Matrix<float, 6, 6> L = Eigen::Matrix<float, 6, 6>::Zero();
    constexpr float EPSILON = 1e-7f;
    L(0, 0) = sycl::sqrt(A(0, 0) > EPSILON ? A(0, 0) : EPSILON);
    const float inv_L00 = 1.0f / L(0, 0);

    for (size_t i = 1; i < 6; ++i) {
        L(i, 0) = A(i, 0) * inv_L00;
    }

    for (size_t j = 1; j < 6; ++j) {
        float diag_sum = 0.0f;
        for (size_t k = 0; k < j; ++k) {
            // diag_sum += L(j, k) * L(j, k);
            diag_sum = sycl::fma(L(j, k), L(j, k), diag_sum);
        }
        const float val = A(j, j) - diag_sum;
        L(j, j) = sycl::sqrt(val > EPSILON ? val : EPSILON);
        const float inv_Ljj = 1.0f / L(j, j);
        for (size_t i = j + 1; i < 6; ++i) {
            float off_diag_sum = 0.0f;
            for (size_t k = 0; k < j; ++k) {
                // off_diag_sum += L(i, k) * L(j, k);
                off_diag_sum = sycl::fma(L(i, k), L(j, k), off_diag_sum);
            }

            L(i, j) = (A(i, j) - off_diag_sum) * inv_Ljj;
        }
    }

    Eigen::Matrix<float, 6, 1> y = Eigen::Matrix<float, 6, 1>::Zero();
    for (size_t i = 0; i < 6; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < i; ++j) {
            // sum += L(i, j) * y(j);
            sum = sycl::fma(L(i, j), y(j), sum);
        }
        const float inv_Lii = 1.0f / (sycl::fabs(L(i, i)) < EPSILON ? EPSILON : L(i, i));
        y(i) = (b(i) - sum) * inv_Lii;
    }

    Eigen::Matrix<float, 6, 1> x = Eigen::Matrix<float, 6, 1>::Zero();
    for (int32_t i = 5; i >= 0; --i) {
        float sum = 0.0f;
        for (size_t j = i + 1; j < 6; ++j) {
            // sum += L(j, i) * x(j);
            sum = sycl::fma(L(j, i), x(j), sum);
        }

        const float inv_Lii = 1.0f / (sycl::fabs(L(i, i)) < EPSILON ? EPSILON : L(i, i));
        x(i) = (y(i) - sum) * inv_Lii;
    }
    return x;
}

template <size_t M, size_t N>
/// @brief copy matrix
/// @param src source matrix
/// @param dst destination matrix
SYCL_EXTERNAL void copy(const Eigen::Matrix<float, M, N>& src, Eigen::Matrix<float, M, N>& dst) {
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
#pragma unroll M
        for (size_t i = 0; i < M; ++i) {
            dst(i, j) = src(i, j);
        }
    }
}

template <size_t M, size_t N>
/// @brief swap matrix
/// @param src source matrix
/// @param dst destination matrix
SYCL_EXTERNAL void swap(Eigen::Matrix<float, M, N>& src, Eigen::Matrix<float, M, N>& dst) {
#pragma unroll N
    for (size_t j = 0; j < N; ++j) {
#pragma unroll M
        for (size_t i = 0; i < M; ++i) {
            std::swap(src(i, j), dst(i, j));
        }
    }
}

/// @brief Convert Eigen::Vector4f to sycl vector
/// @param vec Eigen vector
/// @return sycl vector
inline sycl::float4 to_sycl_vec(const Eigen::Vector4f& vec) { return {vec[0], vec[1], vec[2], vec[3]}; }

/// @brief Convert Eigen::Matrix4f to sycl vectors
/// @param mat Eigen matrix
/// @return sycl vectors
inline std::array<sycl::float4, 4> to_sycl_vec(const Eigen::Matrix4f& mat) {
    std::array<sycl::float4, 4> vecs;
#pragma unroll 4
    for (size_t i = 0; i < 4; ++i) {
        vecs[i] = sycl::float4(mat(i, 0), mat(i, 1), mat(i, 2), mat(i, 3));
    }
    return vecs;
}

/// @brief Convert Eigen::Matrix<float, 6, 6> to sycl vector
/// @param mat Eigen matrix
/// @return sycl vectors
inline std::tuple<sycl::float16, sycl::float16, sycl::float4> to_sycl_vec(const Eigen::Matrix<float, 6, 6>& mat) {
    const sycl::float16 a = {mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3), mat(0, 4), mat(0, 5), mat(1, 0), mat(1, 1),
                             mat(1, 2), mat(1, 3), mat(1, 4), mat(1, 5), mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3)};
    const sycl::float16 b = {mat(2, 4), mat(2, 5), mat(3, 0), mat(3, 1), mat(3, 2), mat(3, 3), mat(3, 4), mat(3, 5),
                             mat(4, 0), mat(4, 1), mat(4, 2), mat(4, 3), mat(4, 4), mat(4, 5), mat(5, 0), mat(5, 1)};
    const sycl::float4 c = {mat(5, 2), mat(5, 3), mat(5, 4), mat(5, 5)};
    return {a, b, c};
}

/// @brief Convert Eigen::Vector<float, 6> to sycl vector
/// @param vec Eigen vector
/// @return sycl vectors
inline std::array<sycl::float3, 2> to_sycl_vec(const Eigen::Vector<float, 6>& vec) {
    const sycl::float3 a = {vec(0), vec(1), vec(2)};
    const sycl::float3 b = {vec(3), vec(4), vec(5)};
    return {a, b};
}

/// @brief Convert sycl vector to Eigen::Vector4f
/// @param vec sycl vector
/// @return Eigen vector
inline Eigen::Vector4f from_sycl_vec(const sycl::float4& vec) { return {vec.x(), vec.y(), vec.z(), vec.w()}; }

/// @brief Convert sycl vector to Eigen::Matrix4f
/// @param vec sycl vectors
/// @return Eigen matrix
inline Eigen::Matrix4f from_sycl_vec(const std::array<sycl::float4, 4>& vecs) {
    Eigen::Matrix4f mat;
#pragma unroll 4
    for (size_t j = 0; j < 4; ++j) {
#pragma unroll 4
        for (size_t i = 0; i < 4; ++i) {
            mat(i, j) = vecs[i][j];
        }
    }
    return mat;
}

/// @brief Convert sycl vector to Eigen::Matrix<float, 6, 6>
/// @param vec sycl vectors
/// @return Eigen matrix
inline Eigen::Matrix<float, 6, 6> from_sycl_vec(const std::tuple<sycl::float16, sycl::float16, sycl::float4>& vecs) {
    Eigen::Matrix<float, 6, 6> mat;
    const auto& [a, b, c] = vecs;
    mat << a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15],  //
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15],     //
        c[0], c[1], c[2], c[3];
    return mat;
}

/// @brief Convert sycl vector to Eigen::Vector<float, 6>
/// @param vec sycl vectors
/// @return Eigen vector
inline Eigen::Vector<float, 6> from_sycl_vec(const std::array<sycl::float3, 2>& vecs) {
    Eigen::Vector<float, 6> ret;
    ret << vecs[0][0], vecs[0][1], vecs[0][2], vecs[1][0], vecs[1][1], vecs[1][2];
    return ret;
}

/// @brief Lie algebra and Lie group operations
namespace lie {
/// @brief Create skew-symmetric matrix
/// @details (x, y, z) -> | 0, -z,  y |
///                       | z,  0, -x |
///                       |-y,  x,  0 |
/// @param x vector
/// @return skew-symmetric matrix
SYCL_EXTERNAL inline Eigen::Matrix3f skew(const Eigen::Vector3f& x) {
    Eigen::Matrix3f ret;
    ret << 0.0f, -x[2], x[1],  // nolint
        x[2], 0.0f, -x[0],     // nolint
        -x[1], x[0], 0.0f;
    return ret;
}

/// @brief Create skew-symmetric matrix
/// @details (x, y, z) -> | 0, -z,  y |
///                       | z,  0, -x |
///                       |-y,  x,  0 |
/// @param x vector
/// @return skew-symmetric matrix
SYCL_EXTERNAL inline Eigen::Matrix3f skew(const Eigen::Vector4f& x) {
    Eigen::Matrix3f ret;
    ret << 0.0f, -x[2], x[1],  // nolint
        x[2], 0.0f, -x[0],     // nolint
        -x[1], x[0], 0.0f;
    return ret;
}

/// @brief SO3 expmap.
/// @param omega Rotation vector [rx, ry, rz]
/// @return Quaternion
/// https://github.com/koide3/small_gicp/blob/master/include/small_gicp/util/lie.hpp
inline Eigen::Quaternionf so3_exp(const Eigen::Vector3f& omega) {
    const float theta_sq = omega.dot(omega);

    float imag_factor;
    float real_factor;
    if (theta_sq < 1e-6f) {
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
/// @param a Twist vector [rx, ry, rz, tx, ty, tz]
/// @return SE3 matrix
/// https://github.com/koide3/small_gicp/blob/master/include/small_gicp/util/lie.hpp
inline Eigen::Isometry3f se3_exp(const Eigen::Matrix<float, 6, 1>& a) {
    const Eigen::Vector3f omega = a.head<3>();

    const float theta_sq = omega.dot(omega);
    const float theta = std::sqrt(theta_sq);

    Eigen::Isometry3f se3 = Eigen::Isometry3f::Identity();
    se3.linear() = so3_exp(omega).toRotationMatrix();

    if (theta < 1e-6f) {
        se3.translation() = se3.linear() * a.tail<3>();
        /// Note: That is an accurate expansion!
    } else {
        const Eigen::Matrix3f Omega = skew(omega);
        const Eigen::Matrix3f V = (Eigen::Matrix3f::Identity() + (1.0f - std::cos(theta)) / theta_sq * Omega +
                                   (theta - std::sin(theta)) / (theta_sq * theta) * Omega * Omega);
        se3.translation() = V * a.tail<3>();
    }

    return se3;
}

/// @brief SO3 logmap.
/// @param quat Quaternion
/// @return Rotation vector [rx, ry, rz]
inline Eigen::Vector3f so3_log(const Eigen::Quaternionf& quat) {
    // Normalize quaternion to ensure unit quaternion
    Eigen::Quaternionf q = quat.normalized();

    // Ensure w >= 0 for canonical representation (shortest rotation)
    if (q.w() < 0.0f) {
        q.coeffs() *= -1.0f;
    }

    const float w = q.w();
    const Eigen::Vector3f xyz(q.x(), q.y(), q.z());

    // Calculate the magnitude of the imaginary part
    const float xyz_norm = xyz.norm();

    // Handle small angle case (near identity)
    if (xyz_norm < 1e-6f) {
        // For small angles: omega ≈ 2 * xyz / w
        // Using Taylor expansion to avoid numerical issues
        const float scale = 2.0f / w * (1.0f + xyz_norm * xyz_norm / (6.0f * w * w));
        return scale * xyz;
    }

    // Handle large angle case (near 180 degrees)
    if (sycl::fabs(w) < 1e-6f) {
        // For angles near π: theta ≈ π, axis = xyz / ||xyz||
        const float theta = PI;
        return (theta / xyz_norm) * xyz;
    }

    // General case
    const float theta = 2.0f * sycl::atan2(xyz_norm, sycl::fabs(w));
    const float scale = theta / xyz_norm;

    return scale * xyz;
}

/// @brief SE3 logmap (Rotation-first).
/// @param transform SE3 transformation matrix
/// @return Twist vector [rx, ry, rz, tx, ty, tz]
inline Eigen::Matrix<float, 6, 1> se3_log(const Eigen::Isometry3f& transform) {
    // Extract rotation and translation
    const Eigen::Matrix3f R = transform.linear();
    const Eigen::Vector3f t = transform.translation();

    // Convert rotation matrix to quaternion and then to rotation vector
    const Eigen::Quaternionf quat(R);
    const Eigen::Vector3f omega = so3_log(quat);

    const float theta = omega.norm();

    Eigen::Matrix<float, 6, 1> result;
    result.head<3>() = omega;

    // Handle small angle case
    if (theta < 1e-6f) {
        // For small angles: V^-1 ≈ I - 0.5 * skew(omega)
        const Eigen::Matrix3f V_inv = Eigen::Matrix3f::Identity() - 0.5f * skew(omega);
        result.tail<3>() = V_inv * t;
        return result;
    }

    // General case: compute V^-1
    const float half_theta = 0.5f * theta;
    const float sin_half_theta = sycl::sin(half_theta);
    const float cos_half_theta = sycl::cos(half_theta);

    // V^-1 = I - 0.5 * Omega + (1/theta^2) * (1 - theta*sin_half_theta/(2*sin_half_theta^2)) * Omega^2
    // Simplified: V^-1 = I - 0.5 * Omega + coeff * Omega^2
    const float coeff = (1.0f - theta * cos_half_theta / (2.0f * sin_half_theta)) / (theta * theta);

    const Eigen::Matrix3f Omega = skew(omega);
    const Eigen::Matrix3f V_inv = Eigen::Matrix3f::Identity() - 0.5f * Omega + coeff * Omega * Omega;

    result.tail<3>() = V_inv * t;

    return result;
}

}  // namespace lie
}  // namespace eigen_utils
}  // namespace sycl_points
