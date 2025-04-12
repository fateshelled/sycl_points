#pragma once

#include <Eigen/Dense>
#include <sycl_points/utils/sycl_utils.hpp>


namespace sycl_points {
namespace eigen_utils {
// Eigen::Matrix is column major

// A + B = C
// row: M, col: N
template <size_t M, size_t N>
SYCL_EXTERNAL inline Eigen::Matrix<float, M, N> add(const Eigen::Matrix<float, M, N>& A,
                                                    const Eigen::Matrix<float, M, N>& B) {
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

// A += B
// row: M, col: N
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
SYCL_EXTERNAL inline Eigen::Matrix<float, M, N> subtract(const Eigen::Matrix<float, M, N>& A,
                                                         const Eigen::Matrix<float, M, N>& B) {
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
SYCL_EXTERNAL inline Eigen::Matrix<float, M, N> multiply(const Eigen::Matrix<float, M, K>& A,
                                                         const Eigen::Matrix<float, K, N>& B) {
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
SYCL_EXTERNAL inline Eigen::Vector<float, M> multiply(const Eigen::Matrix<float, M, N>& A,
                                                      const Eigen::Vector<float, N>& v) {
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

// A *= s
// row: M, col: N
template <size_t M, size_t N>
SYCL_EXTERNAL inline Eigen::Vector<float, N> multiply_zerocopy(Eigen::Matrix<float, M, N>& A, float scalar) {
#pragma unroll
    for (size_t i = 0; i < M; ++i) {
#pragma unroll
        for (size_t j = 0; j < N; ++j) {
            A(i, j) *= scalar;
        }
    }
}

template <size_t M>
SYCL_EXTERNAL inline Eigen::Matrix<float, M, M> ensure_symmetric(const Eigen::Matrix<float, M, M>& A) {
    Eigen::Matrix<float, M, M> ret;
#pragma unroll
    for (size_t i = 0; i < M; ++i) {
#pragma unroll
        for (size_t j = 0; j < M; ++j) {
            if (i == j) {
                ret(i, j) = A(i, j);
            } else {
                ret(i, j) = (A(i, j) + A(j, i)) * 0.5f;
            }
        }
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
        u(2) * v(0) - u(0) * v(2),     // nolint
        u(0) * v(1) - u(1) * v(0);
    return ret;
}

// outer: u ⊗ v = w
SYCL_EXTERNAL inline Eigen::Matrix4f outer(const Eigen::Vector4f& u, const Eigen::Vector4f& v) {
    Eigen::Matrix4f ret;
    ret << u(0) * v(0), u(0) * v(1), u(0) * v(2), u(0) * v(3),  // nolint
        u(1) * v(0), u(1) * v(1), u(1) * v(2), u(1) * v(3),     // nolint
        u(2) * v(0), u(2) * v(1), u(2) * v(2), u(2) * v(3),     // nolint
        u(3) * v(0), u(3) * v(1), u(3) * v(2), u(3) * v(3);
    return ret;
}

SYCL_EXTERNAL inline Eigen::Matrix3f block3x3(const Eigen::Matrix4f& src) {
    Eigen::Matrix3f ret;
    ret << src(0, 0), src(0, 1), src(0, 2),  // nolint
        src(1, 0), src(1, 1), src(1, 2),     // nolint
        src(2, 0), src(2, 1), src(2, 2);
    return ret;
}

template <size_t M>
SYCL_EXTERNAL inline float trace(const Eigen::Matrix<float, M, M>& A) {
    float ret = 0.0f;
#pragma unroll
    for (size_t i = 0; i < M; ++i) {
        ret += A(i, i);
    }
    return ret;
}

SYCL_EXTERNAL inline float determinant(const Eigen::Matrix3f& A) {
    return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
           A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
}

template <size_t M, size_t N>
SYCL_EXTERNAL inline float frobenius_norm(const Eigen::Matrix<float, M, N>& A) {
    float ret = 0.0f;
#pragma unroll
    for (size_t i = 0; i < M; ++i) {
#pragma unroll
        for (size_t j = 0; j < N; ++j) {
            ret += A(i, j) * A(i, j);
        }
    }
    return sycl::sqrt(ret);
}

SYCL_EXTERNAL inline Eigen::Matrix3f inverse(const Eigen::Matrix3f& src) {
    const float det = determinant(src);

    if (sycl::fabs(det) < 1e-6f) {
        return Eigen::Matrix3f::Zero();
    }

    const float invDet = 1.0f / det;

    Eigen::Matrix3f ret;
    ret << (src(1, 1) * src(2, 2) - src(1, 2) * src(2, 1)) * invDet,  // nolint
        (src(0, 2) * src(2, 1) - src(0, 1) * src(2, 2)) * invDet,     // nolint
        (src(0, 1) * src(1, 2) - src(0, 2) * src(1, 1)) * invDet,     // nolint
        (src(1, 2) * src(2, 0) - src(1, 0) * src(2, 2)) * invDet,     // nolint
        (src(0, 0) * src(2, 2) - src(0, 2) * src(2, 0)) * invDet,     // nolint
        (src(0, 2) * src(1, 0) - src(0, 0) * src(1, 2)) * invDet,     // nolint
        (src(1, 0) * src(2, 1) - src(1, 1) * src(2, 0)) * invDet,     // nolint
        (src(0, 1) * src(2, 0) - src(0, 0) * src(2, 1)) * invDet,     // nolint
        (src(0, 0) * src(1, 1) - src(0, 1) * src(1, 0)) * invDet;
    return ret;
}

template <size_t M>
SYCL_EXTERNAL inline Eigen::Matrix3f as_diagonal(const Eigen::Vector<float, M>& diag) {
    Eigen::Matrix<float, M, M> ret = Eigen::Matrix<float, M, M>::Zero();
#pragma unroll
    for (size_t i = 0; i < M; ++i) {
        ret(i, i) = diag(i);
    }
    return ret;
}

SYCL_EXTERNAL inline void symmetric_eigen_decomposition_3x3(const Eigen::Matrix3f& A, Eigen::Vector3f& eigenvalues,
                                                            Eigen::Matrix3f& eigenvectors) {
    // symmetric matrix A
    const float EPSILON = 1e-7f;

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
        if (phi < 0.0f) phi += 3.14159265358979323846f;

        // compute
        eigenvalues(0) = 2.0f * sqrt_neg_p_over_3 * sycl::cos(phi / 3.0f) - c2 / 3.0f;
        eigenvalues(2) =
            2.0f * sqrt_neg_p_over_3 * sycl::cos((phi + 4.0f * 3.14159265358979323846f) / 3.0f) - c2 / 3.0f;
        eigenvalues(1) =
            2.0f * sqrt_neg_p_over_3 * sycl::cos((phi + 2.0f * 3.14159265358979323846f) / 3.0f) - c2 / 3.0f;
    }
    // sort
    if (eigenvalues(0) > eigenvalues(1)) {
        const float temp = eigenvalues(0);
        eigenvalues(0) = eigenvalues(1);
        eigenvalues(1) = temp;
    }
    if (eigenvalues(1) > eigenvalues(2)) {
        const float temp = eigenvalues(1);
        eigenvalues(1) = eigenvalues(2);
        eigenvalues(2) = temp;
    }
    if (eigenvalues(0) > eigenvalues(1)) {
        const float temp = eigenvalues(0);
        eigenvalues(0) = eigenvalues(1);
        eigenvalues(1) = temp;
    }

    // compute eigenvectors
    eigenvectors = Eigen::Matrix3f::Zero();

#pragma unroll
    for (int k = 0; k < 3; ++k) {
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

SYCL_EXTERNAL inline Eigen::Matrix<float, 6, 1> solve_system_6x6(const Eigen::Matrix<float, 6, 6>& A,
                                                                 const Eigen::Matrix<float, 6, 1>& b) {
    // Cholesky decomposition
    Eigen::Matrix<float, 6, 6> L = Eigen::Matrix<float, 6, 6>::Zero();
    constexpr float eps = 1e-8f;

    L(0, 0) = sycl::sqrt(A(0, 0) > eps ? A(0, 0) : eps);
    const float inv_L00 = 1.0f / L(0, 0);

    for (int i = 1; i < 6; ++i) {
        L(i, 0) = A(i, 0) * inv_L00;
    }

    for (int j = 1; j < 6; ++j) {
        float diag_sum = 0.0f;

#pragma unroll
        for (int k = 0; k < j; ++k) {
            diag_sum += L(j, k) * L(j, k);
        }

        float val = A(j, j) - diag_sum;
        L(j, j) = sycl::sqrt(val > eps ? val : eps);
        const float inv_Ljj = 1.0f / L(j, j);

        for (int i = j + 1; i < 6; ++i) {
            float off_diag_sum = 0.0f;

#pragma unroll
            for (int k = 0; k < j; ++k) {
                off_diag_sum += L(i, k) * L(j, k);
            }

            L(i, j) = (A(i, j) - off_diag_sum) * inv_Ljj;
        }
    }

    Eigen::Matrix<float, 6, 1> y = Eigen::Matrix<float, 6, 1>::Zero();
    for (int i = 0; i < 6; ++i) {
        float sum = 0.0f;

#pragma unroll
        for (int j = 0; j < i; ++j) {
            sum += L(i, j) * y(j);
        }

        const float inv_Lii = 1.0f / (sycl::fabs(L(i, i)) < eps ? eps : L(i, i));
        y(i) = (b(i) - sum) * inv_Lii;
    }

    Eigen::Matrix<float, 6, 1> x = Eigen::Matrix<float, 6, 1>::Zero();
    for (int i = 5; i >= 0; --i) {
        float sum = 0.0f;

#pragma unroll
        for (int j = i + 1; j < 6; ++j) {
            sum += L(j, i) * x(j);
        }

        const float inv_Lii = 1.0f / (sycl::fabs(L(i, i)) < eps ? eps : L(i, i));
        x(i) = (y(i) - sum) * inv_Lii;
    }

    return x;
}

inline sycl::vec<float, 4> to_sycl_vec(const Eigen::Vector4f& vec) { return {vec[0], vec[1], vec[2], vec[3]}; }

inline std::array<sycl::vec<float, 4>, 4> to_sycl_vec(const Eigen::Matrix4f& mat) {
    std::array<sycl::vec<float, 4>, 4> vecs;
    for (size_t i = 0; i < 4; ++i) {
        vecs[i] = sycl::vec<float, 4>(mat(i, 0), mat(i, 1), mat(i, 2), mat(i, 3));
    }
    return vecs;
}

inline Eigen::Vector4f from_sycl_vec(const sycl::vec<float, 4>& vec) { return {vec.x(), vec.y(), vec.z(), vec.w()}; }

inline Eigen::Matrix4f from_sycl_vec(const std::array<sycl::vec<float, 4>, 4>& vecs) {
    Eigen::Matrix4f mat;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            mat(i, j) = vecs[i][j];
        }
    }
    return mat;
}

namespace lie {
// (x, y, z) -> | 0, -z,  y |
//              | z,  0, -x |
//              |-y,  x,  0 |
SYCL_EXTERNAL inline Eigen::Matrix3f skew(const Eigen::Vector3f& x) {
    Eigen::Matrix3f ret;
    ret << 0.0f, -x[2], x[1],  // nolint
        x[2], 0.0f, -x[0],     // nolint
        -x[1], x[0], 0.0f;
    return ret;
}

SYCL_EXTERNAL inline Eigen::Matrix3f skew(const Eigen::Vector4f& x) {
    Eigen::Matrix3f ret;
    ret << 0.0f, -x[2], x[1],  // nolint
        x[2], 0.0f, -x[0],     // nolint
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
        const Eigen::Matrix3f V = (Eigen::Matrix3f::Identity() + (1.0f - std::cos(theta)) / theta_sq * Omega +
                                   (theta - std::sin(theta)) / (theta_sq * theta) * Omega * Omega);
        se3.translation() = V * a.tail<3>();
    }

    return se3;
}
}  // namespace lie
}  // namespace eigen_utils
}  // namespace sycl_points
