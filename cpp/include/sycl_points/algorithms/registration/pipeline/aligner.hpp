#pragma once

#include <functional>

#include "sycl_points/algorithms/registration/registration.hpp"

namespace sycl_points {
namespace algorithms {
namespace registration {
namespace pipeline {

/// @brief Callable registration interface used by pipeline wrappers
using RegistrationAligner =
    std::function<RegistrationResult(const PointCloudShared&, const PointCloudShared&, const knn::KNNBase&,
                                     const TransformMatrix&, const Registration::ExecutionOptions&)>;

/// @brief Creates a callable aligner from a Registration instance
inline RegistrationAligner make_registration_aligner(const Registration::Ptr& registration) {
    return
        [registration](const PointCloudShared& source, const PointCloudShared& target, const knn::KNNBase& target_knn,
                       const TransformMatrix& initial_guess, const Registration::ExecutionOptions& options) {
            return registration->align(source, target, target_knn, initial_guess, options);
        };
}

}  // namespace pipeline
}  // namespace registration
}  // namespace algorithms
}  // namespace sycl_points
