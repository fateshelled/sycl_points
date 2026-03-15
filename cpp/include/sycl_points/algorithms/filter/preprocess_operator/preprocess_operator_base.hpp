#pragma once

#include <functional>

#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace algorithms {
namespace filter {
namespace preprocess_operator {

class PreprocessOperatorBase {
public:
    using InitializeFlagsFn = std::function<sycl_utils::events(size_t, uint8_t)>;
    using FilterByFlagsFn = std::function<void(const PointCloudShared&, PointCloudShared&)>;

protected:
    PreprocessOperatorBase(const sycl_utils::DeviceQueue& queue, shared_vector_ptr<uint8_t> flags,
                           InitializeFlagsFn initialize_flags, FilterByFlagsFn filter_by_flags)
        : queue_(queue),
          flags_(std::move(flags)),
          initialize_flags_(std::move(initialize_flags)),
          filter_by_flags_(std::move(filter_by_flags)) {}

    void copy_source_to_output(const PointCloudShared& source, PointCloudShared& output) const {
        output = source;
    }

    sycl_utils::DeviceQueue queue_;
    shared_vector_ptr<uint8_t> flags_;
    InitializeFlagsFn initialize_flags_;
    FilterByFlagsFn filter_by_flags_;
};

}  // namespace preprocess_operator
}  // namespace filter
}  // namespace algorithms
}  // namespace sycl_points
