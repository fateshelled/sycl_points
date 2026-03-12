#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

namespace sycl_points {
namespace algorithms {
namespace mapping {

enum class CovarianceAggregationMode {
    ARITHMETIC = 0,
    LOG_EUCLIDEAN,
};

inline CovarianceAggregationMode CovarianceAggregationMode_from_string(const std::string& str) {
    std::string upper = str;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });
    if (upper == "ARITHMETIC") {
        return CovarianceAggregationMode::ARITHMETIC;
    }
    if (upper == "LOG_EUCLIDEAN") {
        return CovarianceAggregationMode::LOG_EUCLIDEAN;
    }
    throw std::runtime_error("[CovarianceAggregationMode_from_string] Invalid mode '" + str + "'");
}

inline std::string CovarianceAggregationMode_to_string(const CovarianceAggregationMode mode) {
    switch (mode) {
        case CovarianceAggregationMode::ARITHMETIC:
            return "ARITHMETIC";
        case CovarianceAggregationMode::LOG_EUCLIDEAN:
            return "LOG_EUCLIDEAN";
    }
    throw std::runtime_error("[CovarianceAggregationMode_to_string] Invalid covariance aggregation mode");
}

}  // namespace mapping
}  // namespace algorithms
}  // namespace sycl_points
