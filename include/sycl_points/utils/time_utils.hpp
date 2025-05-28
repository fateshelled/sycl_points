#pragma once
#include <chrono>
#include <functional>

namespace sycl_points {
namespace time_utils {

/**
 * @brief Measures execution time of any function and returns its result
 *
 * Executes the given function with arguments and measures elapsed time in microseconds.
 * Supports both void functions and functions with return values using perfect forwarding.
 *
 * @tparam TimeType Numeric type for storing elapsed time (default: double)
 * @tparam Func Function object type (function pointer, lambda, functor, etc.)
 * @tparam Args Function argument types (variadic template)
 *
 * @param func Function object to execute
 * @param elapsed_time [out] Reference to store execution time in microseconds
 * @param args Arguments to pass to the function (variadic)
 *
 * @return Function's return value (void functions return nothing)
 *
 * @note Uses std::chrono::steady_clock for timing measurement
 * @warning Precision may be limited for very short operations
 *
 * @example
 * @code
 * double time = 0.0;
 * int result = measure_execution(my_func, time, arg1, arg2);
 *
 * // Lambda with bound arguments
 * auto lambda = [&]() { return my_func(arg1, arg2); };
 * int result2 = measure_execution(lambda, time);
 * @endcode
 */
template <typename TimeType = double, typename Func, typename... Args>
auto measure_execution(Func&& func, TimeType& elapsed_time, Args&&... args)
    -> decltype(func(std::forward<Args>(args)...)) {
    auto start = std::chrono::steady_clock::now();

    if constexpr (std::is_void_v<decltype(func(std::forward<Args>(args)...))>) {
        func(std::forward<Args>(args)...);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<TimeType, std::micro>(end - start);
        elapsed_time = duration.count();
    } else {
        auto result = func(std::forward<Args>(args)...);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<TimeType, std::micro>(end - start);
        elapsed_time = duration.count();
        return result;
    }
}

}  // namespace time_utils
}  // namespace sycl_points