cmake_minimum_required(VERSION 3.15)

# SYCL implementation selection (must be set before project())
# Options: IntelDPCPP, AdaptiveCpp
set(SYCL_IMPL "IntelDPCPP" CACHE STRING "SYCL implementation to use (IntelDPCPP or AdaptiveCpp)")
set_property(CACHE SYCL_IMPL PROPERTY STRINGS "IntelDPCPP" "AdaptiveCpp")

if(SYCL_IMPL STREQUAL "IntelDPCPP")
  set(CMAKE_C_COMPILER "icx" CACHE PATH "C compiler")
  set(CMAKE_CXX_COMPILER "icpx" CACHE PATH "C++ compiler")
elseif(SYCL_IMPL STREQUAL "AdaptiveCpp")
  if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "acpp" CACHE PATH "C++ compiler")
  endif()
else()
  message(FATAL_ERROR "Unknown SYCL_IMPL: ${SYCL_IMPL}. Choose IntelDPCPP or AdaptiveCpp")
endif()

project(sycl_points_ros2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(Eigen3 REQUIRED)

if(SYCL_IMPL STREQUAL "IntelDPCPP")
  find_package(IntelSYCL REQUIRED)

  set(SYCL_TARGET_FLAGS "spir64")

  # Check for Intel oneAPI NVIDIA GPU support
  function(check_oneapi_nvidia_support RESULT_VAR)
    execute_process(
      COMMAND bash -c "sycl-ls | grep -q NVIDIA"
      RESULT_VARIABLE EXIT_CODE
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(EXIT_CODE EQUAL 0)
      set(${RESULT_VAR} TRUE PARENT_SCOPE)
    else()
      set(${RESULT_VAR} FALSE PARENT_SCOPE)
    endif()
  endfunction()

  check_oneapi_nvidia_support(ENABLE_CUDA_BACKEND)
  if(ENABLE_CUDA_BACKEND)
    message(STATUS "Intel oneAPI for NVIDIA GPU support available")
    set(SYCL_TARGET_FLAGS "${SYCL_TARGET_FLAGS},nvptx64-nvidia-cuda")
  else()
    message(STATUS "Intel oneAPI for NVIDIA GPU support not available")
  endif()

  add_compile_definitions(SYCL_IMPL_INTEL_DPCPP)
  message(STATUS "SYCL_IMPL: IntelDPCPP (targets: ${SYCL_TARGET_FLAGS})")

elseif(SYCL_IMPL STREQUAL "AdaptiveCpp")
  # Auto-detect ACPP_TARGETS from available hardware if not specified by user
  if(NOT ACPP_TARGETS)
    set(_acpp_auto_targets "omp")

    execute_process(
      COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _nvidia_compute_caps
      RESULT_VARIABLE _nvidia_result
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_nvidia_result EQUAL 0 AND _nvidia_compute_caps)
      string(REPLACE "\n" ";" _cap_list "${_nvidia_compute_caps}")
      set(_cuda_targets "")
      foreach(_cap IN LISTS _cap_list)
        string(STRIP "${_cap}" _cap)
        if(_cap)
          string(REPLACE "." "" _sm "${_cap}")
          list(APPEND _cuda_targets "cuda:sm_${_sm}")
        endif()
      endforeach()
      list(REMOVE_DUPLICATES _cuda_targets)
      foreach(_t IN LISTS _cuda_targets)
        string(APPEND _acpp_auto_targets ";${_t}")
      endforeach()
      message(STATUS "AdaptiveCpp: detected NVIDIA GPU(s): ${_cuda_targets}")
    else()
      message(STATUS "AdaptiveCpp: no NVIDIA GPU detected")
    endif()

    set(ACPP_TARGETS "${_acpp_auto_targets}" CACHE STRING "AdaptiveCpp compilation targets" FORCE)
    message(STATUS "AdaptiveCpp: auto-set ACPP_TARGETS=${ACPP_TARGETS}")
  endif()

  find_package(AdaptiveCpp REQUIRED)
  add_compile_definitions(SYCL_IMPL_ADAPTIVECPP)
  message(STATUS "SYCL_IMPL: AdaptiveCpp (ACPP_TARGETS=${ACPP_TARGETS})")
endif()

# ROS 2
ament_auto_find_build_dependencies()

ament_auto_add_library(lidar_odometry SHARED
  src/lidar_odometry_base_node.cpp
  src/lidar_odometry_bag_eval_node.cpp
  src/lidar_odometry_node.cpp
)

target_include_directories(lidar_odometry PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(lidar_odometry
  Eigen3::Eigen
)

if(SYCL_IMPL STREQUAL "IntelDPCPP")
  target_link_libraries(lidar_odometry sycl)
  target_compile_options(lidar_odometry PRIVATE
    -fsycl
    -fsycl-targets=${SYCL_TARGET_FLAGS}
  )
  target_link_options(lidar_odometry PRIVATE
    -fsycl
    -fsycl-targets=${SYCL_TARGET_FLAGS}
  )
elseif(SYCL_IMPL STREQUAL "AdaptiveCpp")
  add_sycl_to_target(TARGET lidar_odometry)
endif()

rclcpp_components_register_node(
  lidar_odometry
  PLUGIN "sycl_points::ros2::LiDAROdometryNode"
  EXECUTABLE lidar_odometry_node
)

rclcpp_components_register_node(
  lidar_odometry
  PLUGIN "sycl_points::ros2::LiDAROdometryBagEvalNode"
  EXECUTABLE lidar_odometry_bag_eval_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
  rviz2
  USE_SCOPED_HEADER_INSTALL_DIR
)
