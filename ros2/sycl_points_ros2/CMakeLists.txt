cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_COMPILER "icx" CACHE PATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "icpx" CACHE PATH "C++ compiler" FORCE)

project(sycl_points_ros2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(IntelSYCL REQUIRED)
find_package(Eigen3 REQUIRED)

# SYCL Target
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
if(${ENABLE_CUDA_BACKEND})
  message(STATUS "Intel oneAPI for NVIDIA GPU support available")
  set(SYCL_TARGET_FLAGS "${SYCL_TARGET_FLAGS},nvptx64-nvidia-cuda")
else()
  message(STATUS "Intel oneAPI for NVIDIA GPU support not available")
endif()

# ROS 2
ament_auto_find_build_dependencies()

ament_auto_add_library(lidar_odometry SHARED
  src/lidar_odometry.cpp
)

target_include_directories(lidar_odometry  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(lidar_odometry
  sycl
  Eigen3::Eigen
)
target_compile_options(lidar_odometry PRIVATE
  -fsycl
  -fsycl-targets=${SYCL_TARGET_FLAGS}
)

target_link_options(lidar_odometry PRIVATE
  -fsycl
  -fsycl-targets=${SYCL_TARGET_FLAGS}
)

rclcpp_components_register_node(
  lidar_odometry
  PLUGIN "sycl_points::ros2::LiDAROdometryNode"
  EXECUTABLE lidar_odometry_node
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
