cmake_minimum_required(VERSION 3.15)

# SYCL implementation selection (must be set before project())
# Options: IntelDPCPP, AdaptiveCpp
set(SYCL_IMPL "IntelDPCPP" CACHE STRING "SYCL implementation to use (IntelDPCPP or AdaptiveCpp)")
set_property(CACHE SYCL_IMPL PROPERTY STRINGS "IntelDPCPP" "AdaptiveCpp")

# Set SYCL compiler before project()
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/SYCLCompiler.cmake")

project(sycl_points_example LANGUAGES CXX)

# Initialize SYCL environment after compiler identification
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/SYCLSetup.cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS   "Build tests"     ON)

find_package(Eigen3 REQUIRED)

# Build examples
if(BUILD_EXAMPLES)
  function(add_example_exec EXEC_NAME)
    add_executable(${EXEC_NAME}
      examples/${EXEC_NAME}.cpp
    )
    target_include_directories(${EXEC_NAME} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(${EXEC_NAME} PRIVATE
      Eigen3::Eigen
    )
    apply_sycl_settings(${EXEC_NAME})
  endfunction()

  add_example_exec(example_memcpy)
  add_example_exec(example_point_cloud)
  add_example_exec(example_registration)
  add_example_exec(device_query)
endif()

# Build tests
if(BUILD_TESTS)
  find_package(GTest REQUIRED)
  include(CTest)
  enable_testing()

  function(add_test_exec EXEC_NAME)
    add_executable(${EXEC_NAME}
      tests/${EXEC_NAME}.cpp
    )
    target_include_directories(${EXEC_NAME} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(${EXEC_NAME} PRIVATE
      Eigen3::Eigen
      GTest::GTest GTest::Main
    )
    apply_sycl_settings(${EXEC_NAME})
    add_test(NAME ${EXEC_NAME} COMMAND ${EXEC_NAME})
  endfunction()

  add_test_exec(test_eigen_utils)
  add_test_exec(test_kdtree)
  add_test_exec(test_octree)
  add_test_exec(test_voxel_hash_map)
  add_test_exec(test_occupancy_grid_map)
  add_test_exec(test_file_io)
  add_test_exec(test_relative_pose_deskew)
  add_test_exec(test_intensity_correction)
  add_test_exec(test_preprocess_filter)
  add_test_exec(test_downsampling_filters)
  add_test_exec(test_registration_pipeline)
  add_test_exec(test_imu_preintegration)
  add_test_exec(test_imu_factor)
  add_test_exec(test_lidar_odometry_imu)
  add_test_exec(test_imu_deskew)
  add_test_exec(test_select_device)
endif()
