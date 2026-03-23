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

project(sycl_points_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS   "Build tests"     ON)

find_package(Eigen3 REQUIRED)

if(SYCL_IMPL STREQUAL "IntelDPCPP")
  find_package(IntelSYCL REQUIRED)
  add_compile_definitions(SYCL_IMPL_INTEL_DPCPP)

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

  message(STATUS "SYCL_IMPL: IntelDPCPP (targets: ${SYCL_TARGET_FLAGS})")

elseif(SYCL_IMPL STREQUAL "AdaptiveCpp")
  # Auto-detect ACPP_TARGETS from available hardware if not specified by user
  if(NOT ACPP_TARGETS)
    set(_acpp_auto_targets "omp")

    # Detect NVIDIA GPUs via nvidia-smi
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

# Apply SYCL settings to a target
macro(apply_sycl_settings TARGET_NAME)
  if(SYCL_IMPL STREQUAL "IntelDPCPP")
    target_link_libraries(${TARGET_NAME} PRIVATE sycl)
    target_compile_options(${TARGET_NAME} PRIVATE
      -fsycl
      -fsycl-targets=${SYCL_TARGET_FLAGS}
    )
    target_link_options(${TARGET_NAME} PRIVATE
      -fsycl
      -fsycl-targets=${SYCL_TARGET_FLAGS}
    )
  elseif(SYCL_IMPL STREQUAL "AdaptiveCpp")
    add_sycl_to_target(TARGET ${TARGET_NAME})
  endif()
endmacro()

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
  add_test_exec(test_lidar_odometry_imu)
  add_test_exec(test_imu_deskew)
endif()
