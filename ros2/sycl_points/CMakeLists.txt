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

project(sycl_points LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)

if(SYCL_IMPL STREQUAL "IntelDPCPP")
  find_package(IntelSYCL REQUIRED)
  message(STATUS "SYCL_IMPL: IntelDPCPP")
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
  message(STATUS "SYCL_IMPL: AdaptiveCpp (ACPP_TARGETS=${ACPP_TARGETS})")
endif()

# Install ROS 2
add_library(${PROJECT_NAME} INTERFACE)

# Propagate SYCL_IMPL definition to consumers of this header-only library
if(SYCL_IMPL STREQUAL "IntelDPCPP")
  target_compile_definitions(${PROJECT_NAME} INTERFACE SYCL_IMPL_INTEL_DPCPP)
elseif(SYCL_IMPL STREQUAL "AdaptiveCpp")
  target_compile_definitions(${PROJECT_NAME} INTERFACE SYCL_IMPL_ADAPTIVECPP)
endif()
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include>
  $<INSTALL_INTERFACE:include>
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_package()

install(
    DIRECTORY "../../cpp/include/"
    DESTINATION include
  )

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
