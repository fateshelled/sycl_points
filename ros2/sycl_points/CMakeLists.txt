cmake_minimum_required(VERSION 3.15)

# SYCL implementation selection (must be set before project())
# Options: IntelDPCPP, AdaptiveCpp
set(SYCL_IMPL "IntelDPCPP" CACHE STRING "SYCL implementation to use (IntelDPCPP or AdaptiveCpp)")
set_property(CACHE SYCL_IMPL PROPERTY STRINGS "IntelDPCPP" "AdaptiveCpp")

# Set SYCL compiler before project()
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/SYCLCompiler.cmake")

project(sycl_points LANGUAGES CXX)

# Initialize SYCL environment after compiler identification
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/SYCLSetup.cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)

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
