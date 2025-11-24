cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_COMPILER "icx" CACHE PATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "icpx" CACHE PATH "C++ compiler" FORCE)

project(sycl_points LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS   "Build tests"     ON)

find_package(IntelSYCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_cmake QUIET)

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

# Install ROS 2
if (ament_cmake_FOUND)
  set(BUILD_EXAMPLES OFF)
  set(BUILD_TESTS    OFF)

  add_library(${PROJECT_NAME} INTERFACE)
  target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
  ament_package()

  install(
      DIRECTORY "include/"
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

endif()

# Build example
if(BUILD_EXAMPLES)
  function(add_example_exec EXEC_NAME)
    add_executable(${EXEC_NAME}
      examples/${EXEC_NAME}.cpp
    )
    target_include_directories(${EXEC_NAME} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(${EXEC_NAME} PRIVATE
      sycl
      Eigen3::Eigen
    )
    target_compile_options(${EXEC_NAME} PRIVATE
      -fsycl
      -fsycl-targets=${SYCL_TARGET_FLAGS}
    )

    target_link_options(${EXEC_NAME} PRIVATE
      -fsycl
      -fsycl-targets=${SYCL_TARGET_FLAGS}
    )
  endfunction()

  add_example_exec(example_memcpy)
  add_example_exec(example_point_cloud)
  add_example_exec(example_registration)
  add_example_exec(device_query)
endif()

# Build test
if (BUILD_TESTS)
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
      sycl
      Eigen3::Eigen
      GTest::GTest GTest::Main
    )
    target_compile_options(${EXEC_NAME} PRIVATE
      -fsycl
      -fsycl-targets=${SYCL_TARGET_FLAGS}
    )

    target_link_options(${EXEC_NAME} PRIVATE
      -fsycl
      -fsycl-targets=${SYCL_TARGET_FLAGS}
    )
    add_test(NAME ${EXEC_NAME} COMMAND ${EXEC_NAME})
  endfunction()

  add_test_exec(test_eigen_utils)
  add_test_exec(test_kdtree)
  add_test_exec(test_octree)
  add_test_exec(test_voxel_hash_map)
  add_test_exec(test_occupancy_grid_map)
  add_test_exec(test_file_io)
  add_test_exec(test_imu_deskew)
  add_test_exec(test_relative_pose_deskew)
endif()
