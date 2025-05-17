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

option(BUILD_TEST "Build test" ON)

find_package(IntelSYCL REQUIRED)
find_package(Eigen3 REQUIRED)

# For OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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

# add example
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

# add test
if (BUILD_TEST)
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
  endfunction()

  add_test_exec(test_eigen_utils)
  add_test_exec(test_algorithms)
  add_test_exec(test_bitonic_sort)
  add_test_exec(test_kdtree)
endif()
