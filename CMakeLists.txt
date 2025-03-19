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

option(BUILD_WITH_MARCH_NATIVE "Build with -march=native" ON)

find_package(IntelSYCL REQUIRED)
find_package(Eigen3 REQUIRED)

# For OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# test_memcpy
function(add_test_exec EXEC_NAME)
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
    -fsycl-targets=spir64,nvptx64-nvidia-cuda
  )

  if(BUILD_WITH_MARCH_NATIVE)
    target_compile_options(${EXEC_NAME} PRIVATE
      -march=native
    )
  endif()

  target_link_options(${EXEC_NAME} PRIVATE
    -fsycl
    -fsycl-targets=spir64,nvptx64-nvidia-cuda
  )
endfunction()

add_test_exec(test_memcpy)
add_test_exec(test_point_cloud)

# end test_memcpy
