# SYCLSetup.cmake
# Common SYCL implementation detection and target setup.
# Reads SYCL_IMPL (IntelDPCPP | AdaptiveCpp) and sets up:
#   - find_package for the chosen implementation
#   - SYCL_IMPL_INTEL_DPCPP / SYCL_IMPL_ADAPTIVECPP compile definition
#   - SYCL_TARGET_FLAGS (IntelDPCPP only)
#   - ACPP_TARGETS default (AdaptiveCpp only)
#   - apply_sycl_settings(TARGET) macro

if(CMAKE_PROJECT_NAME) # project() 実行後のみ動作するようにガード

if(SYCL_IMPL STREQUAL "IntelDPCPP")
  find_package(IntelSYCL QUIET)
  if(IntelSYCL_FOUND)
    if(NOT TARGET sycl AND TARGET IntelSYCL::IntelSYCL)
      # Map IntelSYCL::IntelSYCL to sycl target if necessary
      add_library(sycl INTERFACE IMPORTED)
      target_link_libraries(sycl INTERFACE IntelSYCL::IntelSYCL)
    endif()
  else()
    message(STATUS "IntelSYCL package not found, using compiler driver flags (-fsycl)")
  endif()

  if(NOT TARGET sycl)
    add_library(sycl INTERFACE IMPORTED)
  endif()
  add_compile_definitions(SYCL_IMPL_INTEL_DPCPP)

  set(SYCL_TARGET_FLAGS "spir64")

  # Check for Intel oneAPI NVIDIA GPU support robustly without spawning bash
  function(check_oneapi_nvidia_support RESULT_VAR)
    find_program(SYCL_LS_PATH sycl-ls)
    if(NOT SYCL_LS_PATH)
      set(${RESULT_VAR} FALSE PARENT_SCOPE)
      return()
    endif()

    execute_process(
      COMMAND ${SYCL_LS_PATH}
      OUTPUT_VARIABLE _sycl_devices
      ERROR_QUIET
      RESULT_VARIABLE _sycl_result
    )

    if(_sycl_result EQUAL 0 AND _sycl_devices MATCHES "NVIDIA")
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
  # Default ACPP_TARGETS to "generic" (SSCP/JIT) if not specified by user.
  # The generic flow JIT-compiles for the device available at runtime
  # (CPU, Intel GPU, NVIDIA GPU, ...) and is the flow recommended by AdaptiveCpp.
  # Explicit AOT targets (e.g. -DACPP_TARGETS="generic;cuda:sm_90") use the legacy
  # cuda/hip flows, which AdaptiveCpp discourages outside specific use cases
  # (acpp emits a warning unless --acpp-no-warn-legacy-flows is given).
  if(NOT ACPP_TARGETS)
    set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp compilation targets" FORCE)
    message(STATUS "AdaptiveCpp: auto-set ACPP_TARGETS=${ACPP_TARGETS}")
  endif()

  find_package(AdaptiveCpp REQUIRED)
  add_compile_definitions(SYCL_IMPL_ADAPTIVECPP)
  message(STATUS "SYCL_IMPL: AdaptiveCpp (ACPP_TARGETS=${ACPP_TARGETS})")

else()
  message(FATAL_ERROR "Unknown SYCL_IMPL: ${SYCL_IMPL}. Must be one of IntelDPCPP or AdaptiveCpp.")
endif()

# Apply SYCL settings to a target.
# For IntelDPCPP: adds -fsycl flags and links sycl library.
# For AdaptiveCpp: calls add_sycl_to_target().
macro(apply_sycl_settings TARGET_NAME)
  if(SYCL_IMPL STREQUAL "IntelDPCPP")
    if(NOT TARGET sycl)
      message(FATAL_ERROR "SYCL target not found. Ensure include(SYCLSetup.cmake) is called after project().")
    endif()
    target_compile_options(${TARGET_NAME} PRIVATE
      -fsycl
      -fsycl-targets=${SYCL_TARGET_FLAGS}
    )
    target_link_options(${TARGET_NAME} PRIVATE
      -fsycl
      -fsycl-targets=${SYCL_TARGET_FLAGS}
    )
  elseif(SYCL_IMPL STREQUAL "AdaptiveCpp")
    # Manually replicate add_sycl_to_target to avoid target_link_libraries keyword/plain mixing
    # (ament uses plain form; AdaptiveCpp's add_sycl_to_target uses keyword PRIVATE form)
    get_target_property(_existing_compile_rule "${TARGET_NAME}" RULE_LAUNCH_COMPILE)
    if("${_existing_compile_rule}" STREQUAL "_existing_compile_rule-NOTFOUND")
      set(_existing_compile_rule "")
    endif()
    get_target_property(_existing_link_rule "${TARGET_NAME}" RULE_LAUNCH_LINK)
    if("${_existing_link_rule}" STREQUAL "_existing_link_rule-NOTFOUND")
      set(_existing_link_rule "")
    endif()
    set_target_properties("${TARGET_NAME}" PROPERTIES
      RULE_LAUNCH_COMPILE "${_existing_compile_rule} ${ACPP_COMPILER_LAUNCH_RULE}"
      RULE_LAUNCH_LINK    "${_existing_link_rule} ${ACPP_COMPILER_LAUNCH_RULE}"
    )
    # Use set_property instead of target_link_libraries to avoid keyword/plain mixing
    # (ament uses plain form; cpp targets use keyword form; this works with both)
    set_property(TARGET ${TARGET_NAME} APPEND PROPERTY LINK_LIBRARIES AdaptiveCpp::acpp-rt)
  else()
    message(FATAL_ERROR "Unknown SYCL_IMPL: ${SYCL_IMPL}. Must be one of IntelDPCPP or AdaptiveCpp.")
  endif()
endmacro()


endif() # CMAKE_PROJECT_NAME
