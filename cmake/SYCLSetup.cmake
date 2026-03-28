# SYCLSetup.cmake
# Common SYCL implementation detection and target setup.
# Reads SYCL_IMPL (IntelDPCPP | AdaptiveCpp) and sets up:
#   - find_package for the chosen implementation
#   - SYCL_IMPL_INTEL_DPCPP / SYCL_IMPL_ADAPTIVECPP compile definition
#   - SYCL_TARGET_FLAGS (IntelDPCPP only)
#   - ACPP_TARGETS auto-detection (AdaptiveCpp only)
#   - apply_sycl_settings(TARGET) macro

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
    # Base target is always "generic" (SSCP/JIT) to support CPU and Intel GPU at runtime
    set(_acpp_auto_targets "generic")

    # Detect NVIDIA GPUs via nvidia-smi and add cuda:sm_XX targets
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

# Apply SYCL settings to a target.
# For IntelDPCPP: adds -fsycl flags and links sycl library.
# For AdaptiveCpp: calls add_sycl_to_target().
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
  endif()
endmacro()
