#!/usr/bin/env bash
#
# detect_sycl_aot_targets.sh
# -----------------------------------------------------------------------------
# Detect the available SYCL devices on this machine and emit an explicit
# Ahead-Of-Time (AOT) compilation target specification suitable for passing
# to CMake.
#
# Usage:
#   ./detect_sycl_aot_targets.sh [--impl IntelDPCPP|AdaptiveCpp] [--print-cmake]
#
#   --impl <impl>   SYCL implementation. Default: $SYCL_IMPL or "IntelDPCPP".
#   --print-cmake   Also print a ready-to-use CMake -D argument.
#
# Examples:
#   ./detect_sycl_aot_targets.sh --impl IntelDPCPP
#       -> spir64_x86_64,nvidia_gpu_sm_120
#   ./detect_sycl_aot_targets.sh --impl AdaptiveCpp --print-cmake
#       -> -DACPP_TARGETS="omp;cuda:sm_120"
#
# Notes:
#   * For Intel DPC++, output is a comma-separated list of -fsycl-targets triples.
#     For AdaptiveCpp, output is a semicolon-separated list of --acpp-targets.
#   * NVIDIA compute capability is read from `nvidia-smi` (falls back to a
#     generic nvptx64/cuda target if unavailable, with a warning).
#   * For AdaptiveCpp the emitted var name is ACPP_TARGETS (already consumed by
#     cmake/SYCLSetup.cmake). For Intel DPC++ it is SYCL_AOT_TARGETS, which must
#     be wired into cmake/SYCLSetup.cmake (see the companion note in the repo).
# -----------------------------------------------------------------------------

set -euo pipefail

IMPL="${SYCL_IMPL:-IntelDPCPP}"
PRINT_CMAKE=0

for arg in "$@"; do
  case "$arg" in
    --impl) : ;; # value consumed below
    --print-cmake) PRINT_CMAKE=1 ;;
    --help|-h)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    IntelDPCPP|AdaptiveCpp) IMPL="$arg" ;;
    *) # ignore unknown positional (e.g. value of a preceding --impl)
      : ;;
  esac
done

# Re-scan for --impl <value> form
while [[ $# -gt 0 ]]; do
  case "$1" in
    --impl) IMPL="${2:-$IMPL}"; shift 2 ;;
    *) shift ;;
  esac
done

if [[ "$IMPL" != "IntelDPCPP" && "$IMPL" != "AdaptiveCpp" ]]; then
  echo "error: unknown SYCL_IMPL '$IMPL' (expected IntelDPCPP or AdaptiveCpp)" >&2
  exit 1
fi

# ---- gather device list -------------------------------------------------------
SYCL_LS="$(command -v sycl-ls || true)"
if [[ -z "$SYCL_LS" ]]; then
  echo "error: sycl-ls not found in PATH. Source the SYCL toolchain (e.g. setvars.sh) first." >&2
  exit 1
fi

DEVICE_LINES="$("$SYCL_LS" 2>/dev/null || true)"

has_nvidia=0
has_intel_gpu=0
has_amd=0
has_cpu=0

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  backend="${line#\[}"
  backend="${backend%%:*}"
  lower="$(echo "$line" | tr '[:upper:]' '[:lower:]')"
  case "$backend" in
    *cuda*)            has_nvidia=1 ;;
    *hip*)             has_amd=1 ;;
    *level_zero*|*opencl*)
      if [[ "$lower" == *intel* ]]; then has_intel_gpu=1; fi
      if [[ "$lower" == *cpu* || "$backend" == "opencl" ]]; then :; fi
      ;;
    host)              has_cpu=1 ;;
  esac
  # opencl:cpu reports as Intel CPU; treat as CPU capability
  if [[ "$backend" == "opencl" && "$lower" == *cpu* ]]; then has_cpu=1; fi
done <<< "$DEVICE_LINES"

# ---- NVIDIA compute capability ------------------------------------------------
nvidia_sm=""
if [[ "$has_nvidia" -eq 1 ]]; then
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' \r')"
  if [[ "$cc" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    maj="${BASH_REMATCH[1]}"
    min="${BASH_REMATCH[2]}"
    # strip leading zero on minor (e.g. 12.0 -> sm_120, 9.0 -> sm_90)
    min="$((10#$min))"
    nvidia_sm="sm_${maj}${min}"
  else
    echo "warning: could not read NVIDIA compute capability (nvidia-smi); using generic target." >&2
  fi
fi

# ---- Intel GPU arch (best-effort name -> arch) --------------------------------
intel_arch=""
if [[ "$has_intel_gpu" -eq 1 ]]; then
  gpu_name="$(echo "$DEVICE_LINES" | tr '[:upper:]' '[:lower:]')"
  case "$gpu_name" in
    *battlemage*|*b580*|*b570*) intel_arch="bmg" ;;
    *arc*|*alchemist*|*a3*|*a5*|*a7*) intel_arch="dg2" ;;
    *ponte*vecchio*|*max*)        intel_arch="pvc" ;;
    *tiger*lake*|*tgllp*)         intel_arch="tgllp" ;;
    *ice*lake*|*icllp*)           intel_arch="icllp" ;;
    *skylake*|*skl*)              intel_arch="skl" ;;
  esac
fi

# ---- AMD gfx (best-effort via rocminfo) --------------------------------------
amd_gfx=""
if [[ "$has_amd" -eq 1 ]]; then
  amd_gfx="$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)"
fi

# ---- build target list --------------------------------------------------------
targets=()

if [[ "$IMPL" == "IntelDPCPP" ]]; then
  [[ "$has_cpu" -eq 1 ]]        && targets+=("spir64_x86_64")
  if [[ "$has_intel_gpu" -eq 1 ]]; then
    if [[ -n "$intel_arch" ]]; then
      targets+=("intel_gpu_${intel_arch}")
    else
      targets+=("spir64_gen")
      echo "warning: Intel GPU arch not mapped; use -Xsycl-target-backend=spir64_gen \"-device *\" at link for AOT." >&2
    fi
  fi
  if [[ "$has_nvidia" -eq 1 ]]; then
    if [[ -n "$nvidia_sm" ]]; then
      targets+=("nvidia_gpu_${nvidia_sm}")
    else
      targets+=("nvptx64-nvidia-cuda")
    fi
  fi
  if [[ "$has_amd" -eq 1 ]]; then
    if [[ -n "$amd_gfx" ]]; then
      targets+=("amd_gpu_${amd_gfx}")
    else
      targets+=("amdgcn-amd-amdhsa")
    fi
  fi
  IFS=','; TARGET_LIST="${targets[*]}"; IFS=$' \t\n'
  CMAKE_VAR="SYCL_AOT_TARGETS"
else # AdaptiveCpp
  [[ "$has_cpu" -eq 1 ]] && targets+=("omp")
  [[ "$has_intel_gpu" -eq 1 ]] && targets+=("level0")
  if [[ "$has_nvidia" -eq 1 ]]; then
    if [[ -n "$nvidia_sm" ]]; then
      targets+=("cuda:${nvidia_sm}")
    else
      targets+=("cuda")
      echo "warning: NVIDIA compute capability unknown; 'cuda' without sm may not AOT." >&2
    fi
  fi
  if [[ "$has_amd" -eq 1 ]]; then
    if [[ -n "$amd_gfx" ]]; then
      targets+=("hip:${amd_gfx}")
    else
      targets+=("hip")
    fi
  fi
  IFS=';'; TARGET_LIST="${targets[*]}"; IFS=$' \t\n'
  CMAKE_VAR="ACPP_TARGETS"
fi

if [[ -z "${TARGET_LIST:-}" ]]; then
  echo "error: no SYCL devices detected." >&2
  exit 1
fi

echo "$TARGET_LIST"

if [[ "$PRINT_CMAKE" -eq 1 ]]; then
  echo "-D${CMAKE_VAR}=\"${TARGET_LIST}\""
fi
