# sycl_points
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fateshelled/sycl_points)

A C++ header-only point cloud processing library accelerated with SYCL.

## Overview

sycl_points implements common point cloud processing operations with SYCL, targeting CPUs, Intel iGPUs, and NVIDIA GPUs. Two SYCL implementations are supported:

- **Intel oneAPI DPC++** (default) — Intel CPUs, Intel iGPUs, NVIDIA GPUs via CUDA (source build required)
- **AdaptiveCpp** (formerly hipSYCL) — OpenMP CPUs, NVIDIA (CUDA), AMD (HIP; not supported by this project), and more

Developed with reference to [small_gicp](https://github.com/koide3/small_gicp) and [gtsam_points](https://github.com/koide3/gtsam_points).

### Key features

- Point cloud data structures for CPU and accelerator memory, accessible from host and device kernels
- Nearest neighbor search: KD-tree, octree, brute force
- Registration
  - ICP: Point to Point, Point to Plane, Point to Distribution, GICP, Genz-ICP
  - Robust estimation (HUBER, TUKEY, CAUCHY, GEMAN_MCCLURE)
  - VICP: estimates sensor velocity to compensate motion distortion; source cloud requires a `time` field
  - Rotation constraint via Jensen-Bregman LogDet divergence; both clouds require `covariance` matrices
- Submapping: voxel hashmap, occupancy grid map
- Filtering: L∞ (Chebyshev) distance, random sampling, FPS, angle incidence, voxel grid and polar grid downsampling
- File I/O: PLY and PCD, ASCII and binary, CPU and shared memory

Planned work: further parallelization of algorithms.

### Supported devices

- Intel CPU (OpenCL backend)
- Intel iGPU (Level Zero or OpenCL backend)
- NVIDIA GPU (CUDA backend)

Notes: AMD CPUs are expected to work but are untested; AMD GPUs are not tested or supported by this project (no hardware available).

## Requirements

- C++20 or later
- Eigen, GTest
- One of the following SYCL implementations:
  - **Intel oneAPI DPC++** (default) — https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/
  - **AdaptiveCpp** — https://github.com/AdaptiveCpp/AdaptiveCpp
- Optional (Intel DPC++ only):
  - Intel oneAPI for NVIDIA GPUs — binary packages discontinued, build from source: https://github.com/intel/llvm
  - Intel GPU driver — https://dgpu-docs.intel.com/driver/client/overview.html

### Install dependencies

#### Eigen and GTest
```bash
sudo apt update
sudo apt install libeigen3-dev libgtest-dev
```

#### oneAPI DPC++
See the [installation guide](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/).

```bash
# download the key to system keyring
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# add signed entry to apt sources and configure the APT client to use Intel repository:
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

# install package
sudo apt update
sudo apt install intel-cpp-essentials
```

#### Intel oneAPI for NVIDIA GPUs (optional)

> Binary packages are discontinued; Intel DPC++ must be built from source.
> See [docs/install_dpcpp_nvidia.md](docs/install_dpcpp_nvidia.md).

#### Intel GPU driver (optional)
See the [driver documentation](https://dgpu-docs.intel.com/driver/client/overview.html).

```bash
# add apt repository
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics

# install compute related packages
sudo apt install -y intel-metrics-discovery intel-opencl-icd clinfo intel-gsc

# install media related packages
sudo apt install -y intel-media-va-driver-non-free libmfx1 libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo

# install OpenCL
sudo apt-get install -y intel-ocloc

# add permission
sudo gpasswd -a ${USER} render
newgrp render
```

#### AdaptiveCpp

See [docs/install_adaptive_cpp.md](docs/install_adaptive_cpp.md).

## Build and run example

### Intel oneAPI DPC++ (default)

```bash
source /opt/intel/oneapi/setvars.sh

# build example
cd cpp
mkdir build && cd build
cmake .. -DSYCL_IMPL=IntelDPCPP
make

# show device list
sycl-ls

# # example output
# [level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Graphics 12.70.4 [1.6.33276+22]
# [opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Core(TM) Ultra 7 265K OpenCL 3.0 (Build 0) [2025.19.4.0.18_160000.xmain-hotfix]
# [opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Graphics OpenCL 3.0 NEO  [25.13.33276]
# [cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA GeForce RTX 3060 8.6 [CUDA 12.9]

# select the device with `ONEAPI_DEVICE_SELECTOR`

# run example with Level Zero device (iGPU)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./example_registration

# run example with OpenCL device (iGPU, alternative backend)
ONEAPI_DEVICE_SELECTOR=opencl:1 ./example_registration

# run example with CUDA device
ONEAPI_DEVICE_SELECTOR=cuda:0 ./example_registration
```

### AdaptiveCpp

```bash
cd cpp

mkdir build && cd build
cmake .. -DSYCL_IMPL=AdaptiveCpp
make

# show device list
acpp-info -l

# # example output
# =================Backend information===================
# Loaded backend 0: Level Zero
#   Found device: Intel(R) Graphics
# Loaded backend 1: CUDA
#   Found device: NVIDIA GeForce RTX 5060 Ti
# Loaded backend 2: OpenCL
#   Found device: Intel(R) Graphics
# Loaded backend 3: OpenMP
#   Found device: AdaptiveCpp OpenMP host device

# select the device with `ACPP_VISIBILITY_MASK`
# for a specific CUDA device number, use `CUDA_VISIBLE_DEVICES`

# run example with Level Zero device
ACPP_VISIBILITY_MASK=ze ./example_registration

# run example with OpenCL device (alternative backend)
ACPP_VISIBILITY_MASK=ocl ./example_registration

# run example with CPU OpenMP
ACPP_VISIBILITY_MASK=omp ./example_registration

# run example with CUDA device
ACPP_VISIBILITY_MASK=cuda ./example_registration
# or
CUDA_VISIBLE_DEVICES=0 ./example_registration
```

> **Note:** Switching `SYCL_IMPL` or `ACPP_TARGETS` requires a fresh build directory or deleting `CMakeCache.txt`,
> since CMake caches compiler and target selections. Use separate directories (e.g. `build_omp`, `build_cuda`)
> when testing multiple configurations.

## References
- https://github.com/koide3/small_gicp
- https://github.com/koide3/gtsam_points
- **GenZ-ICP: Generalizable and Degeneracy-Robust LiDAR Odometry Using an Adaptive Weighting (2024)** — https://arxiv.org/abs/2411.06766
- **Informed, Constrained, Aligned: A Field Analysis on Degeneracy-aware Point Cloud Registration in the Wild (2024)** — https://arxiv.org/abs/2408.11809

## License
Released under Apache License 2.0.
