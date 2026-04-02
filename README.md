# sycl_points
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fateshelled/sycl_points)

A C++ header-only point cloud processing library accelerated with SYCL for heterogeneous computing systems.

## Overview

sycl_points provides efficient implementations of common point cloud processing operations using the SYCL (Single-source heterogeneous programming for C++) standard. The library enables accelerated processing on CPUs, Intel iGPU and NVIDIA GPUs supported by SYCL.

Two SYCL implementations are supported:
- **Intel oneAPI DPC++** (default) — Intel CPUs, Intel iGPUs, NVIDIA GPUs via Codeplay plugin
- **AdaptiveCpp** (formerly hipSYCL) — CPUs (OpenMP), NVIDIA GPUs (CUDA), AMD GPUs (HIP), and more

This project was developed with reference to small_gicp and gtsam_points
- https://github.com/koide3/small_gicp
- https://github.com/koide3/gtsam_points

### Key features
- Efficient point cloud data structures for CPU and accelerator memory
  - Data structures accessible from CPU and device kernel
- K-nearest neighbor search
  - KD-trees
  - Octree
  - Brute force
- Point cloud registration
  - Iterative Closest Point (ICP)
    - Point to Point
    - Point to Plane
    - Point to Distribution
    - Generalized ICP (GICP)
    - Genz-ICP
  - Robust Estimation (HUBER, TUKEY, CAUCHY, GEMAN_MCCLURE)
  - Colored ICP / Intensity ICP
    - Point cloud must have RGB or Intensity fields, and the target cloud requires pre-computed color or intensity gradients and geometric normal vector.
  - Velocity updating ICP (VICP)
    - Estimates sensor velocity to compensate for motion distortion in the source point cloud. The source cloud must have a `time` field for each point.
  - Rotation Constraint
    - Adding rotation constraints using `Jensen-Bregman LogDet` divergence. Source and target cloud require pre-computed `covariance` matrices.
- Submapping
  - Voxel hashmap
  - Occupancy grid map
- Filtering
  - L∞ distance (chebyshev distance) filter
  - Random sampling
  - Farthest point sampling (FPS)
  - Angle incidence filter
  - Voxel grid downsampling
  - Polar grid downsampling
- Point cloud file I/O
  - PLY and PCD format support
  - ASCII and binary format reading/writing
  - CPU and shared memory interface compatibility

### Future optimization work will include
- Algorithm refinements for better parallelization

### Supported Device
- Intel CPU (OpenCL backend)
- Intel iGPU (OpenCL backend)
- NVIDIA GPU (CUDA backend)

note:
- `level_zero` backend is not supported.
- AMD CPU will work fine.
- I do not own an AMD GPU, so it is not supported.

## Requirements

- C++20 or later
- Eigen
- GTest
- One of the following SYCL implementations:
  - **Intel oneAPI DPC++** (default)
      - https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/
  - **AdaptiveCpp** (formerly hipSYCL)
      - https://github.com/AdaptiveCpp/AdaptiveCpp
- Intel oneAPI for NVIDIA® GPUs (optional, Intel DPC++ only)
    - **Binary packages are no longer distributed**; must be built from source
    - https://github.com/intel/llvm
- Intel GPU Driver (optional, Intel DPC++ only)
    - https://dgpu-docs.intel.com/driver/client/overview.html

### Install dependencies

#### Eigen and GTest
```bash
sudo apt update
sudo apt install libeigen3-dev libgtest-dev
```

#### oneAPI DPC++
For the latest information, please refer to the following URL:
- https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/

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

#### Intel oneAPI for NVIDIA® GPUs (optional)

> **Note:** Binary packages are no longer distributed. Intel DPC++ must be built from source.
> See [docs/install_dpcpp_nvidia.md](docs/install_dpcpp_nvidia.md) for details.

#### Intel GPU Driver (Optional)
For the latest information, please refer to the following URL:
- https://dgpu-docs.intel.com/driver/client/overview.html

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

See [docs/install_adaptive_cpp.md](docs/install_adaptive_cpp.md) for details.


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

# specify the device to be used with `ONEAPI_DEVICE_SELECTOR`
# note: level_zero is not supported.

# run example with OpenCL device (iGPU)
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

# specify the device to be used with `ACPP_VISIBILITY_MASK`
# if specify cuda devive number, use `CUDA_VISIBLE_DEVICES`
# note: level_zero is not supported.

# run example with OpenCL device
ACPP_VISIBILITY_MASK=ocl ./example_registration

# run example with CPU OpenMP
ACPP_VISIBILITY_MASK=omp ./example_registration

# run example with CUDA device
ACPP_VISIBILITY_MASK=cuda ./example_registration
# or
CUDA_VISIBLE_DEVICES=0 ./example_registration
```

> **Note:** When switching between `IntelDPCPP` and `AdaptiveCpp`, or changing `ACPP_TARGETS`,
> use a **fresh build directory** or delete `CMakeCache.txt`, since the compiler and target
> selections are cached by CMake. Using separate directories (e.g. `build_omp`, `build_cuda`)
> is recommended when testing multiple target configurations.

## References
- https://github.com/koide3/small_gicp
- https://github.com/koide3/gtsam_points
- **GenZ-ICP: Generalizable and Degeneracy-Robust LiDAR Odometry Using an Adaptive Weighting (2024)**
  - https://arxiv.org/abs/2411.06766
- **Informed, Constrained, Aligned: A Field Analysis on Degeneracy-aware Point Cloud Registration in the Wild (2024)**
  - https://arxiv.org/abs/2408.11809

## License
This library is released under Apache License 2.0
