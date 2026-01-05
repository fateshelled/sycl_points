# sycl_points
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fateshelled/sycl_points)

A C++ header-only point cloud processing library accelerated with SYCL for heterogeneous computing systems.

## Overview

sycl_points provides efficient implementations of common point cloud processing operations using Intel's SYCL (Single-source heterogeneous programming for C++) standard. The library enables accelerated processing on CPUs, Intel iGPU and NVIDIA GPUs supported by SYCL.

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
- Intel oneAPI DPC++
    - https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/
- Intel oneAPI for NVIDIA® GPUs (optional)
    - https://developer.codeplay.com/apt/index.html
- Intel GPU Driver (optional)
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
For the latest information, please refer to the following URL:
- https://developer.codeplay.com/apt/index.html

```bash
# Add Intel®'s package signing key and APT repository:
sudo wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \ | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

# Add Codeplay's package signing key and APT repository:
sudo wget -qO - https://developer.codeplay.com/apt/public.key | gpg --dearmor | sudo tee /usr/share/keyrings/codeplay-keyring.gpg > /dev/null && echo "deb [signed-by=/usr/share/keyrings/codeplay-keyring.gpg] https://developer.codeplay.com/apt all main" | sudo tee /etc/apt/sources.list.d/codeplay.list

# update apt repository
sudo apt update

# install package, specify cuda version
sudo apt purge oneapi-nvidia*
sudo apt install oneapi-nvidia-12.6
```

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

## Build and run example

```bash
source /opt/intel/oneapi/setvars.sh

# build example
cd cpp
mkdir build && cd build
cmake ..
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
#       Intel Arc GPU is worked but too slow.

# run example with OpenCL device (iGPU)
ONEAPI_DEVICE_SELECTOR=opencl:1 ./example_registration

# run example with CUDA device
ONEAPI_DEVICE_SELECTOR=cuda:0 ./example_registration
```

#### device_query

- Intel Core Ultra 7 265K

```bash
Platform: Intel(R) OpenCL
        Device: Intel(R) Core(TM) Ultra 7 265K
        type: CPU
        Vendor: Intel(R) Corporation
        VendorID: 32902
        Backend name: opencl
        Backend version: 3.0
        Driver version: 2025.20.6.0.04_224945
        Global Memory Size: 30.7256 GB
        Local Memory Size: 256 KB
        Global Memory Cache Size: 3 MB
        Global Memory Cache Line Size: 64 byte
        Max Memory Allocation Size: 15.3628 GB
        Max Work Group Size: 8192
        Max Work Item Dimensions: 3
        Max Work Item Sizes: [1, 1, 1]
        Max Sub Groups num: 2048
        Sub Group Sizes: [4, 8, 16, 32, 64]
        Max compute units: 20
        Max Clock Frequency: 0 GHz
        Double precision support: true
        Atomic 64bit support: true
        USM host allocations: true
        USM device allocations: true
        USM shared allocations: true
        USM atomic shared allocations: true
        Available: true

Platform: Intel(R) OpenCL Graphics
        Device: Intel(R) Graphics
        type: GPU
        Vendor: Intel(R) Corporation
        VendorID: 32902
        Backend name: opencl
        Backend version: 3.0
        Driver version: 25.27.34303
        Global Memory Size: 28.4426 GB
        Local Memory Size: 64 KB
        Global Memory Cache Size: 4 MB
        Global Memory Cache Line Size: 64 byte
        Max Memory Allocation Size: 3.99999 GB
        Max Work Group Size: 1024
        Max Work Item Dimensions: 3
        Max Work Item Sizes: [1, 1, 1]
        Max Sub Groups num: 128
        Sub Group Sizes: [8, 16, 32]
        Max compute units: 64
        Max Clock Frequency: 2 GHz
        Double precision support: true
        Atomic 64bit support: true
        USM host allocations: true
        USM device allocations: true
        USM shared allocations: true
        USM atomic shared allocations: false
        Available: true
```

- RTX 5060 Ti

```bash
Platform: NVIDIA CUDA BACKEND
        Device: NVIDIA GeForce RTX 5060 Ti
        type: GPU
        Vendor: NVIDIA Corporation
        VendorID: 4318
        Backend name: ext_oneapi_cuda
        Backend version: 12.0
        Driver version: CUDA 12.8
        Global Memory Size: 15.4649 GB
        Local Memory Size: 99 KB
        Global Memory Cache Size: 32 MB
        Global Memory Cache Line Size: 128 byte
        Max Memory Allocation Size: 15.4649 GB
        Max Work Group Size: 1024
        Max Work Item Dimensions: 3
        Max Work Item Sizes: [1, 1, 1]
        Max Sub Groups num: 32
        Sub Group Sizes: [32]
        Max compute units: 36
        Max Clock Frequency: 2.692 GHz
        Double precision support: true
        Atomic 64bit support: true
        USM host allocations: true
        USM device allocations: true
        USM shared allocations: true
        USM atomic shared allocations: true
        Available: true
```

- Intel Core i5 12600H

```bash
Platform: Intel(R) OpenCL
        Device: 12th Gen Intel(R) Core(TM) i5-12600H
        type: CPU
        Vendor: Intel(R) Corporation
        VendorID: 32902
        Backend name: opencl
        Backend version: 3.0
        Driver version: 2025.20.6.0.04_224945
        Global Memory Size: 31.0784 GB
        Local Memory Size: 256 KB
        Global Memory Cache Size: 1.25 MB
        Global Memory Cache Line Size: 64 byte
        Max Memory Allocation Size: 15.5392 GB
        Max Work Group Size: 8192
        Max Work Item Dimensions: 3
        Max Work Item Sizes: [1, 1, 1]
        Max Sub Groups num: 2048
        Sub Group Sizes: [4, 8, 16, 32, 64]
        Max compute units: 16
        Max Clock Frequency: 0 GHz
        Double precision support: true
        Atomic 64bit support: true
        USM host allocations: true
        USM device allocations: true
        USM shared allocations: true
        USM atomic shared allocations: true
        Available: true

Platform: Intel(R) OpenCL Graphics
        Device: Intel(R) Iris(R) Xe Graphics
        type: GPU
        Vendor: Intel(R) Corporation
        VendorID: 32902
        Backend name: opencl
        Backend version: 3.0
        Driver version: 25.27.34303
        Global Memory Size: 28.7742 GB
        Local Memory Size: 64 KB
        Global Memory Cache Size: 0.46875 MB
        Global Memory Cache Line Size: 64 byte
        Max Memory Allocation Size: 3.99999 GB
        Max Work Group Size: 512
        Max Work Item Dimensions: 3
        Max Work Item Sizes: [1, 1, 1]
        Max Sub Groups num: 64
        Sub Group Sizes: [8, 16, 32]
        Max compute units: 80
        Max Clock Frequency: 1.4 GHz
        Double precision support: false
        Atomic 64bit support: true
        USM host allocations: true
        USM device allocations: true
        USM shared allocations: true
        USM atomic shared allocations: false
        Available: true
```

## References
- https://github.com/koide3/small_gicp
- https://github.com/koide3/gtsam_points
- **GenZ-ICP: Generalizable and Degeneracy-Robust LiDAR Odometry Using an Adaptive Weighting (2024)**
  - https://arxiv.org/abs/2411.06766
- **Informed, Constrained, Aligned: A Field Analysis on Degeneracy-aware Point Cloud Registration in the Wild (2024)**
  - https://arxiv.org/abs/2408.11809

## License
This library is released under Apache License 2.0
