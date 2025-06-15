# sycl_points
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fateshelled/sycl_points)

A C++ header-only point cloud processing library accelerated with SYCL for heterogeneous computing systems.

## Overview

sycl_points provides efficient implementations of common point cloud processing operations using Intel's SYCL (Single-source heterogeneous programming for C++) standard. The library enables accelerated processing on CPUs, Intel iGPU and NVIDIA GPUs supported by SYCL.

This project was developed with reference to small_gicp, a lightweight point cloud registration library.
- https://github.com/koide3/small_gicp

### Key features
- Efficient point cloud data structures for CPU and accelerator memory
    - Data structures accessible from CPU and device kernel
- K-nearest neighbor search
    - KD-trees
    - Brute force
- Point cloud registration
    - Iterative Closest Point (ICP Point to Point)
    - Generalized Iterative Closest Point (GICP)
    - Robust ICP Estimation (HUBER, TUKEY, CAUCHY, GERMAN_MCCLURE)
    - Adaptive max correspondence distance by inlier points ratio
- Preprocessing filter
    - L∞ distance (chebyshev distance) filter
    - Random sampling
    - Voxel downsampling

### Future optimization work will include
- Algorithm refinements for better parallelization

### Supported Device
- Intel CPU (OpenCL backend)
- Intel iGPU (OpenCL backend)
- NVIDIA GPU (CUDA backend)

note:
- `level_zero` backend is not supported.
- AMD CPU is worked fine but not optimized.
- Intel Arc GPU is worked but too slow.

## Requirements

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
sudo apt install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc

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

### Benchmark

#### example_registration

- `CPU`: Intel Core Ultra 7 265K (Power Mode: Performance)
- `iGPU`: Intel Graphics

| process                |        CPU |       iGPU |   RTX 3060 |
| ---------------------- | ---------: | ---------: | ---------: |
| 1. to PointCloudShared |  229.85 us |  982.98 us |  296.26 us |
| 2. Downsampling        | 1136.21 us | 1662.89 us | 1769.61 us |
| 3. KDTree build        | 1026.72 us | 1010.99 us | 1012.56 us |
| 4. KDTree kNN Search   | 1121.59 us | 1651.67 us |  865.65 us |
| 5. compute Covariances |  105.05 us |  214.01 us |   87.51 us |
| 6. compute Normals     |   93.06 us |  193.18 us |   61.66 us |
| 7. update Covariance   |   88.56 us |  176.38 us |   20.54 us |
| 8. Registration        |  898.55 us | 2612.08 us |  821.15 us |
| Total                  | 4699.59 us | 8504.18 us | 4934.94 us |

## License
This library is released under Apache License
