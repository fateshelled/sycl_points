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
    - Iterative Closest Point (ICP)
      - Point to Point
      - Point to Plane
      - Generalized ICP (GICP)
    - Robust ICP Estimation (HUBER, TUKEY, CAUCHY, GERMAN_MCCLURE)
    - Adaptive max correspondence distance by inlier points ratio
- Preprocessing filter
    - L∞ distance (chebyshev distance) filter
    - Random sampling
    - Farthest point sampling (FPS)
    - Voxel downsampling
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

- Intel Core Ultra 7 265K (Power Mode: Performance) + RTX 3060

| process                |        CPU |       iGPU |   RTX 3060 |
| ---------------------- | ---------: | ---------: | ---------: |
| 1. to PointCloudShared |  213.45 us |  983.95 us |  291.99 us |
| 2. Downsampling        | 1139.45 us | 1621.40 us | 1829.96 us |
| 3. KDTree build        |  924.04 us |  908.26 us |  932.47 us |
| 4. KDTree kNN Search   | 1106.85 us | 1641.91 us |  954.67 us |
| 5. compute Covariances |  103.35 us |  213.60 us |  121.15 us |
| 6. compute Normals     |   86.73 us |  191.48 us |   62.35 us |
| 7. update Covariance   |   80.93 us |  170.75 us |   19.07 us |
| 8. Registration        |  876.55 us | 2591.27 us |  861.98 us |
| Total                  | 4531.35 us | 8322.62 us | 5073.64 us |


- Intel Core i5 12600H (Power Mode: Performance)

| process                |        CPU |       iGPU |
| ---------------------- | ---------: | ---------: |
| 1. to PointCloudShared |  381.53 us |  744.67 us |
| 2. Downsampling        | 1959.32 us | 1567.27 us |
| 3. KDTree build        | 1176.00 us | 1087.54 us |
| 4. KDTree kNN Search   | 2642.29 us | 1827.11 us |
| 5. compute Covariances |  189.26 us |  159.68 us |
| 6. compute Normals     |  152.16 us |  141.44 us |
| 7. update Covariance   |  121.76 us |  119.87 us |
| 8. Registration        | 1749.61 us | 2136.16 us |
| Total                  | 8371.93 us | 7783.74 us |

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
        Driver version: 2025.19.4.0.18_160000.xmain-hotfix
        Global Memory Size: 30.7247 GB
        Local Memory Size: 256 KB
        Global Memory Cache Size: 3 MB
        Global Memory Cache Line Size: 64 byte
        Max Memory Allocation Size: 15.3623 GB
        Max Work Group Size: 8192
        Max Work Item Dimensions: 3
        Max Work Item Sizes: [1, 1, 1]
        Max Sub Groups num: 2048
        Sub Group Sizes: [4, 8, 16, 32, 64]
        Max compute units: 20
        Max Clock Frequency: 0 GHz
        Double precision support: true
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
        Driver version: 25.13.33276
        Global Memory Size: 28.4417 GB
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
        USM host allocations: true
        USM device allocations: true
        USM shared allocations: true
        USM atomic shared allocations: false
        Available: true
```

- RTX 3060

```bash
Platform: NVIDIA CUDA BACKEND
        Device: NVIDIA GeForce RTX 3060
        type: GPU
        Vendor: NVIDIA Corporation
        VendorID: 4318
        Backend name: ext_oneapi_cuda
        Backend version: 8.6
        Driver version: CUDA 12.9
        Global Memory Size: 11.6307 GB
        Local Memory Size: 99 KB
        Global Memory Cache Size: 2.25 MB
        Global Memory Cache Line Size: 128 byte
        Max Memory Allocation Size: 11.6307 GB
        Max Work Group Size: 1024
        Max Work Item Dimensions: 3
        Max Work Item Sizes: [1, 1, 1]
        Max Sub Groups num: 32
        Sub Group Sizes: [32]
        Max compute units: 28
        Max Clock Frequency: 1.807 GHz
        Double precision support: true
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
        Driver version: 2025.19.4.0.18_160000.xmain-hotfix
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
        Driver version: 25.18.33578
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
        USM host allocations: true
        USM device allocations: true
        USM shared allocations: true
        USM atomic shared allocations: false
        Available: true

```

## License
This library is released under Apache License
