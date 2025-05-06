# sycl_points

A C++ header-only point cloud processing library accelerated with SYCL for heterogeneous computing systems.

## Overview

sycl_points provides efficient implementations of common point cloud processing operations using Intel's SYCL (Single-source heterogeneous programming for C++) standard. The library enables accelerated processing on CPUs, iGPUs and dGPUs supported by SYCL.

This project was developed with reference to small_gicp, a lightweight point cloud registration library.
- https://github.com/koide3/small_gicp

Key features:
- Efficient point cloud data structures for CPU and accelerator memory
- K-nearest neighbor search using KD-trees
- Point cloud registration (Point-to-Point and GICP)
- Voxel downsampling for point cloud simplification
- Covariance estimation
- Point transformations (rotation, translation)
- PLY/PCD file format support


Future optimization work will include:
- Memory access pattern improvements
- Workgroup size tuning
- Algorithm refinements for better parallelization

## Requirements

- Intel oneAPI DPC++
    - https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/
- Intel oneAPI for NVIDIA® GPUs (optional)
    - https://developer.codeplay.com/apt/index.html
- Intel GPU Driver (optional)
    - https://dgpu-docs.intel.com/driver/client/overview.html
- Eigen
- GTest

### install oneAPI DPC++
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

### Intel oneAPI for NVIDIA® GPUs (optional)
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

### install Intel GPU Driver (Optional)
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

# install level-zero and openCL 
sudo apt-get install -y libze-dev intel-ocloc

# add permission
sudo gpasswd -a ${USER} render
newgrp render
```

### install Eigen and GTest
```bash
sudo apt install libeigen3-dev libgtest-dev
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

# example output
# [level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) A310 LP Graphics 12.56.5 [1.6.32961+8]
# [level_zero:gpu][level_zero:1] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Iris(R) Xe Graphics 12.3.0 [1.6.32961+8]
# [opencl:cpu][opencl:0] Intel(R) OpenCL, 12th Gen Intel(R) Core(TM) i5-12600H OpenCL 3.0 (Build 0) [2025.19.4.0.18_160000.xmain-hotfix]
# [opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A310 LP Graphics OpenCL 3.0 NEO  [25.09.32961]
# [opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, Intel(R) Iris(R) Xe Graphics OpenCL 3.0 NEO  [25.09.32961]

# specify the device to be used
# note: not work fine with level_zero
export ONEAPI_DEVICE_SELECTOR=opencl:1

# run example
./example_registration.cpp
```
