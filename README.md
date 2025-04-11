# sycl_points

A C++ header-only point cloud processing library accelerated with SYCL for heterogeneous computing systems.

## Overview

sycl_points provides efficient implementations of common point cloud processing operations using Intel's SYCL (Single-source heterogeneous programming for C++) standard. The library enables accelerated processing on various hardware architectures including CPUs, GPUs, and other accelerators that support SYCL.

This project was developed with reference to small_gicp, a lightweight point cloud registration library.
- https://github.com/koide3/small_gicp

Key features:
- Efficient point cloud data structures for CPU and accelerator memory
- K-nearest neighbor search using KD-trees
- Point cloud registration (ICP with GICP variant)
- Voxel downsampling for point cloud simplification
- Covariance estimation
- Point transformations (rotation, translation)
- PLY/PCD file format support


Future optimization work will include:
- Memory access pattern improvements
- Workgroup size tuning
- Algorithm refinements for better parallelization
- Specialized optimizations for different hardware targets

## Requirements

- Intel oneAPI DPC++
    - https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-0/overview.html
- Intel oneAPI for NVIDIA® GPUs (optional)
    - https://developer.codeplay.com/apt/index.html
- Eigen

## Build and run example

To build examples:

```bash
source /opt/intel/oneapi/setvars.sh

mkdir build && cd build
cmake ..
make


./example_registration.cpp
```
