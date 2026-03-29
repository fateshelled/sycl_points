# sycl_points

[sycl_points](https://github.com/fateshelled/sycl_points)
はSYCL (System-wide Compute Language)を用いて実装されたC++ Headerオンリー点群処理ライブラリです。

点群処理ライブラリではありますが、LiDARオドメトリ推定向けの機能をメインに実装しています。

SYCLを使うことでCPU, Intel iGPU, NVIDIA GPUなどのハードウェア上で同一のコードを実行できます。

SYCLはあくまで規格であり、Intel oneAPI DPC++, AdaptiveCppなど、いくつかのSYCL実装が存在しています。

このライブラリでは、以下の SYCL 実装での動作を確認しています：

- [oneAPI DPC++ Compiler (intel/llvm)](install_dpcpp_nvidia.md)
- [AdaptiveCpp](install_adaptive_cpp.md)
