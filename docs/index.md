# sycl_points

[sycl_points](https://github.com/fateshelled/sycl_points)
はSYCL (System-wide Compute Language)を用いて実装されたC++ Headerオンリー点群処理ライブラリです。

点群処理ライブラリではありますが、LiDARオドメトリ推定向けの機能をメインに実装しています。

SYCLを使うことでCPU, Intel iGPU, NVIDIA GPUなどのハードウェア上で同一のコードを実行できます。

SYCLはあくまで規格であり、Intel oneAPI DPC++, AdaptiveCppなど、いくつかのSYCL実装が存在しています。

このライブラリではIntel oneAPI DPC++を使うことを想定した実装となっています。

※ 2025年12月時点でDPC++のNVIDIA GPU向けプラグインがダウンロードできない状況となっています。筆者が以前にインストール済みであったバージョンではsycl_pointsが動作することは確認しています。まずはIntel iGPUで動作させてみることをおすすめいたします。
