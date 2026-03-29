# AdaptiveCpp ビルド手順

Intel 内蔵 GPU (OpenCL & Level Zero) および NVIDIA GPU (CUDA) の両方に対応した AdaptiveCpp のビルド手順です。

最新情報は公式ドキュメントを参照してください:
- https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md

## 動作確認環境

| 項目 | バージョン / 値 |
|------|----------------|
| OS | Ubuntu 24.04 |
| CPU | x86_64 (20コア) |
| Intel GPU | Intel(R) Graphics (内蔵 GPU) |
| NVIDIA GPU | NVIDIA GeForce RTX 5060 Ti |
| NVIDIA ドライバ | 570.172.08 |
| CUDA | 12.8 (`/usr/local/cuda`) |
| LLVM / Clang | 20.1.2 |
| Boost | 1.83 |
| Level Zero | 1.24.1 |
| CMake | 3.28.3 |
| AdaptiveCpp | v25.10.0 |

---

## 1. 依存パッケージのインストール

```bash
export LLVM_VERSION=20
# LLVM / Clang 開発ファイル
sudo apt-get install -y \
    clang-${LLVM_VERSION} \
    llvm-${LLVM_VERSION}-dev \
    libclang-${LLVM_VERSION}-dev \
    lld-${LLVM_VERSION} \
    libomp-${LLVM_VERSION}-dev


# OpenCL・Level Zero
# (intel-opencl-icd, libze-dev, libze-intel-gpu1, libze1 がインストール済みであること)
sudo apt-get install -y \
    intel-opencl-icd \
    libze-dev \
    libze-intel-gpu1 \
    libze1

# Boost
sudo apt-get install -y libboost-dev

# CUDA Toolkit は別途 NVIDIA 公式からインストール済みであること
# ※ RTX 5000シリーズの場合、CUDA Toolkit のバージョンは12.8以降であること
# https://developer.nvidia.com/cuda-downloads
```

> **Note:** SPIRV-LLVM-Translator は CMake ビルド時に自動的に GitHub からダウンロード・ビルドされます。
> 事前インストールは不要です。

---

## 2. リポジトリのクローン（または既存ディレクトリを使用）

```bash
git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git
cd AdaptiveCpp
```

---

## 3. ビルドディレクトリの作成

```bash
mkdir -p build
cd build
```

---

## 4. CMake 設定

```bash
export LLVM_VERSION=20
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-${LLVM_VERSION} \
  -DCMAKE_C_COMPILER=/usr/bin/clang-${LLVM_VERSION} \
  -DWITH_CUDA_BACKEND=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DCUDA_DEVICE_LIBS_PATH=/usr/local/cuda/nvvm/libdevice \
  -DWITH_LEVEL_ZERO_BACKEND=ON \
  -DWITH_OPENCL_BACKEND=ON
```

### 主要な CMake オプションの説明

| オプション | 値 | 説明 |
|-----------|-----|------|
| `CMAKE_BUILD_TYPE` | `Release` | リリースビルド (最適化あり) |
| `CMAKE_CXX_COMPILER` | `clang++-20` | C++ コンパイラに Clang 20 を指定 |
| `LLVM_DIR` | `/usr/lib/llvm-20/lib/cmake/llvm` | LLVM 20 の cmake 設定ディレクトリ |
| `WITH_CUDA_BACKEND` | `ON` | NVIDIA GPU (CUDA) バックエンドを有効化 |
| `CUDA_TOOLKIT_ROOT_DIR` | `/usr/local/cuda` | CUDA Toolkit のルートディレクトリ |
| `CUDA_DEVICE_LIBS_PATH` | `/usr/local/cuda/nvvm/libdevice` | CUDA デバイス用ライブラリ (`libdevice.10.bc`) のパス |
| `WITH_LEVEL_ZERO_BACKEND` | `ON` | Intel GPU (Level Zero) バックエンドを有効化 |
| `WITH_OPENCL_BACKEND` | `ON` | OpenCL バックエンドを有効化 (Intel GPU でも使用可) |

> **Note:** `WITH_LEVEL_ZERO_BACKEND=ON` または `WITH_OPENCL_BACKEND=ON` を指定すると、
> SSCP コンパイラと LLVM-to-SPIR-V 変換レイヤが自動的に有効になります。

## 5. ビルド

```bash
cmake --build . --parallel $(nproc)
```

ビルド中に SPIRV-LLVM-Translator が自動的に GitHub からクローン・ビルドされます
(`https://github.com/AdaptiveCpp/SPIRV-LLVM-Translator` の `llvm_release_180` ブランチ)。
インターネット接続が必要です。

ビルドには数分〜十数分かかります。

---

## 6. インストール

```bash
sudo cmake --install .
```

インストール先 (`CMAKE_INSTALL_PREFIX`, デフォルト`/usr/local`) に以下が展開されます。

```
/usr/local/
├── bin/
│   ├── acpp          # AdaptiveCpp コンパイラドライバ
│   ├── acpp-info     # バックエンド・デバイス情報の表示ツール
│   ├── acpp-hcf-tool
│   ├── acpp-appdb-tool
│   ├── acpp-info
│   ├── acpp-pcuda-pp
│   ├── syclcc
│   └── syclcc-clang
├── lib/
│   ├── libacpp-common.so           # AdaptiveCpp ランタイム
│   ├── libacpp-clang.so           # AdaptiveCpp ランタイム
│   ├── libacpp-rt.so           # AdaptiveCpp ランタイム
│   ├── hipSYCL/
│   │   ├── librt-backend-cuda.so   # CUDA バックエンド
│   │   ├── librt-backend-ze.so     # Level Zero バックエンド
│   │   ├── librt-backend-ocl.so    # OpenCL バックエンド
│   │   ├── librt-backend-omp.so    # OpenMP バックエンド
│   │   ├── llvm-to-backend/        # LLVM→各バックエンド変換ライブラリ
│   │   │   ├── libllvm-to-ptx.so   # LLVM IR → PTX (NVIDIA)
│   │   │   └── libllvm-to-spirv.so # LLVM IR → SPIR-V (Intel)
│   │   └── ext/llvm-spirv/         # SPIRV-LLVM-Translator ツール
│   ├── cmake
│   │   ├──hipSYCL
│   │   ├──OpenSYCL
│   │   └── AdaptiveCpp
│   └── ...
├── etc/
│   └── AdaptiveCpp
├──share/cmake/OpenCLHeadersCpp
└── include/
     ├── AdaptiveCpp
     ├── CL
     └── ...
```

---

## 7. 動作確認

```bash
acpp-info
```

以下のように Intel GPU と NVIDIA GPU が両方認識されれば成功です。

```
=================Backend information===================
Loaded backend 0: Level Zero
  Found device: Intel(R) Graphics
Loaded backend 1: CUDA
  Found device: NVIDIA GeForce RTX 5060 Ti
Loaded backend 2: OpenCL
  Found device: Intel(R) Graphics
Loaded backend 3: OpenMP
  Found device: AdaptiveCpp OpenMP host device
```

---

## トラブルシューティング

### lld が見つからない

```
Cannot find ld.lld.
```

`lld-20` をインストールしてください。

```bash
sudo apt-get install -y lld-20
```

### Level Zero ヘッダが見つからない

```
Could not find Level Zero headers
```

`libze-dev` をインストールしてください。

```bash
sudo apt-get install -y libze-dev libze1 libze-intel-gpu1
```

### SPIRV-LLVM-Translator のビルドが失敗する

CMake 設定時に `libclang-20-dev` と `llvm-20-dev` が必要です。

```bash
sudo apt-get install -y libclang-20-dev llvm-20-dev
```

### CUDA が見つからない

`CUDA_TOOLKIT_ROOT_DIR` を CUDA インストール先に正しく指定してください。

```bash
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8
```
