# oneAPI DPC++ Compiler (NVIDIA GPU 向け) ビルド手順

Intel が主導するオープンソースの SYCL コンパイラである [oneAPI DPC++ Compiler (intel/llvm)](https://github.com/intel/llvm) を、NVIDIA GPU (CUDA) バックエンドを有効にしてソースからビルドする手順です。

> **注意:** Intel oneAPI for NVIDIA® GPUs は、以前 Codeplay が `oneapi-nvidia-*` パッケージとして apt 配布していましたが、**現在はバイナリ配布が終了**しています。
> NVIDIA GPU で DPC++ を使用するには、`intel/llvm` リポジトリからコンパイラを自前でビルドする必要があります。
> この場合、**`intel-cpp-essentials` は apt からインストール不要**です（ソースビルドした DPC++ がその代わりになります）。

---

## 参考リンク

- https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md

---

## 1. リポジトリのクローン

特定のリリースバージョン（`v6.3.0` 等）を指定してクローンします。最新のタグはリポジトリのリリース一覧を確認してください。

- リリース一覧: https://github.com/intel/llvm/releases

```bash
git clone https://github.com/intel/llvm.git --depth 1 --branch v6.3.0
cd llvm
```

---

## 2. ビルド設定 (CUDA バックエンド有効)

```bash
python buildbot/configure.py --cuda
```

CUDA Toolkit が `/usr/local/cuda` 以外にある場合は `--cmake-opt` で指定してください:

```bash
python buildbot/configure.py --cuda \
    --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.x"
```

---

## 3. ビルド & デプロイ

```bash
python buildbot/compile.py -j $(nproc)
```

> **注意:** `cmake --install build` は使用しないでください。全コンポーネントをインストールしようとするためエラーになります。
> `buildbot/compile.py` は内部で `deploy-sycl-toolchain` ターゲットをビルドし、
> SYCL ツールチェーンに必要なファイルのみを `build/install/` に配置します。

ビルドには数十分かかる場合があります。

---

## 4. 環境変数の設定

ビルド成果物は `build/install/` に配置されます。

```bash
export DPCPP_HOME=/path/to/llvm   # クローンしたディレクトリ
export PATH=$DPCPP_HOME/build/install/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/build/install/lib:$LD_LIBRARY_PATH
```

任意のパス（例: `/opt/intel/llvm`）にインストールしたい場合は、以下の2通りの方法があります。

**案1: ディレクトリを事前に作成してオーナーを変える**

```bash
sudo mkdir -p /opt/intel/llvm
sudo chown $USER /opt/intel/llvm
python buildbot/configure.py --cuda \
    --cmake-opt="-DCMAKE_INSTALL_PREFIX=/opt/intel/llvm"
python buildbot/compile.py -j $(nproc)
```

**案2: `build/install/` にビルドしてからコピー**

```bash
python buildbot/compile.py -j $(nproc)

# build/install/ は $ORIGIN 相対 RPATH を使用しているため、
# 別ディレクトリへコピーしてもライブラリの依存関係は正しく解決される
sudo mkdir -p /opt/intel/llvm
sudo cp -r build/install/. /opt/intel/llvm/
```

いずれの場合も環境変数は以下:

```bash
export PATH=/opt/intel/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/intel/llvm/lib:$LD_LIBRARY_PATH
```

---

## 5. 動作確認

```bash
# 利用可能な SYCL デバイスを確認
sycl-ls
```

NVIDIA GPU が表示されれば成功です。

```bash
# NVIDIA GPU を指定して実行する場合
ONEAPI_DEVICE_SELECTOR=cuda:0 ./your_program
```
