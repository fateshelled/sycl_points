#include <gtest/gtest.h>

#include <string>

#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace sycl_utils {
namespace device_selector {
namespace test {

// 利用可能なデバイスを動的に調べるヘルパー
struct AvailableDevices {
    bool has_intel_cpu = false;  // DPC++: opencl backend
    bool has_intel_gpu = false;
    bool has_nvidia_gpu = false;
    bool has_amd_gpu = false;
#ifdef SYCL_IMPL_ADAPTIVECPP
    bool has_omp = false;  // AdaptiveCpp: OMP backend (CPU)
#endif

    AvailableDevices() {
        for (auto platform : sycl::platform::get_platforms()) {
            for (auto device : platform.get_devices()) {
                if (!is_supported_device(device)) continue;
                const auto vid = device.get_info<sycl::info::device::vendor_id>();
                if (vid == VENDOR_ID::INTEL && device.is_cpu()) has_intel_cpu = true;
                if (vid == VENDOR_ID::INTEL && device.is_gpu()) has_intel_gpu = true;
                if (vid == VENDOR_ID::NVIDIA && device.is_gpu()) has_nvidia_gpu = true;
                if (vid == VENDOR_ID::AMD && device.is_gpu()) has_amd_gpu = true;
#ifdef SYCL_IMPL_ADAPTIVECPP
                if (vid == VENDOR_ID::OMP) has_omp = true;
#endif
            }
        }
    }
};

static const AvailableDevices g_devices;

// ---------------------------------------------------------------
// 異常系: 無効なベンダー文字列
// ---------------------------------------------------------------
TEST(SelectDeviceTest, InvalidVendorThrows) {
    EXPECT_THROW(
        { select_device("unknown_vendor", "cpu"); },
        std::runtime_error);
}

TEST(SelectDeviceTest, EmptyVendorThrows) {
    EXPECT_THROW(
        { select_device("", "cpu"); },
        std::runtime_error);
}

TEST(SelectDeviceTest, InvalidVendorErrorMessage) {
    try {
        select_device("bogus", "cpu");
        FAIL() << "例外が投げられるべきでした";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("bogus"), std::string::npos)
            << "エラーメッセージにベンダー名が含まれるべきです: " << e.what();
    }
}

// ---------------------------------------------------------------
// 異常系: 無効なデバイスタイプ文字列
// ---------------------------------------------------------------
TEST(SelectDeviceTest, InvalidDeviceTypeThrows) {
    EXPECT_THROW(
        { select_device("intel", "fpga"); },
        std::runtime_error);
}

TEST(SelectDeviceTest, EmptyDeviceTypeThrows) {
    EXPECT_THROW(
        { select_device("intel", ""); },
        std::runtime_error);
}

TEST(SelectDeviceTest, InvalidDeviceTypeErrorMessage) {
    try {
        select_device("intel", "fpga");
        FAIL() << "例外が投げられるべきでした";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("fpga"), std::string::npos)
            << "エラーメッセージにタイプ名が含まれるべきです: " << e.what();
    }
}

// ---------------------------------------------------------------
// 大文字小文字の正規化
// ---------------------------------------------------------------
TEST(SelectDeviceTest, VendorCaseInsensitive) {
#ifdef SYCL_IMPL_ADAPTIVECPP
    if (!g_devices.has_nvidia_gpu) {
        GTEST_SKIP() << "NVIDIA GPU が利用できないためスキップ";
    }
    EXPECT_NO_THROW({ select_device("nvidia", "gpu"); });
    EXPECT_NO_THROW({ select_device("Nvidia", "gpu"); });
    EXPECT_NO_THROW({ select_device("NVIDIA", "gpu"); });
    EXPECT_NO_THROW({ select_device("nViDiA", "gpu"); });
#else  // DPC++
    if (!g_devices.has_intel_cpu) {
        GTEST_SKIP() << "Intel CPU が利用できないためスキップ";
    }
    EXPECT_NO_THROW({ select_device("intel", "cpu"); });
    EXPECT_NO_THROW({ select_device("Intel", "cpu"); });
    EXPECT_NO_THROW({ select_device("INTEL", "cpu"); });
    EXPECT_NO_THROW({ select_device("iNtEl", "cpu"); });
#endif
}

TEST(SelectDeviceTest, DeviceTypeCaseInsensitive) {
#ifdef SYCL_IMPL_ADAPTIVECPP
    if (!g_devices.has_nvidia_gpu) {
        GTEST_SKIP() << "NVIDIA GPU が利用できないためスキップ";
    }
    EXPECT_NO_THROW({ select_device("nvidia", "gpu"); });
    EXPECT_NO_THROW({ select_device("nvidia", "Gpu"); });
    EXPECT_NO_THROW({ select_device("nvidia", "GPU"); });
#else  // DPC++
    if (!g_devices.has_intel_cpu) {
        GTEST_SKIP() << "Intel CPU が利用できないためスキップ";
    }
    EXPECT_NO_THROW({ select_device("intel", "cpu"); });
    EXPECT_NO_THROW({ select_device("intel", "Cpu"); });
    EXPECT_NO_THROW({ select_device("intel", "CPU"); });
#endif
}

// ---------------------------------------------------------------
// 正常系: Intel CPU
// ---------------------------------------------------------------
TEST(SelectDeviceTest, IntelCpuFound) {
    if (!g_devices.has_intel_cpu) {
        GTEST_SKIP() << "Intel CPU が利用できないためスキップ";
    }
    sycl::device dev;
    ASSERT_NO_THROW({ dev = select_device("intel", "cpu"); });
    EXPECT_TRUE(dev.is_cpu());
    EXPECT_EQ(dev.get_info<sycl::info::device::vendor_id>(), VENDOR_ID::INTEL);
}

// ---------------------------------------------------------------
// 正常系: Intel GPU
// ---------------------------------------------------------------
TEST(SelectDeviceTest, IntelGpuFound) {
    if (!g_devices.has_intel_gpu) {
        GTEST_SKIP() << "Intel GPU が利用できないためスキップ";
    }
    sycl::device dev;
    ASSERT_NO_THROW({ dev = select_device("intel", "gpu"); });
    EXPECT_TRUE(dev.is_gpu());
    EXPECT_EQ(dev.get_info<sycl::info::device::vendor_id>(), VENDOR_ID::INTEL);
}

// ---------------------------------------------------------------
// 正常系: NVIDIA GPU
// ---------------------------------------------------------------
TEST(SelectDeviceTest, NvidiaGpuFound) {
    if (!g_devices.has_nvidia_gpu) {
        GTEST_SKIP() << "NVIDIA GPU が利用できないためスキップ";
    }
    sycl::device dev;
    ASSERT_NO_THROW({ dev = select_device("nvidia", "gpu"); });
    EXPECT_TRUE(dev.is_gpu());
    EXPECT_EQ(dev.get_info<sycl::info::device::vendor_id>(), VENDOR_ID::NVIDIA);
}

// ---------------------------------------------------------------
// 異常系: 存在しないデバイスの組み合わせ
// ---------------------------------------------------------------
TEST(SelectDeviceTest, NvidiaGpuNotFoundThrows) {
    if (g_devices.has_nvidia_gpu) {
        GTEST_SKIP() << "NVIDIA GPU が存在するためスキップ";
    }
    EXPECT_THROW({ select_device("nvidia", "gpu"); }, std::runtime_error);
}

TEST(SelectDeviceTest, AmdGpuNotFoundThrows) {
    if (g_devices.has_amd_gpu) {
        GTEST_SKIP() << "AMD GPU が存在するためスキップ";
    }
    EXPECT_THROW({ select_device("amd", "gpu"); }, std::runtime_error);
}

TEST(SelectDeviceTest, IntelGpuNotFoundThrows) {
    if (g_devices.has_intel_gpu) {
        GTEST_SKIP() << "Intel GPU が存在するためスキップ";
    }
    EXPECT_THROW({ select_device("intel", "gpu"); }, std::runtime_error);
}

TEST(SelectDeviceTest, NotFoundErrorMessage) {
    // Intel CPU は通常存在するが、GPU として指定してもエラーになる状況を想定。
    // どちらかが存在しない組み合わせを選ぶ。
    if (g_devices.has_amd_gpu) {
        GTEST_SKIP() << "AMD GPU が存在するためスキップ";
    }
    try {
        select_device("amd", "gpu");
        FAIL() << "例外が投げられるべきでした";
    } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("amd"), std::string::npos) << "エラーメッセージにベンダー名が含まれるべきです: " << msg;
        EXPECT_NE(msg.find("gpu"), std::string::npos) << "エラーメッセージにタイプ名が含まれるべきです: " << msg;
    }
}

// ---------------------------------------------------------------
// 正常系: 返されたデバイスが is_supported_device を満たす
// ---------------------------------------------------------------
TEST(SelectDeviceTest, ReturnedDeviceIsSupported) {
#ifdef SYCL_IMPL_ADAPTIVECPP
    if (!g_devices.has_nvidia_gpu) {
        GTEST_SKIP() << "NVIDIA GPU が利用できないためスキップ";
    }
    const sycl::device dev = select_device("nvidia", "gpu");
#else  // DPC++
    if (!g_devices.has_intel_cpu) {
        GTEST_SKIP() << "Intel CPU が利用できないためスキップ";
    }
    const sycl::device dev = select_device("intel", "cpu");
#endif
    EXPECT_TRUE(is_supported_device(dev));
}

// ---------------------------------------------------------------
// AdaptiveCpp 専用: OMP ベンダー
// ---------------------------------------------------------------
#ifdef SYCL_IMPL_ADAPTIVECPP
TEST(SelectDeviceTest, OmpVendorNotAllowedInInvalidType) {
    // omp ベンダーはデバイスタイプのチェック前に返されるが、
    // 無効なタイプ文字列は先にエラーになる
    EXPECT_THROW({ select_device("omp", "fpga"); }, std::runtime_error);
}

TEST(SelectDeviceTest, OmpVendorFoundAsCpu) {
    if (!g_devices.has_omp) {
        GTEST_SKIP() << "OMP デバイスが利用できないためスキップ";
    }
    // omp バックエンドは is_cpu()/is_gpu() に依存せず即返却される
    sycl::device dev;
    ASSERT_NO_THROW({ dev = select_device("omp", "cpu"); });
    EXPECT_EQ(dev.get_info<sycl::info::device::vendor_id>(), VENDOR_ID::OMP);
    EXPECT_TRUE(is_supported_device(dev));
}

TEST(SelectDeviceTest, OmpVendorCaseInsensitive) {
    if (!g_devices.has_omp) {
        GTEST_SKIP() << "OMP デバイスが利用できないためスキップ";
    }
    EXPECT_NO_THROW({ select_device("omp", "cpu"); });
    EXPECT_NO_THROW({ select_device("OMP", "cpu"); });
    EXPECT_NO_THROW({ select_device("Omp", "cpu"); });
}
#endif  // SYCL_IMPL_ADAPTIVECPP

// ---------------------------------------------------------------
// Level Zero はDPC++・AdaptiveCpp両方でコンパイラがネイティブ対応するが、
// このライブラリでは動作未検証のためサポート外。select_device は両バックエンドで
// Level Zero デバイスをスキップし、代わりに opencl/ocl バックエンドを返す。
// ---------------------------------------------------------------
TEST(SelectDeviceTest, LevelZeroDevicesNotReturned) {
    if (!g_devices.has_intel_gpu) {
        GTEST_SKIP() << "Intel GPU が利用できないためスキップ";
    }
    const sycl::device dev = select_device("intel", "gpu");
#ifdef SYCL_IMPL_INTEL_DPCPP
    EXPECT_NE(dev.get_backend(), sycl::backend::ext_oneapi_level_zero)
        << "level_zero バックエンドのデバイスが返されるべきではありません";
    EXPECT_EQ(dev.get_backend(), sycl::backend::opencl)
        << "opencl バックエンドのデバイスが返されるべきです";
#else  // AdaptiveCpp
    EXPECT_NE(dev.get_backend(), sycl::backend::level_zero)
        << "level_zero バックエンドのデバイスが返されるべきではありません";
    EXPECT_EQ(dev.get_backend(), sycl::backend::ocl)
        << "ocl バックエンドのデバイスが返されるべきです";
#endif
}

}  // namespace test
}  // namespace device_selector
}  // namespace sycl_utils
}  // namespace sycl_points
