#include <gtest/gtest.h>

#include <string>

#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace sycl_utils {
namespace device_selector {
namespace test {

// Helper to dynamically enumerate available devices
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
// Error cases: invalid vendor string
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
        FAIL() << "Expected exception to be thrown";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("bogus"), std::string::npos)
            << "Error message should contain vendor name: " << e.what();
    }
}

// ---------------------------------------------------------------
// Error cases: invalid device type string
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
        FAIL() << "Expected exception to be thrown";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("fpga"), std::string::npos)
            << "Error message should contain device type: " << e.what();
    }
}

// ---------------------------------------------------------------
// Case normalization
// ---------------------------------------------------------------
TEST(SelectDeviceTest, VendorCaseInsensitive) {
#ifdef SYCL_IMPL_ADAPTIVECPP
    if (!g_devices.has_nvidia_gpu) {
        GTEST_SKIP() << "NVIDIA GPU not available";
    }
    EXPECT_NO_THROW({ select_device("nvidia", "gpu"); });
    EXPECT_NO_THROW({ select_device("Nvidia", "gpu"); });
    EXPECT_NO_THROW({ select_device("NVIDIA", "gpu"); });
    EXPECT_NO_THROW({ select_device("nViDiA", "gpu"); });
#else  // DPC++
    if (!g_devices.has_intel_cpu) {
        GTEST_SKIP() << "Intel CPU not available";
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
        GTEST_SKIP() << "NVIDIA GPU not available";
    }
    EXPECT_NO_THROW({ select_device("nvidia", "gpu"); });
    EXPECT_NO_THROW({ select_device("nvidia", "Gpu"); });
    EXPECT_NO_THROW({ select_device("nvidia", "GPU"); });
#else  // DPC++
    if (!g_devices.has_intel_cpu) {
        GTEST_SKIP() << "Intel CPU not available";
    }
    EXPECT_NO_THROW({ select_device("intel", "cpu"); });
    EXPECT_NO_THROW({ select_device("intel", "Cpu"); });
    EXPECT_NO_THROW({ select_device("intel", "CPU"); });
#endif
}

// ---------------------------------------------------------------
// Normal cases: Intel CPU
// ---------------------------------------------------------------
TEST(SelectDeviceTest, IntelCpuFound) {
    if (!g_devices.has_intel_cpu) {
        GTEST_SKIP() << "Intel CPU not available";
    }
    sycl::device dev;
    ASSERT_NO_THROW({ dev = select_device("intel", "cpu"); });
    EXPECT_TRUE(dev.is_cpu());
    EXPECT_EQ(dev.get_info<sycl::info::device::vendor_id>(), VENDOR_ID::INTEL);
}

// ---------------------------------------------------------------
// Normal cases: Intel GPU
// ---------------------------------------------------------------
TEST(SelectDeviceTest, IntelGpuFound) {
    if (!g_devices.has_intel_gpu) {
        GTEST_SKIP() << "Intel GPU not available";
    }
    sycl::device dev;
    ASSERT_NO_THROW({ dev = select_device("intel", "gpu"); });
    EXPECT_TRUE(dev.is_gpu());
    EXPECT_EQ(dev.get_info<sycl::info::device::vendor_id>(), VENDOR_ID::INTEL);
}

// ---------------------------------------------------------------
// Normal cases: NVIDIA GPU
// ---------------------------------------------------------------
TEST(SelectDeviceTest, NvidiaGpuFound) {
    if (!g_devices.has_nvidia_gpu) {
        GTEST_SKIP() << "NVIDIA GPU not available";
    }
    sycl::device dev;
    ASSERT_NO_THROW({ dev = select_device("nvidia", "gpu"); });
    EXPECT_TRUE(dev.is_gpu());
    EXPECT_EQ(dev.get_info<sycl::info::device::vendor_id>(), VENDOR_ID::NVIDIA);
}

// ---------------------------------------------------------------
// Error cases: device not found
// ---------------------------------------------------------------
TEST(SelectDeviceTest, NvidiaGpuNotFoundThrows) {
    if (g_devices.has_nvidia_gpu) {
        GTEST_SKIP() << "NVIDIA GPU is present";
    }
    EXPECT_THROW({ select_device("nvidia", "gpu"); }, std::runtime_error);
}

TEST(SelectDeviceTest, AmdGpuNotFoundThrows) {
    if (g_devices.has_amd_gpu) {
        GTEST_SKIP() << "AMD GPU is present";
    }
    EXPECT_THROW({ select_device("amd", "gpu"); }, std::runtime_error);
}

TEST(SelectDeviceTest, IntelGpuNotFoundThrows) {
    if (g_devices.has_intel_gpu) {
        GTEST_SKIP() << "Intel GPU is present";
    }
    EXPECT_THROW({ select_device("intel", "gpu"); }, std::runtime_error);
}

TEST(SelectDeviceTest, NotFoundErrorMessage) {
    // Use a combination that is unlikely to exist
    if (g_devices.has_amd_gpu) {
        GTEST_SKIP() << "AMD GPU is present";
    }
    try {
        select_device("amd", "gpu");
        FAIL() << "Expected exception to be thrown";
    } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("amd"), std::string::npos) << "Error message should contain vendor name: " << msg;
        EXPECT_NE(msg.find("gpu"), std::string::npos) << "Error message should contain device type: " << msg;
    }
}

// ---------------------------------------------------------------
// Normal cases: returned device satisfies is_supported_device
// ---------------------------------------------------------------
TEST(SelectDeviceTest, ReturnedDeviceIsSupported) {
#ifdef SYCL_IMPL_ADAPTIVECPP
    if (!g_devices.has_nvidia_gpu) {
        GTEST_SKIP() << "NVIDIA GPU not available";
    }
    const sycl::device dev = select_device("nvidia", "gpu");
#else  // DPC++
    if (!g_devices.has_intel_cpu) {
        GTEST_SKIP() << "Intel CPU not available";
    }
    const sycl::device dev = select_device("intel", "cpu");
#endif
    EXPECT_TRUE(is_supported_device(dev));
}

// ---------------------------------------------------------------
// AdaptiveCpp only: OMP vendor
// ---------------------------------------------------------------
#ifdef SYCL_IMPL_ADAPTIVECPP
TEST(SelectDeviceTest, IntelCpuFallsBackToOmp) {
    if (!g_devices.has_omp) {
        GTEST_SKIP() << "OMP device not available";
    }
    // AdaptiveCpp: CPU vendor request falls back to OMP backend
    sycl::device dev;
    ASSERT_NO_THROW({ dev = select_device("intel", "cpu"); });
    EXPECT_EQ(dev.get_info<sycl::info::device::vendor_id>(), VENDOR_ID::OMP);
    EXPECT_EQ(dev.get_backend(), sycl::backend::omp);
    EXPECT_TRUE(is_supported_device(dev));
}

TEST(SelectDeviceTest, OmpVendorNotAllowedInInvalidType) {
    // Invalid device type is rejected before the OMP vendor check
    EXPECT_THROW({ select_device("omp", "fpga"); }, std::runtime_error);
}

TEST(SelectDeviceTest, OmpVendorFoundAsCpu) {
    if (!g_devices.has_omp) {
        GTEST_SKIP() << "OMP device not available";
    }
    // OMP backend device is returned regardless of is_cpu()/is_gpu()
    sycl::device dev;
    ASSERT_NO_THROW({ dev = select_device("omp", "cpu"); });
    EXPECT_EQ(dev.get_info<sycl::info::device::vendor_id>(), VENDOR_ID::OMP);
    EXPECT_TRUE(is_supported_device(dev));
}

TEST(SelectDeviceTest, OmpVendorCaseInsensitive) {
    if (!g_devices.has_omp) {
        GTEST_SKIP() << "OMP device not available";
    }
    EXPECT_NO_THROW({ select_device("omp", "cpu"); });
    EXPECT_NO_THROW({ select_device("OMP", "cpu"); });
    EXPECT_NO_THROW({ select_device("Omp", "cpu"); });
}
#endif  // SYCL_IMPL_ADAPTIVECPP

// ---------------------------------------------------------------
// Level Zero is natively supported by both DPC++ and AdaptiveCpp compilers,
// but is untested in this library and therefore excluded.
// select_device skips Level Zero devices and returns opencl/ocl backend instead.
// ---------------------------------------------------------------
TEST(SelectDeviceTest, LevelZeroDevicesNotReturned) {
    if (!g_devices.has_intel_gpu) {
        GTEST_SKIP() << "Intel GPU not available";
    }
    const sycl::device dev = select_device("intel", "gpu");
#ifdef SYCL_IMPL_INTEL_DPCPP
    EXPECT_NE(dev.get_backend(), sycl::backend::ext_oneapi_level_zero)
        << "level_zero backend device should not be returned";
    EXPECT_EQ(dev.get_backend(), sycl::backend::opencl)
        << "opencl backend device should be returned";
#else  // AdaptiveCpp
    EXPECT_NE(dev.get_backend(), sycl::backend::level_zero)
        << "level_zero backend device should not be returned";
    EXPECT_EQ(dev.get_backend(), sycl::backend::ocl)
        << "ocl backend device should be returned";
#endif
}

}  // namespace test
}  // namespace device_selector
}  // namespace sycl_utils
}  // namespace sycl_points
