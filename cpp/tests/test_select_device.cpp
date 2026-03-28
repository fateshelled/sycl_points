#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "sycl_points/utils/sycl_utils.hpp"

namespace sycl_points {
namespace sycl_utils {
namespace device_selector {
namespace test {

// Helper to dynamically enumerate available devices.
// DPC++: cpu_vendor holds the first available CPU vendor string ("intel", "amd", etc.)
// AdaptiveCpp: any CPU vendor maps to OMP backend, so cpu_vendor is not needed.
struct AvailableDevices {
#ifdef SYCL_IMPL_INTEL_DPCPP
    std::string cpu_vendor;  // first available CPU vendor string, empty if none
    bool has_cpu() const { return !cpu_vendor.empty(); }
#endif
    bool has_intel_gpu = false;
    bool has_nvidia_gpu = false;
    bool has_amd_gpu = false;
#ifdef SYCL_IMPL_ADAPTIVECPP
    bool has_omp = false;  // OMP backend handles all CPU vendors
#endif

    AvailableDevices() {
#ifdef SYCL_IMPL_INTEL_DPCPP
        // Map vendor_id to the string accepted by select_device()
        static const std::pair<uint32_t, const char*> kCpuVendors[] = {
            {VENDOR_ID::INTEL, "intel"},
            {VENDOR_ID::AMD, "amd"},
        };
#endif
        for (auto platform : sycl::platform::get_platforms()) {
            for (auto device : platform.get_devices()) {
                if (!is_supported_device(device)) continue;
                const auto vid = device.get_info<sycl::info::device::vendor_id>();
#ifdef SYCL_IMPL_INTEL_DPCPP
                if (cpu_vendor.empty() && device.is_cpu()) {
                    for (const auto& [vid_val, name] : kCpuVendors) {
                        if (vid == vid_val) { cpu_vendor = name; break; }
                    }
                }
#endif
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
    if (!g_devices.has_cpu()) {
        GTEST_SKIP() << "No CPU device available";
    }
    const std::string& v = g_devices.cpu_vendor;
    std::string upper = v;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    std::string capitalized = v;
    capitalized[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(capitalized[0])));

    EXPECT_NO_THROW({ select_device(v, "cpu"); });
    EXPECT_NO_THROW({ select_device(upper, "cpu"); });
    EXPECT_NO_THROW({ select_device(capitalized, "cpu"); });
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
    if (!g_devices.has_cpu()) {
        GTEST_SKIP() << "No CPU device available";
    }
    const std::string& v = g_devices.cpu_vendor;
    EXPECT_NO_THROW({ select_device(v, "cpu"); });
    EXPECT_NO_THROW({ select_device(v, "Cpu"); });
    EXPECT_NO_THROW({ select_device(v, "CPU"); });
#endif
}

// ---------------------------------------------------------------
// Normal cases: CPU (DPC++ only — vendor-agnostic)
// AdaptiveCpp CPU is covered by AnyCpuVendorResolvesToOmp below.
// ---------------------------------------------------------------
#ifdef SYCL_IMPL_INTEL_DPCPP
TEST(SelectDeviceTest, CpuFound) {
    if (!g_devices.has_cpu()) {
        GTEST_SKIP() << "No CPU device available";
    }
    sycl::device dev;
    ASSERT_NO_THROW({ dev = select_device(g_devices.cpu_vendor, "cpu"); });
    EXPECT_TRUE(dev.is_cpu());
    EXPECT_EQ(dev.get_backend(), sycl::backend::opencl)
        << "DPC++: CPU should use opencl backend";
}
#endif

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
#ifdef SYCL_IMPL_INTEL_DPCPP
    EXPECT_EQ(dev.get_backend(), sycl::backend::ext_oneapi_cuda)
        << "DPC++: NVIDIA GPU should use cuda backend";
#else  // AdaptiveCpp
    EXPECT_EQ(dev.get_backend(), sycl::backend::cuda)
        << "AdaptiveCpp: NVIDIA GPU should use cuda backend";
#endif
}

// ---------------------------------------------------------------
// Normal cases: AMD GPU
// ---------------------------------------------------------------
TEST(SelectDeviceTest, AmdGpuFound) {
    if (!g_devices.has_amd_gpu) {
        GTEST_SKIP() << "AMD GPU not available";
    }
    sycl::device dev;
    ASSERT_NO_THROW({ dev = select_device("amd", "gpu"); });
    EXPECT_TRUE(dev.is_gpu());
    EXPECT_EQ(dev.get_info<sycl::info::device::vendor_id>(), VENDOR_ID::AMD);
#ifdef SYCL_IMPL_INTEL_DPCPP
    EXPECT_EQ(dev.get_backend(), sycl::backend::ext_oneapi_hip)
        << "DPC++: AMD GPU should use hip backend";
#else  // AdaptiveCpp
    EXPECT_EQ(dev.get_backend(), sycl::backend::hip)
        << "AdaptiveCpp: AMD GPU should use hip backend";
#endif
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
    if (!g_devices.has_cpu()) {
        GTEST_SKIP() << "No CPU device available";
    }
    const sycl::device dev = select_device(g_devices.cpu_vendor, "cpu");
#endif
    EXPECT_TRUE(is_supported_device(dev));
}

// ---------------------------------------------------------------
// AdaptiveCpp only: OMP backend handles all CPU vendors.
// In AdaptiveCpp, the OMP backend runs on any CPU regardless of vendor
// (Intel, AMD, Qualcomm, ARM, etc.). All recognized vendor strings
// resolve to the same OMP device.
// ---------------------------------------------------------------
#ifdef SYCL_IMPL_ADAPTIVECPP
TEST(SelectDeviceTest, AnyCpuVendorResolvesToOmp) {
    if (!g_devices.has_omp) {
        GTEST_SKIP() << "OMP device not available";
    }
    // All CPU vendor requests resolve to the OMP backend device
    for (const char* vendor : {"intel", "amd"}) {
        sycl::device dev;
        ASSERT_NO_THROW({ dev = select_device(vendor, "cpu"); }) << "vendor: " << vendor;
        EXPECT_EQ(dev.get_info<sycl::info::device::vendor_id>(), VENDOR_ID::OMP) << "vendor: " << vendor;
        EXPECT_EQ(dev.get_backend(), sycl::backend::omp) << "vendor: " << vendor;
        EXPECT_TRUE(is_supported_device(dev)) << "vendor: " << vendor;
    }
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
