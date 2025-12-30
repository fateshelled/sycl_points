#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>

#include "sycl_points/io/point_cloud_reader.hpp"
#include "sycl_points/io/point_cloud_writer.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

class PointCloudIOTest : public ::testing::Test {
protected:
    sycl_points::sycl_utils::DeviceQueue::Ptr queue;
    const float tolerance = 1e-6f;  // Tolerance for floating point comparison

    void SetUp() override {
        try {
            sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
            queue = std::make_shared<sycl_points::sycl_utils::DeviceQueue>(device);
            std::cout << "Using device: " << device.get_info<sycl::info::device::name>() << std::endl;
        } catch (const sycl::exception& e) {
            std::cerr << "SYCL exception caught: " << e.what() << std::endl;
            FAIL() << "Failed to initialize SYCL device";
        }
    }

    // Generate test point cloud data
    sycl_points::PointCloudCPU generateTestData(size_t num_points) {
        sycl_points::PointCloudCPU cloud;

        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducible tests
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

        cloud.points->reserve(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            float x = dist(gen);
            float y = dist(gen);
            float z = dist(gen);
            cloud.points->emplace_back(x, y, z, 1.0f);
        }

        return cloud;
    }

    // Compare two point clouds for equality
    void comparePointClouds(const sycl_points::PointCloudCPU& cloud1, const sycl_points::PointCloudCPU& cloud2) {
        ASSERT_EQ(cloud1.size(), cloud2.size()) << "Point cloud sizes don't match";

        const size_t N = cloud1.size();
        for (size_t i = 0; i < N; ++i) {
            const auto& pt1 = (*cloud1.points)[i];
            const auto& pt2 = (*cloud2.points)[i];

            EXPECT_NEAR(pt1.x(), pt2.x(), tolerance) << "X coordinate mismatch at point " << i;
            EXPECT_NEAR(pt1.y(), pt2.y(), tolerance) << "Y coordinate mismatch at point " << i;
            EXPECT_NEAR(pt1.z(), pt2.z(), tolerance) << "Z coordinate mismatch at point " << i;
            EXPECT_NEAR(pt1.w(), pt2.w(), tolerance) << "W coordinate mismatch at point " << i;
        }
    }

    // Compare CPU and Shared point clouds
    void comparePointClouds(const sycl_points::PointCloudCPU& cpu_cloud,
                            const sycl_points::PointCloudShared& shared_cloud) {
        ASSERT_EQ(cpu_cloud.size(), shared_cloud.size()) << "Point cloud sizes don't match";

        // Set memory access hints for shared cloud
        const size_t N = shared_cloud.size();
        if (N > 0) {
            queue->set_accessed_by_host(shared_cloud.points_ptr(), N);
        }

        for (size_t i = 0; i < N; ++i) {
            const auto& pt1 = (*cpu_cloud.points)[i];
            const auto& pt2 = (*shared_cloud.points)[i];

            EXPECT_NEAR(pt1.x(), pt2.x(), tolerance) << "X coordinate mismatch at point " << i;
            EXPECT_NEAR(pt1.y(), pt2.y(), tolerance) << "Y coordinate mismatch at point " << i;
            EXPECT_NEAR(pt1.z(), pt2.z(), tolerance) << "Z coordinate mismatch at point " << i;
            EXPECT_NEAR(pt1.w(), pt2.w(), tolerance) << "W coordinate mismatch at point " << i;
        }

        // Clear memory access hints
        if (N > 0) {
            queue->clear_accessed_by_host(shared_cloud.points_ptr(), N);
        }
    }

    // Test round-trip I/O for CPU point clouds
    void testRoundTripCPU(const std::string& filename, bool binary = false) {
        // Generate test data
        auto original_cloud = generateTestData(1000);

        // Write to file
        sycl_points::PointCloudWriter::writeFile(filename, original_cloud, binary);

        // Read back from file
        auto loaded_cloud = sycl_points::PointCloudReader::readFile(filename);

        // Compare
        comparePointClouds(original_cloud, loaded_cloud);

        // Cleanup
        std::remove(filename.c_str());
    }

    // Test round-trip I/O for Shared point clouds
    void testRoundTripShared(const std::string& filename, bool binary = false) {
        // Generate test data and convert to shared
        auto original_cpu_cloud = generateTestData(1000);
        sycl_points::PointCloudShared original_shared_cloud(*queue, original_cpu_cloud);

        // Write to file
        sycl_points::PointCloudWriter::writeFile(filename, original_shared_cloud, binary);

        // Read back from file (CPU version)
        auto loaded_cpu_cloud = sycl_points::PointCloudReader::readFile(filename);

        // Read back from file (Shared version)
        auto loaded_shared_cloud = sycl_points::PointCloudReader::readFile(filename, *queue);

        // Compare original CPU with loaded CPU
        comparePointClouds(original_cpu_cloud, loaded_cpu_cloud);

        // Compare original CPU with loaded Shared
        comparePointClouds(original_cpu_cloud, loaded_shared_cloud);

        // Compare loaded CPU with loaded Shared
        comparePointClouds(loaded_cpu_cloud, loaded_shared_cloud);

        // Cleanup
        std::remove(filename.c_str());
    }

    // Test cross-format compatibility (read with different readers)
    void testCrossFormat(const std::string& filename, bool binary = false) {
        // Generate test data
        auto original_cloud = generateTestData(500);

        // Write with CPU interface
        sycl_points::PointCloudWriter::writeFile(filename, original_cloud, binary);

        // Read with both CPU and Shared interfaces
        auto loaded_cpu_cloud = sycl_points::PointCloudReader::readFile(filename);
        auto loaded_shared_cloud = sycl_points::PointCloudReader::readFile(filename, *queue);

        // Compare all combinations
        comparePointClouds(original_cloud, loaded_cpu_cloud);
        comparePointClouds(original_cloud, loaded_shared_cloud);
        comparePointClouds(loaded_cpu_cloud, loaded_shared_cloud);

        // Cleanup
        std::remove(filename.c_str());
    }

    // Test with existing files
    void testExistingFileRoundTrip(const std::string& input_file, const std::string& temp_file, bool binary = false) {
        // Check if input file exists
        std::ifstream check_file(input_file);
        if (!check_file.good()) {
            GTEST_SKIP() << "Input file not found: " << input_file;
            return;
        }
        check_file.close();

        // Read original file
        auto original_cloud = sycl_points::PointCloudReader::readFile(input_file);

        if (original_cloud.size() == 0) {
            GTEST_SKIP() << "Input file is empty: " << input_file;
            return;
        }

        // Write to temp file
        sycl_points::PointCloudWriter::writeFile(temp_file, original_cloud, binary);

        // Read back from temp file
        auto reloaded_cloud = sycl_points::PointCloudReader::readFile(temp_file);

        // Compare
        comparePointClouds(original_cloud, reloaded_cloud);

        // Cleanup
        std::remove(temp_file.c_str());
    }

    // Test format conversion
    void testFormatConversion(const std::string& input_file, const std::string& output_ext, bool binary = false) {
        // Check if input file exists
        std::ifstream check_file(input_file);
        if (!check_file.good()) {
            GTEST_SKIP() << "Input file not found: " << input_file;
            return;
        }
        check_file.close();

        // Read original file
        auto original_cloud = sycl_points::PointCloudReader::readFile(input_file);

        if (original_cloud.size() == 0) {
            GTEST_SKIP() << "Input file is empty: " << input_file;
            return;
        }

        // Generate output filename
        std::string output_file = "converted_" + std::to_string(std::rand()) + output_ext;

        // Write in different format
        sycl_points::PointCloudWriter::writeFile(output_file, original_cloud, binary);

        // Read back converted file
        auto converted_cloud = sycl_points::PointCloudReader::readFile(output_file);

        // Compare
        comparePointClouds(original_cloud, converted_cloud);

        // Cleanup
        std::remove(output_file.c_str());
    }
};

// Test PLY format - ASCII mode
TEST_F(PointCloudIOTest, PLY_ASCII_CPU_RoundTrip) { testRoundTripCPU("test_ply_ascii.ply", false); }

TEST_F(PointCloudIOTest, PLY_ASCII_Shared_RoundTrip) { testRoundTripShared("test_ply_ascii_shared.ply", false); }

TEST_F(PointCloudIOTest, PLY_ASCII_CrossFormat) { testCrossFormat("test_ply_ascii_cross.ply", false); }

// Test PLY format - Binary mode
TEST_F(PointCloudIOTest, PLY_Binary_CPU_RoundTrip) { testRoundTripCPU("test_ply_binary.ply", true); }

TEST_F(PointCloudIOTest, PLY_Binary_Shared_RoundTrip) { testRoundTripShared("test_ply_binary_shared.ply", true); }

TEST_F(PointCloudIOTest, PLY_Binary_CrossFormat) { testCrossFormat("test_ply_binary_cross.ply", true); }

// Test PCD format - ASCII mode
TEST_F(PointCloudIOTest, PCD_ASCII_CPU_RoundTrip) { testRoundTripCPU("test_pcd_ascii.pcd", false); }

TEST_F(PointCloudIOTest, PCD_ASCII_Shared_RoundTrip) { testRoundTripShared("test_pcd_ascii_shared.pcd", false); }

TEST_F(PointCloudIOTest, PCD_ASCII_CrossFormat) { testCrossFormat("test_pcd_ascii_cross.pcd", false); }

// Test PCD format - Binary mode
TEST_F(PointCloudIOTest, PCD_Binary_CPU_RoundTrip) { testRoundTripCPU("test_pcd_binary.pcd", true); }

TEST_F(PointCloudIOTest, PCD_Binary_Shared_RoundTrip) { testRoundTripShared("test_pcd_binary_shared.pcd", true); }

TEST_F(PointCloudIOTest, PCD_Binary_CrossFormat) { testCrossFormat("test_pcd_binary_cross.pcd", true); }

TEST_F(PointCloudIOTest, PLY_ASCII_WithRGB) {
    const std::string filename = "test_rgb_ascii.ply";
    {
        std::ofstream file(filename);
        file << "ply\nformat ascii 1.0\n";
        file << "element vertex 2\n";
        file << "property float x\nproperty float y\nproperty float z\n";
        file << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
        file << "end_header\n";
        file << "0 0 0 255 0 0\n";
        file << "1 2 3 0 255 128\n";
    }
    auto cloud = sycl_points::PointCloudReader::readFile(filename);
    ASSERT_TRUE(cloud.has_rgb());
    ASSERT_EQ(cloud.size(), 2u);
    EXPECT_NEAR((*cloud.rgb)[0].x(), 1.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[0].y(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[0].z(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].x(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].y(), 1.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].z(), 128.f / 255.f, tolerance);
    std::remove(filename.c_str());
}

TEST_F(PointCloudIOTest, PLY_Binary_WithRGB) {
    const std::string filename = "test_rgb_binary.ply";
    {
        std::ofstream file(filename, std::ios::binary);
        file << "ply\nformat binary_little_endian 1.0\n";
        file << "element vertex 2\n";
        file << "property float x\nproperty float y\nproperty float z\n";
        file << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
        file << "end_header\n";
        float x0 = 0.f, y0 = 0.f, z0 = 0.f;
        uint8_t r0 = 255, g0 = 0, b0 = 0;
        file.write(reinterpret_cast<char*>(&x0), sizeof(float));
        file.write(reinterpret_cast<char*>(&y0), sizeof(float));
        file.write(reinterpret_cast<char*>(&z0), sizeof(float));
        file.write(reinterpret_cast<char*>(&r0), sizeof(uint8_t));
        file.write(reinterpret_cast<char*>(&g0), sizeof(uint8_t));
        file.write(reinterpret_cast<char*>(&b0), sizeof(uint8_t));
        float x1 = 1.f, y1 = 2.f, z1 = 3.f;
        uint8_t r1 = 0, g1 = 255, b1 = 128;
        file.write(reinterpret_cast<char*>(&x1), sizeof(float));
        file.write(reinterpret_cast<char*>(&y1), sizeof(float));
        file.write(reinterpret_cast<char*>(&z1), sizeof(float));
        file.write(reinterpret_cast<char*>(&r1), sizeof(uint8_t));
        file.write(reinterpret_cast<char*>(&g1), sizeof(uint8_t));
        file.write(reinterpret_cast<char*>(&b1), sizeof(uint8_t));
    }
    auto cloud = sycl_points::PointCloudReader::readFile(filename);
    ASSERT_TRUE(cloud.has_rgb());
    ASSERT_EQ(cloud.size(), 2u);
    EXPECT_NEAR((*cloud.rgb)[0].x(), 1.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[0].y(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[0].z(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].x(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].y(), 1.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].z(), 128.f / 255.f, tolerance);
    std::remove(filename.c_str());
}

TEST_F(PointCloudIOTest, PCD_ASCII_WithRGB) {
    const std::string filename = "test_rgb_ascii.pcd";
    {
        std::ofstream file(filename);
        file << "# .PCD v0.7 - Point Cloud Data file format\n";
        file << "FIELDS x y z r g b\n";
        file << "SIZE 4 4 4 1 1 1\n";
        file << "TYPE F F F U U U\n";
        file << "COUNT 1 1 1 1 1 1\n";
        file << "WIDTH 2\nHEIGHT 1\n";
        file << "POINTS 2\nDATA ascii\n";
        file << "0 0 0 255 0 0\n";
        file << "1 2 3 0 255 128\n";
    }
    auto cloud = sycl_points::PointCloudReader::readFile(filename);
    ASSERT_TRUE(cloud.has_rgb());
    ASSERT_EQ(cloud.size(), 2u);
    EXPECT_NEAR((*cloud.rgb)[0].x(), 1.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[0].y(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[0].z(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].x(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].y(), 1.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].z(), 128.f / 255.f, tolerance);
    std::remove(filename.c_str());
}

TEST_F(PointCloudIOTest, PCD_Binary_WithRGB) {
    const std::string filename = "test_rgb_binary.pcd";
    {
        std::ofstream file(filename, std::ios::binary);
        file << "# .PCD v0.7 - Point Cloud Data file format\n";
        file << "FIELDS x y z r g b\n";
        file << "SIZE 4 4 4 1 1 1\n";
        file << "TYPE F F F U U U\n";
        file << "COUNT 1 1 1 1 1 1\n";
        file << "WIDTH 2\nHEIGHT 1\n";
        file << "POINTS 2\nDATA binary\n";
        float x0 = 0.f, y0 = 0.f, z0 = 0.f;
        uint8_t r0 = 255, g0 = 0, b0 = 0;
        file.write(reinterpret_cast<char*>(&x0), sizeof(float));
        file.write(reinterpret_cast<char*>(&y0), sizeof(float));
        file.write(reinterpret_cast<char*>(&z0), sizeof(float));
        file.write(reinterpret_cast<char*>(&r0), sizeof(uint8_t));
        file.write(reinterpret_cast<char*>(&g0), sizeof(uint8_t));
        file.write(reinterpret_cast<char*>(&b0), sizeof(uint8_t));
        float x1 = 1.f, y1 = 2.f, z1 = 3.f;
        uint8_t r1 = 0, g1 = 255, b1 = 128;
        file.write(reinterpret_cast<char*>(&x1), sizeof(float));
        file.write(reinterpret_cast<char*>(&y1), sizeof(float));
        file.write(reinterpret_cast<char*>(&z1), sizeof(float));
        file.write(reinterpret_cast<char*>(&r1), sizeof(uint8_t));
        file.write(reinterpret_cast<char*>(&g1), sizeof(uint8_t));
        file.write(reinterpret_cast<char*>(&b1), sizeof(uint8_t));
    }
    auto cloud = sycl_points::PointCloudReader::readFile(filename);
    ASSERT_TRUE(cloud.has_rgb());
    ASSERT_EQ(cloud.size(), 2u);
    EXPECT_NEAR((*cloud.rgb)[0].x(), 1.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[0].y(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[0].z(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].x(), 0.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].y(), 1.0f, tolerance);
    EXPECT_NEAR((*cloud.rgb)[1].z(), 128.f / 255.f, tolerance);
    std::remove(filename.c_str());
}

// Test edge cases
TEST_F(PointCloudIOTest, EmptyPointCloud) {
    sycl_points::PointCloudCPU empty_cloud;

    // Writing empty cloud should throw exception
    EXPECT_THROW(sycl_points::PointCloudWriter::writeFile("empty.ply", empty_cloud, false), std::runtime_error);
}

TEST_F(PointCloudIOTest, SinglePoint) {
    sycl_points::PointCloudCPU single_cloud;
    single_cloud.points->emplace_back(1.23f, 4.56f, 7.89f, 1.0f);

    // Test PLY
    sycl_points::PointCloudWriter::writeFile("single.ply", single_cloud, false);
    auto loaded_ply = sycl_points::PointCloudReader::readFile("single.ply");
    comparePointClouds(single_cloud, loaded_ply);
    std::remove("single.ply");

    // Test PCD
    sycl_points::PointCloudWriter::writeFile("single.pcd", single_cloud, false);
    auto loaded_pcd = sycl_points::PointCloudReader::readFile("single.pcd");
    comparePointClouds(single_cloud, loaded_pcd);
    std::remove("single.pcd");
}

TEST_F(PointCloudIOTest, LargePointCloud) {
    // Test with a larger point cloud (10,000 points)
    auto large_cloud = generateTestData(10000);

    // Test PLY binary (more efficient for large data)
    sycl_points::PointCloudWriter::writeFile("large.ply", large_cloud, true);
    auto loaded_large = sycl_points::PointCloudReader::readFile("large.ply");
    comparePointClouds(large_cloud, loaded_large);
    std::remove("large.ply");
}

TEST_F(PointCloudIOTest, InvalidFileFormat) {
    auto test_cloud = generateTestData(100);

    // Unsupported format should throw exception
    EXPECT_THROW(sycl_points::PointCloudWriter::writeFile("test.xyz", test_cloud, false), std::runtime_error);
}

TEST_F(PointCloudIOTest, NonExistentFile) {
    // Reading non-existent file should throw exception
    EXPECT_THROW(sycl_points::PointCloudReader::readFile("non_existent.ply"), std::runtime_error);
}

// Test existing PLY files
TEST_F(PointCloudIOTest, ExistingPLY_ASCII_RoundTrip) {
    testExistingFileRoundTrip("../data/source.ply", "test_existing_source_ascii.ply", false);
    testExistingFileRoundTrip("../data/target.ply", "test_existing_target_ascii.ply", false);
}

TEST_F(PointCloudIOTest, ExistingPLY_Binary_RoundTrip) {
    testExistingFileRoundTrip("../data/source.ply", "test_existing_source_binary.ply", true);
    testExistingFileRoundTrip("../data/target.ply", "test_existing_target_binary.ply", true);
}

// Test format conversion PLY -> PCD
TEST_F(PointCloudIOTest, FormatConversion_PLY_to_PCD_ASCII) {
    testFormatConversion("../data/source.ply", ".pcd", false);
    testFormatConversion("../data/target.ply", ".pcd", false);
}

TEST_F(PointCloudIOTest, FormatConversion_PLY_to_PCD_Binary) {
    testFormatConversion("../data/source.ply", ".pcd", true);
    testFormatConversion("../data/target.ply", ".pcd", true);
}

// Test multiple existing files in directory
TEST_F(PointCloudIOTest, BatchExistingFiles) {
    std::vector<std::string> test_files = {"../data/source.ply", "../data/target.ply"};

    for (const auto& file : test_files) {
        std::ifstream check_file(file);
        if (!check_file.good()) {
            std::cout << "Skipping non-existent file: " << file << std::endl;
            continue;
        }
        check_file.close();

        try {
            // Test ASCII round-trip
            std::string temp_ascii =
                "batch_ascii_" + std::to_string(std::rand()) + "_temp" + file.substr(file.find_last_of('.'));
            testExistingFileRoundTrip(file, temp_ascii, false);

            // Test Binary round-trip
            std::string temp_binary =
                "batch_binary_" + std::to_string(std::rand()) + "_temp" + file.substr(file.find_last_of('.'));
            testExistingFileRoundTrip(file, temp_binary, true);

            std::cout << "Successfully tested file: " << file << std::endl;
        } catch (const std::exception& e) {
            FAIL() << "Failed to process file " << file << ": " << e.what();
        }
    }
}

// Test existing files with Shared memory interface
TEST_F(PointCloudIOTest, ExistingFiles_SharedInterface) {
    std::vector<std::string> test_files = {"../data/source.ply", "../data/target.ply"};

    for (const auto& file : test_files) {
        std::ifstream check_file(file);
        if (!check_file.good()) {
            std::cout << "Skipping non-existent file: " << file << std::endl;
            continue;
        }
        check_file.close();

        try {
            // Read with CPU interface
            auto cpu_cloud = sycl_points::PointCloudReader::readFile(file);

            // Read with Shared interface
            auto shared_cloud = sycl_points::PointCloudReader::readFile(file, *queue);

            // Compare CPU and Shared versions
            comparePointClouds(cpu_cloud, shared_cloud);

            // Write Shared cloud to temporary file
            std::string temp_file = "shared_temp_" + std::to_string(std::rand()) + file.substr(file.find_last_of('.'));
            sycl_points::PointCloudWriter::writeFile(temp_file, shared_cloud, false);

            // Read back and compare
            auto reloaded_cpu = sycl_points::PointCloudReader::readFile(temp_file);
            comparePointClouds(cpu_cloud, reloaded_cpu);

            // Cleanup
            std::remove(temp_file.c_str());

            std::cout << "Successfully tested shared interface with file: " << file << std::endl;
        } catch (const std::exception& e) {
            FAIL() << "Failed to process file with shared interface " << file << ": " << e.what();
        }
    }
}

// Performance comparison test
TEST_F(PointCloudIOTest, PerformanceComparison) {
    const size_t num_points = 50000;
    auto test_cloud = generateTestData(num_points);

    auto start = std::chrono::high_resolution_clock::now();

    // Test ASCII performance
    sycl_points::PointCloudWriter::writeFile("perf_ascii.ply", test_cloud, false);
    auto ascii_write_time = std::chrono::high_resolution_clock::now();

    auto loaded_ascii = sycl_points::PointCloudReader::readFile("perf_ascii.ply");
    auto ascii_read_time = std::chrono::high_resolution_clock::now();

    // Test Binary performance
    sycl_points::PointCloudWriter::writeFile("perf_binary.ply", test_cloud, true);
    auto binary_write_time = std::chrono::high_resolution_clock::now();

    auto loaded_binary = sycl_points::PointCloudReader::readFile("perf_binary.ply");
    auto binary_read_time = std::chrono::high_resolution_clock::now();

    // Verify correctness
    comparePointClouds(test_cloud, loaded_ascii);
    comparePointClouds(test_cloud, loaded_binary);

    // Calculate times
    auto ascii_write_us = std::chrono::duration_cast<std::chrono::microseconds>(ascii_write_time - start).count();
    auto ascii_read_us =
        std::chrono::duration_cast<std::chrono::microseconds>(ascii_read_time - ascii_write_time).count();
    auto binary_write_us =
        std::chrono::duration_cast<std::chrono::microseconds>(binary_write_time - ascii_read_time).count();
    auto binary_read_us =
        std::chrono::duration_cast<std::chrono::microseconds>(binary_read_time - binary_write_time).count();

    std::cout << "Performance results for " << num_points << " points:\n";
    std::cout << "ASCII  Write: " << ascii_write_us << "us, Read: " << ascii_read_us << "us\n";
    std::cout << "Binary Write: " << binary_write_us << "us, Read: " << binary_read_us << "us\n";

    // Cleanup
    std::remove("perf_ascii.ply");
    std::remove("perf_binary.ply");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
