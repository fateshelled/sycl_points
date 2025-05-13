#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <sycl_points/algorithms/sort/bitonic_sort.hpp>
#include <vector>

// Setup for the SYCL environment
class BitonicSortTest : public ::testing::Test {
protected:
    sycl_points::sycl_utils::DeviceQueue::Ptr queue;

    void SetUp() override {
        // Setup for SYCL device and queue
        try {
            sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
            queue = std::make_shared<sycl_points::sycl_utils::DeviceQueue>(device);
            std::cout << "Using device: " << device.get_info<sycl::info::device::name>() << std::endl;
        } catch (const sycl::exception& e) {
            std::cerr << "SYCL exception caught: " << e.what() << std::endl;
            FAIL() << "Failed to initialize SYCL device";
        }
    }

    void TearDown() override {}

    // Helper function to generate random integer array (for int32_t)
    std::vector<int32_t> generateRandomInt32(size_t size, int32_t min = -1000, int32_t max = 1000) {
        std::vector<int32_t> result(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int32_t> dist(min, max);

        for (size_t i = 0; i < size; ++i) {
            result[i] = dist(gen);
        }
        return result;
    }

    // Helper function to generate random integer array (for uint64_t)
    std::vector<uint64_t> generateRandomUInt64(size_t size, uint64_t min = 0, uint64_t max = 1000000) {
        std::vector<uint64_t> result(size);
        std::random_device rd;
        std::mt19937_64 gen(rd());  // 64-bit version of Mersenne Twister
        std::uniform_int_distribution<uint64_t> dist(min, max);

        for (size_t i = 0; i < size; ++i) {
            result[i] = dist(gen);
        }
        return result;
    }

    // Helper function to generate random floating point array
    std::vector<float> generateRandomFloats(size_t size, float min = -1000.0f, float max = 1000.0f) {
        std::vector<float> result(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        for (size_t i = 0; i < size; ++i) {
            result[i] = dist(gen);
        }
        return result;
    }
};

// Test for basic integer sorting (ascending order - int32_t)
TEST_F(BitonicSortTest, SortInt32Ascending) {
    // Test with arrays of power-of-2 sizes
    const size_t sizes[] = {16, 32, 64, 128, 256};

    for (size_t size : sizes) {
        std::vector<int32_t> data = generateRandomInt32(size);
        std::vector<int32_t> expected = data;
        std::sort(expected.begin(), expected.end());

        sycl_points::algorithms::sort::BitonicSortShared<int32_t> sorter(*queue);
        sorter.sort(data);

        EXPECT_EQ(data, expected) << "Failed to sort int32_t array of size " << size;
    }
}

// Test for basic integer sorting (ascending order - uint64_t)
TEST_F(BitonicSortTest, SortUInt64Ascending) {
    // Test with arrays of power-of-2 sizes
    const size_t sizes[] = {16, 32, 64, 128};

    for (size_t size : sizes) {
        std::vector<uint64_t> data = generateRandomUInt64(size);
        std::vector<uint64_t> expected = data;
        std::sort(expected.begin(), expected.end());

        sycl_points::algorithms::sort::BitonicSortShared<uint64_t> sorter(*queue);
        sorter.sort(data);

        EXPECT_EQ(data, expected) << "Failed to sort uint64_t array of size " << size;
    }
}

// // Test for basic integer sorting (descending order - int32_t)
// TEST_F(BitonicSortTest, SortInt32Descending) {
//     const size_t size = 128;
//     std::vector<int32_t> data = generateRandomInt32(size);
//     std::vector<int32_t> expected = data;
//     std::sort(expected.begin(), expected.end(), std::greater<int32_t>());

//     sycl_points::algorithms::sort::BitonicSortShared<int32_t> sorter(queue_ptr);
//     sorter.sort(data, std::greater<int32_t>());

//     EXPECT_EQ(data, expected) << "Failed to sort int32_t array in descending order";
// }

// Test for floating point number sorting
TEST_F(BitonicSortTest, SortFloats) {
    const size_t size = 128;
    std::vector<float> data = generateRandomFloats(size);
    std::vector<float> expected = data;
    std::sort(expected.begin(), expected.end());

    sycl_points::algorithms::sort::BitonicSortShared<float> sorter(*queue);
    sorter.sort(data);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]) << "Mismatch at index " << i;
    }
}

// // Test for custom comparison function (absolute value)
// TEST_F(BitonicSortTest, SortWithAbsoluteValueComparator) {
//     const size_t size = 128;
//     std::vector<int32_t> data = generateRandomInt32(size, -100, 100);
//     std::vector<int32_t> expected = data;

//     auto abs_comparator = [](const int32_t& a, const int32_t& b) { return std::abs(a) < std::abs(b); };

//     std::sort(expected.begin(), expected.end(), abs_comparator);

//     sycl_points::algorithms::sort::BitonicSortShared<int32_t> sorter(queue_ptr);
//     sorter.sort(data, abs_comparator);

//     EXPECT_EQ(data, expected) << "Failed to sort int32_t array with custom comparator";
// }

// Edge case: empty array
TEST_F(BitonicSortTest, SortEmptyArray) {
    std::vector<int32_t> data;
    std::vector<int32_t> expected;

    sycl_points::algorithms::sort::BitonicSortShared<int32_t> sorter(*queue);
    sorter.sort(data);

    EXPECT_EQ(data, expected) << "Failed to handle empty array";
}

// Edge case: array with a single element
TEST_F(BitonicSortTest, SortSingleElementArray) {
    std::vector<int32_t> data = {42};
    std::vector<int32_t> expected = {42};

    sycl_points::algorithms::sort::BitonicSortShared<int32_t> sorter(*queue);
    sorter.sort(data);

    EXPECT_EQ(data, expected) << "Failed to handle single element array";
}

// Edge case: array with all identical elements
TEST_F(BitonicSortTest, SortAllSameElements) {
    const size_t size = 64;
    std::vector<int32_t> data(size, 7);  // Array with all elements set to 7
    std::vector<int32_t> expected(size, 7);

    sycl_points::algorithms::sort::BitonicSortShared<int32_t> sorter(*queue);
    sorter.sort(data);

    EXPECT_EQ(data, expected) << "Failed to handle array with all same elements";
}

// Edge case: array containing extreme values
TEST_F(BitonicSortTest, SortWithExtremeValues) {
    std::vector<int32_t> data = {std::numeric_limits<int32_t>::max(), 0, std::numeric_limits<int32_t>::min(), 42, -42};
    std::vector<int32_t> expected = data;
    std::sort(expected.begin(), expected.end());

    sycl_points::algorithms::sort::BitonicSortShared<int32_t> sorter(*queue);
    sorter.sort(data);

    EXPECT_EQ(data, expected) << "Failed to sort array with extreme values";
}

// Test with arrays of non-power-of-2 sizes (uint64_t)
TEST_F(BitonicSortTest, SortNonPowerOfTwoSizeUInt64) {
    const size_t sizes[] = {3, 7, 11, 19, 33, 63, 100, 129};

    for (size_t size : sizes) {
        std::vector<uint64_t> data = generateRandomUInt64(size);
        std::vector<uint64_t> expected = data;
        std::sort(expected.begin(), expected.end());

        sycl_points::algorithms::sort::BitonicSortShared<uint64_t> sorter(*queue);
        sorter.sort(data);

        EXPECT_EQ(data, expected) << "Failed to sort uint64_t array of non-power-of-2 size " << size;
    }
}

// Test for re-sorting an already sorted array (int32_t)
TEST_F(BitonicSortTest, SortAlreadySortedArray) {
    const size_t size = 128;
    std::vector<int32_t> data = generateRandomInt32(size);
    std::sort(data.begin(), data.end());
    std::vector<int32_t> expected = data;

    sycl_points::algorithms::sort::BitonicSortShared<int32_t> sorter(*queue);
    sorter.sort(data);

    EXPECT_EQ(data, expected) << "Failed to sort already sorted array";
}

// Test for sorting a reverse-ordered array (uint64_t)
TEST_F(BitonicSortTest, SortReverseOrderArray) {
    const size_t size = 128;
    std::vector<uint64_t> data = generateRandomUInt64(size);
    std::sort(data.begin(), data.end(), std::greater<uint64_t>());
    std::vector<uint64_t> expected = data;
    std::sort(expected.begin(), expected.end());

    sycl_points::algorithms::sort::BitonicSortShared<uint64_t> sorter(*queue);
    sorter.sort(data);

    EXPECT_EQ(data, expected) << "Failed to sort reverse-ordered uint64_t array";
}

// Test for sorting structures
struct TestStruct {
    int32_t key;
    float value;

    bool operator==(const TestStruct& other) const { return key == other.key && value == other.value; }
};

// Test for retrieving sorted indices (int32_t)
TEST_F(BitonicSortTest, GetSortedIndices) {
    const size_t size = 64;
    std::vector<int32_t> data = generateRandomInt32(size);
    std::vector<int32_t> original = data;

    // Create reference sort by pairing indices with data
    std::vector<std::pair<int32_t, size_t>> pairs(size);
    for (size_t i = 0; i < size; ++i) {
        pairs[i] = {original[i], i};
    }
    std::sort(pairs.begin(), pairs.end());

    std::vector<size_t> expected_indices(size);
    for (size_t i = 0; i < size; ++i) {
        expected_indices[i] = pairs[i].second;
    }

    sycl_points::algorithms::sort::BitonicSortShared<int32_t> sorter(*queue);
    sorter.sort(data);
    std::vector<size_t> indices = sorter.get_sorted_indices();

    EXPECT_EQ(indices, expected_indices) << "Failed to get correct sorted indices";

    // Verify that accessing original array elements using indices produces sorted result
    std::vector<int32_t> reconstructed(size);
    for (size_t i = 0; i < size; ++i) {
        reconstructed[i] = original[indices[i]];
    }

    EXPECT_EQ(reconstructed, data) << "Sorted indices do not correctly map original data";
}

// Test with a large array (uint64_t)
TEST_F(BitonicSortTest, SortLargeArray) {
    const size_t size = 1 << 15;  // 32,768
    std::vector<uint64_t> data = generateRandomUInt64(size);
    std::vector<uint64_t> expected = data;
    std::sort(expected.begin(), expected.end());

    sycl_points::algorithms::sort::BitonicSortShared<uint64_t> sorter(*queue);
    sorter.sort(data);

    EXPECT_EQ(data, expected) << "Failed to sort large uint64_t array";
}

// Test reusing the same sorter multiple times
TEST_F(BitonicSortTest, ReuseSorterMultipleTimes) {
    sycl_points::algorithms::sort::BitonicSortShared<int32_t> sorter(*queue);

    // Sort in order: small array, large array, then small array again
    {
        const size_t size = 32;
        std::vector<int32_t> data = generateRandomInt32(size);
        std::vector<int32_t> expected = data;
        std::sort(expected.begin(), expected.end());

        sorter.sort(data);
        EXPECT_EQ(data, expected) << "Failed on first small array";
    }

    {
        const size_t size = 1024;
        std::vector<int32_t> data = generateRandomInt32(size);
        std::vector<int32_t> expected = data;
        std::sort(expected.begin(), expected.end());

        sorter.sort(data);
        EXPECT_EQ(data, expected) << "Failed on large array";
    }

    {
        const size_t size = 16;
        std::vector<int32_t> data = generateRandomInt32(size);
        std::vector<int32_t> expected = data;
        std::sort(expected.begin(), expected.end());

        sorter.sort(data);
        EXPECT_EQ(data, expected) << "Failed on second small array";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
