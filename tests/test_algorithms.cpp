#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <sycl_points/algorithms/sort/sort.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

class SortTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0, std::numeric_limits<uint64_t>::max());

        randomData.resize(100000);
        for (auto& val : randomData) {
            val = dis(gen);
        }

        sycl::device dev;  // set from Environments variable `ONEAPI_DEVICE_SELECTOR`
        queue_ptr = std::make_shared<sycl::queue>(dev);
    }

    std::vector<uint64_t> emptyArray;
    std::vector<uint64_t> singleElement = {42};
    std::vector<uint64_t> sortedArray = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<uint64_t> reversedArray = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<uint64_t> duplicateArray = {5, 2, 7, 2, 9, 7, 1, 5, 3, 5};
    std::vector<uint64_t> randomData;
    std::shared_ptr<sycl::queue> queue_ptr = nullptr;
};

TEST_F(SortTest, AscendingHeapSortTest) {
    using namespace sycl_points::algorithms::sort::kernel;
    // 空の配列
    heap_sort(emptyArray.data(), emptyArray.size());
    EXPECT_TRUE(emptyArray.empty());

    // 単一要素の配列
    heap_sort(singleElement.data(), singleElement.size());
    EXPECT_EQ(singleElement[0], 42);

    // ソート済み配列
    auto sortedCopy = sortedArray;
    heap_sort(sortedCopy.data(), sortedCopy.size());
    EXPECT_TRUE(std::is_sorted(sortedCopy.begin(), sortedCopy.end()));

    // 逆順配列
    auto reversedCopy = reversedArray;
    heap_sort(reversedCopy.data(), reversedCopy.size());
    EXPECT_TRUE(std::is_sorted(reversedCopy.begin(), reversedCopy.end()));

    // 重複要素を含む配列
    auto duplicateCopy = duplicateArray;
    heap_sort(duplicateCopy.data(), duplicateCopy.size());
    EXPECT_TRUE(std::is_sorted(duplicateCopy.begin(), duplicateCopy.end()));

    // ランダムデータ
    auto randomCopy = randomData;
    heap_sort(randomCopy.data(), randomCopy.size());
    EXPECT_TRUE(std::is_sorted(randomCopy.begin(), randomCopy.end()));

}

TEST_F(SortTest, AscendingHeapSortOnDeviceTest) {
    using namespace sycl_points::algorithms::sort::kernel;

    sycl_points::shared_vector<uint64_t> sharedRandom(randomData.size(), *queue_ptr);
    for (size_t i = 0; i < randomData.size(); ++i) {
        sharedRandom[i] = randomData[i];
    }

    queue_ptr->submit([&](sycl::handler& h) {
        auto ptr = sharedRandom.data();
        auto size = sharedRandom.size();
        h.single_task([=]() {
            heap_sort(ptr, size);
        });
    });
    queue_ptr->wait();

    EXPECT_TRUE(std::is_sorted(sharedRandom.begin(), sharedRandom.end()));
}

TEST_F(SortTest, DescendingHeapSortTest) {
    using namespace sycl_points::algorithms::sort::kernel;
    // 空の配列
    heap_sort_descending(emptyArray.data(), emptyArray.size());
    EXPECT_TRUE(emptyArray.empty());

    // 単一要素の配列
    heap_sort_descending(singleElement.data(), singleElement.size());
    EXPECT_EQ(singleElement[0], 42);

    // ソート済み配列
    auto sortedCopy = sortedArray;
    heap_sort_descending(sortedCopy.data(), sortedCopy.size());
    EXPECT_TRUE(std::is_sorted(sortedCopy.begin(), sortedCopy.end(), std::greater<uint64_t>()));

    // 逆順配列
    auto reversedCopy = reversedArray;
    heap_sort_descending(reversedCopy.data(), reversedCopy.size());
    EXPECT_TRUE(std::is_sorted(reversedCopy.begin(), reversedCopy.end(), std::greater<uint64_t>()));

    // 重複要素を含む配列
    auto duplicateCopy = duplicateArray;
    heap_sort_descending(duplicateCopy.data(), duplicateCopy.size());
    EXPECT_TRUE(std::is_sorted(duplicateCopy.begin(), duplicateCopy.end(), std::greater<uint64_t>()));

    // ランダムデータ
    auto randomCopy = randomData;
    heap_sort_descending(randomCopy.data(), randomCopy.size());
    EXPECT_TRUE(std::is_sorted(randomCopy.begin(), randomCopy.end(), std::greater<uint64_t>()));
}

TEST_F(SortTest, AscendingQuickSortTest) {
    using namespace sycl_points::algorithms::sort::kernel;

    bool success;
    // 空の配列
    success = quick_sort(emptyArray.data(), 0, emptyArray.size());
    EXPECT_TRUE(success);
    EXPECT_TRUE(emptyArray.empty());

    // 単一要素の配列
    success = quick_sort(singleElement.data(), 0, singleElement.size());
    EXPECT_TRUE(success);
    EXPECT_EQ(singleElement[0], 42);

    // ソート済み配列
    auto sortedCopy = sortedArray;
    success = quick_sort(sortedCopy.data(), 0, sortedCopy.size());
    EXPECT_TRUE(success);
    EXPECT_TRUE(std::is_sorted(sortedCopy.begin(), sortedCopy.end()));

    // 逆順配列
    auto reversedCopy = reversedArray;
    success = quick_sort(reversedCopy.data(), 0, reversedCopy.size());
    EXPECT_TRUE(success);
    EXPECT_TRUE(std::is_sorted(reversedCopy.begin(), reversedCopy.end()));

    // 重複要素を含む配列
    auto duplicateCopy = duplicateArray;
    success = quick_sort(duplicateCopy.data(), 0, duplicateCopy.size());
    EXPECT_TRUE(success);
    EXPECT_TRUE(std::is_sorted(duplicateCopy.begin(), duplicateCopy.end()));

    // ランダムデータ
    auto randomCopy = randomData;
    success = quick_sort<uint64_t, 32>(randomCopy.data(), 0, randomCopy.size());
    if (!success) success = quick_sort<uint64_t, 64>(randomCopy.data(), 0, randomCopy.size());
    if (!success) success = quick_sort<uint64_t, 128>(randomCopy.data(), 0, randomCopy.size());
    if (!success) success = quick_sort<uint64_t, 256>(randomCopy.data(), 0, randomCopy.size());
    EXPECT_TRUE(success);
    EXPECT_TRUE(std::is_sorted(randomCopy.begin(), randomCopy.end()));

}

TEST_F(SortTest, DescendingQuickSortTest) {
    using namespace sycl_points::algorithms::sort::kernel;

    bool success;
    // 空の配列
    success = quick_sort_descending(emptyArray.data(), 0, emptyArray.size());
    EXPECT_TRUE(success);
    EXPECT_TRUE(emptyArray.empty());

    // 単一要素の配列
    success = quick_sort_descending(singleElement.data(), 0, singleElement.size());
    EXPECT_TRUE(success);
    EXPECT_EQ(singleElement[0], 42);

    // ソート済み配列
    auto sortedCopy = sortedArray;
    success = quick_sort_descending(sortedCopy.data(), 0, sortedCopy.size());
    EXPECT_TRUE(success);
    EXPECT_TRUE(std::is_sorted(sortedCopy.begin(), sortedCopy.end(), std::greater<uint64_t>()));

    // 逆順配列
    auto reversedCopy = reversedArray;
    success = quick_sort_descending(reversedCopy.data(), 0, reversedCopy.size());
    EXPECT_TRUE(success);
    EXPECT_TRUE(std::is_sorted(reversedCopy.begin(), reversedCopy.end(), std::greater<uint64_t>()));

    // 重複要素を含む配列
    auto duplicateCopy = duplicateArray;
    success = quick_sort_descending(duplicateCopy.data(), 0, duplicateCopy.size());
    EXPECT_TRUE(success);
    EXPECT_TRUE(std::is_sorted(duplicateCopy.begin(), duplicateCopy.end(), std::greater<uint64_t>()));

    // ランダムデータ
    auto randomCopy = randomData;
    success = quick_sort_descending(randomCopy.data(), 0, randomCopy.size());
    if (!success) success = quick_sort_descending<uint64_t, 64>(randomCopy.data(), 0, randomCopy.size());
    if (!success) success = quick_sort_descending<uint64_t, 128>(randomCopy.data(), 0, randomCopy.size());
    if (!success) success = quick_sort_descending<uint64_t, 256>(randomCopy.data(), 0, randomCopy.size());
    EXPECT_TRUE(success);
    EXPECT_TRUE(std::is_sorted(randomCopy.begin(), randomCopy.end(), std::greater<uint64_t>()));

}

TEST_F(SortTest, AscendingQuickSortOnDeviceTest) {
    using namespace sycl_points::algorithms::sort;
    using namespace sycl_points::algorithms::sort::kernel;

    sycl_points::shared_vector<uint64_t> sharedRandom(randomData.size(), *queue_ptr);
    for (size_t i = 0; i < randomData.size(); ++i) {
        sharedRandom[i] = randomData[i];
    }

    queue_ptr->submit([&](sycl::handler& h) {
        auto data_ptr = sharedRandom.data();
        auto size = sharedRandom.size();
        h.single_task([=]() {
            bool success = quick_sort(data_ptr, 0, size);
            if (!success) success = quick_sort<uint64_t, 64>(data_ptr, 0, size);
            if (!success) success = quick_sort<uint64_t, 128>(data_ptr, 0, size);
            if (!success) success = quick_sort<uint64_t, 256>(data_ptr, 0, size);
        });
    });
    queue_ptr->wait();

    EXPECT_TRUE(std::is_sorted(sharedRandom.begin(), sharedRandom.end()));
}

TEST_F(SortTest, DescendingQuickSortOnDeviceTest) {
    using namespace sycl_points::algorithms::sort;
    using namespace sycl_points::algorithms::sort::kernel;

    sycl_points::shared_vector<uint64_t> sharedRandom(randomData.size(), *queue_ptr);
    for (size_t i = 0; i < randomData.size(); ++i) {
        sharedRandom[i] = randomData[i];
    }

    queue_ptr->submit([&](sycl::handler& h) {
        auto data_ptr = sharedRandom.data();
        auto size = sharedRandom.size();
        h.single_task([=]() {
            bool success = quick_sort_descending(data_ptr, 0, size);
            if (!success) success = quick_sort_descending<uint64_t, 64>(data_ptr, 0, size);
            if (!success) success = quick_sort_descending<uint64_t, 128>(data_ptr, 0, size);
            if (!success) success = quick_sort_descending<uint64_t, 256>(data_ptr, 0, size);
        });
    });
    queue_ptr->wait();

    EXPECT_TRUE(std::is_sorted(sharedRandom.begin(), sharedRandom.end(), std::greater<uint64_t>()));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
