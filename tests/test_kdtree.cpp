#include <gtest/gtest.h>

#include <random>
#include <sycl_points/algorithms/knn_search.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

class KDTreeTest : public ::testing::Test {
protected:
    sycl_points::sycl_utils::DeviceQueue::Ptr queue;
    sycl_points::PointCloudShared::Ptr target_cloud;
    sycl_points::PointCloudShared::Ptr query_cloud;
    std::shared_ptr<sycl_points::algorithms::knn_search::KDTree> kdtree;

    // Parameters for testing
    const size_t num_target_points = 1000;
    const size_t num_query_points = 100;
    const float point_range = 10.0f;

    void SetUp() override {
        try {
            // Setup SYCL device
            sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
            queue = std::make_shared<sycl_points::sycl_utils::DeviceQueue>(device);
            std::cout << "Using device: " << device.get_info<sycl::info::device::name>() << std::endl;

            // Generate random point clouds
            target_cloud = std::make_shared<sycl_points::PointCloudShared>(*queue);
            query_cloud = std::make_shared<sycl_points::PointCloudShared>(*queue);

            // Generate target point cloud
            sycl_points::PointCloudCPU target_cpu;
            target_cpu.points->resize(num_target_points);
            generateRandomPoints(target_cpu.points, num_target_points, point_range);

            // Generate query point cloud
            sycl_points::PointCloudCPU query_cpu;
            query_cpu.points->resize(num_query_points);
            generateRandomPoints(query_cpu.points, num_query_points, point_range);

            // Create shared point clouds from CPU point clouds
            *target_cloud = sycl_points::PointCloudShared(*queue, target_cpu);
            *query_cloud = sycl_points::PointCloudShared(*queue, query_cpu);

            // Build KDTree
            kdtree = std::make_shared<sycl_points::algorithms::knn_search::KDTree>(
                sycl_points::algorithms::knn_search::KDTree::build(*queue, *target_cloud));

        } catch (const sycl::exception& e) {
            std::cerr << "SYCL exception caught: " << e.what() << std::endl;
            FAIL() << "Failed to initialize SYCL device";
        }
    }

    void TearDown() override {
        // Cleanup
    }

    // Helper function to generate random point clouds
    void generateRandomPoints(std::shared_ptr<sycl_points::PointContainerCPU> points,
                             size_t num_points, float range) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-range, range);

        for (size_t i = 0; i < num_points; ++i) {
            (*points)[i] = sycl_points::PointType(dist(gen), dist(gen), dist(gen), 1.0f);
        }
    }

    // Helper function to compare KNN search results
    void compareKNNResults(const sycl_points::algorithms::knn_search::KNNResultSYCL& result1,
                         const sycl_points::algorithms::knn_search::KNNResultSYCL& result2,
                         size_t k, float epsilon = 1e-4f) {
        ASSERT_EQ(result1.query_size, result2.query_size);
        ASSERT_EQ(result1.k, result2.k);

        for (size_t i = 0; i < result1.query_size; ++i) {
            std::vector<std::pair<float, int32_t>> sorted_result1;
            std::vector<std::pair<float, int32_t>> sorted_result2;

            // Create pairs of indices and distances
            for (size_t j = 0; j < k; ++j) {
                sorted_result1.push_back({(*result1.distances)[i * k + j], (*result1.indices)[i * k + j]});
                sorted_result2.push_back({(*result2.distances)[i * k + j], (*result2.indices)[i * k + j]});
            }

            // Sort by distance (and by index in case of equal distances)
            auto comparator = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                if (a.first - b.first) {
                    return a.second < b.second;
                }
                return a.first < b.first;
            };

            std::sort(sorted_result1.begin(), sorted_result1.end(), comparator);
            std::sort(sorted_result2.begin(), sorted_result2.end(), comparator);

            // Compare results
            for (size_t j = 0; j < k; ++j) {
                EXPECT_NEAR(sorted_result1[j].first, sorted_result2[j].first, epsilon)
                    << "Distance mismatch at query " << i << ", neighbor " << j;

                // Check if indices match, or if the points have the same distance
                bool valid_index = (sorted_result1[j].second == sorted_result2[j].second) ||
                                  (std::abs(sorted_result1[j].first - sorted_result2[j].first) < epsilon);

                EXPECT_TRUE(valid_index)
                    << "Index mismatch at query " << i << ", neighbor " << j
                    << ": " << sorted_result1[j].second << " vs " << sorted_result2[j].second
                    << " (distances: " << sorted_result1[j].first << " vs " << sorted_result2[j].second << ")";
            }
        }
    }
};

// Basic KDTree kNN search test
TEST_F(KDTreeTest, BasicKNNSearch) {
    const size_t k = 5;

    // Run kNN search with KDTree
    auto kdtree_result = kdtree->knn_search(*query_cloud, k);

    // Confirm that the search is successful
    EXPECT_EQ(kdtree_result.query_size, num_query_points);
    EXPECT_EQ(kdtree_result.k, k);

    // Verify that results exist for all query points
    for (size_t i = 0; i < num_query_points; ++i) {
        for (size_t j = 0; j < k; ++j) {
            const size_t index = i * k + j;
            EXPECT_GE((*kdtree_result.indices)[index], 0);
            EXPECT_LT((*kdtree_result.indices)[index], num_target_points);
            EXPECT_GE((*kdtree_result.distances)[index], 0.0f);
        }
    }
}

// Test comparing KDTree and brute force results
TEST_F(KDTreeTest, CompareWithBruteForce) {
    const std::vector<size_t> k_values = {1, 5, 10};

    for (size_t k : k_values) {
        // Run kNN search with KDTree
        auto kdtree_result = kdtree->knn_search(*query_cloud, k);

        // Run kNN search with brute force
        auto bruteforce_result = sycl_points::algorithms::knn_search::knn_search_bruteforce(
            *queue, *query_cloud, *target_cloud, k);

        // Compare results
        compareKNNResults(kdtree_result, bruteforce_result, k);

        std::cout << "KDTree and BruteForce match for k=" << k << std::endl;
    }
}

// Test with point clouds of various sizes
TEST_F(KDTreeTest, VariousSizePointClouds) {
    const std::vector<size_t> target_sizes = {10, 100, 500};
    const std::vector<size_t> query_sizes = {5, 20};
    const size_t k = 3;

    for (size_t target_size : target_sizes) {
        for (size_t query_size : query_sizes) {
            // Generate new point clouds
            sycl_points::PointCloudCPU target_cpu;
            target_cpu.points->resize(target_size);
            generateRandomPoints(target_cpu.points, target_size, point_range);

            sycl_points::PointCloudCPU query_cpu;
            query_cpu.points->resize(query_size);
            generateRandomPoints(query_cpu.points, query_size, point_range);

            auto test_target = sycl_points::PointCloudShared(*queue, target_cpu);
            auto test_query = sycl_points::PointCloudShared(*queue, query_cpu);

            // Build KDTree
            auto test_kdtree = sycl_points::algorithms::knn_search::KDTree::build(*queue, test_target);

            // Run kNN search with KDTree
            auto kdtree_result = test_kdtree.knn_search(test_query, k);

            // Run kNN search with brute force
            auto bruteforce_result = sycl_points::algorithms::knn_search::knn_search_bruteforce(
                *queue, test_query, test_target, k);

            // Compare results
            compareKNNResults(kdtree_result, bruteforce_result, k);

            std::cout << "Match for target_size=" << target_size << ", query_size=" << query_size << std::endl;
        }
    }
}

// Edge case: Single point
TEST_F(KDTreeTest, SinglePoint) {
    const size_t k = 1;

    // Generate a point cloud with a single point
    sycl_points::PointCloudCPU target_cpu;
    target_cpu.points->resize(1);
    (*target_cpu.points)[0] = sycl_points::PointType(0.0f, 0.0f, 0.0f, 1.0f);

    sycl_points::PointCloudCPU query_cpu;
    query_cpu.points->resize(1);
    (*query_cpu.points)[0] = sycl_points::PointType(1.0f, 1.0f, 1.0f, 1.0f);

    auto single_target = sycl_points::PointCloudShared(*queue, target_cpu);
    auto single_query = sycl_points::PointCloudShared(*queue, query_cpu);

    // Build KDTree
    auto single_kdtree = sycl_points::algorithms::knn_search::KDTree::build(*queue, single_target);

    // Run kNN search with KDTree
    auto kdtree_result = single_kdtree.knn_search(single_query, k);

    // Run kNN search with brute force
    auto bruteforce_result = sycl_points::algorithms::knn_search::knn_search_bruteforce(
        *queue, single_query, single_target, k);

    // Compare results
    compareKNNResults(kdtree_result, bruteforce_result, k);

    // Also check specific values
    EXPECT_EQ((*kdtree_result.indices)[0], 0);
    EXPECT_NEAR((*kdtree_result.distances)[0], 3.0f, 1e-6f);  // Distance is (1-0)^2 + (1-0)^2 + (1-0)^2 = 3
}

// Test accuracy with different k values
TEST_F(KDTreeTest, AccuracyWithDifferentK) {
    const std::vector<size_t> k_values = {1, 3, 5, 10, 20};

    for (size_t k : k_values) {
        // Run kNN search with KDTree
        auto kdtree_result = kdtree->knn_search(*query_cloud, k);

        // Run kNN search with brute force
        auto bruteforce_result = sycl_points::algorithms::knn_search::knn_search_bruteforce(
            *queue, *query_cloud, *target_cloud, k);

        // Compare results
        compareKNNResults(kdtree_result, bruteforce_result, k);

        std::cout << "Accuracy test passed for k=" << k << std::endl;
    }
}

// Performance test (large dataset)
TEST_F(KDTreeTest, PerformanceLargeDataset) {
    // This test only measures time, without verifying accuracy for large datasets
    const size_t large_target_size = 10000; // Adjust as needed
    const size_t large_query_size = 100;
    const size_t k = 10;

    // Generate large point clouds
    sycl_points::PointCloudCPU target_cpu;
    target_cpu.points->resize(large_target_size);
    generateRandomPoints(target_cpu.points, large_target_size, point_range);

    sycl_points::PointCloudCPU query_cpu;
    query_cpu.points->resize(large_query_size);
    generateRandomPoints(query_cpu.points, large_query_size, point_range);

    auto large_target = sycl_points::PointCloudShared(*queue, target_cpu);
    auto large_query = sycl_points::PointCloudShared(*queue, query_cpu);

    // Time measurement - KDTree construction
    auto build_start = std::chrono::high_resolution_clock::now();
    auto large_kdtree = sycl_points::algorithms::knn_search::KDTree::build(*queue, large_target);
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);

    // Time measurement - KDTree search
    auto kdtree_start = std::chrono::high_resolution_clock::now();
    auto kdtree_result = large_kdtree.knn_search(large_query, k);
    auto kdtree_end = std::chrono::high_resolution_clock::now();
    auto kdtree_duration = std::chrono::duration_cast<std::chrono::milliseconds>(kdtree_end - kdtree_start);

    // Time measurement - Brute force search
    auto bf_start = std::chrono::high_resolution_clock::now();
    auto bf_result = sycl_points::algorithms::knn_search::knn_search_bruteforce(
        *queue, large_query, large_target, k);
    auto bf_end = std::chrono::high_resolution_clock::now();
    auto bf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bf_end - bf_start);

    // Output results
    std::cout << "Performance test with " << large_target_size << " target points, "
              << large_query_size << " query points, k=" << k << ":\n";
    std::cout << "  KDTree build time: " << build_duration.count() << " ms\n";
    std::cout << "  KDTree search time: " << kdtree_duration.count() << " ms\n";
    std::cout << "  BruteForce search time: " << bf_duration.count() << " ms\n";
    std::cout << "  Speedup: " << (double)bf_duration.count() / kdtree_duration.count() << "x\n";

    // Also verify the results are correct
    compareKNNResults(kdtree_result, bf_result, k);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
