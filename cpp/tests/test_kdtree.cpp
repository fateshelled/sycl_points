#include <gtest/gtest.h>

#include <limits>
#include <random>

#include "sycl_points/algorithms/common/filter_by_flags.hpp"
#include "sycl_points/algorithms/knn/bruteforce.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

class KDTreeTest : public ::testing::Test {
protected:
    sycl_points::sycl_utils::DeviceQueue::Ptr queue;
    sycl_points::PointCloudShared::Ptr target_cloud;
    sycl_points::PointCloudShared::Ptr query_cloud;
    std::shared_ptr<sycl_points::algorithms::knn::KDTree> kdtree;

    // Parameters for testing
    const size_t num_target_points = 1000;
    const size_t num_query_points = 100;
    const float point_range = 10.0f;
    const size_t random_seed = 1234;
    std::mt19937 gen;

    void SetUp() override {
        try {
            // set random seed
            gen.seed(random_seed);

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
            kdtree = sycl_points::algorithms::knn::KDTree::build(*queue, *target_cloud);

        } catch (const sycl::exception& e) {
            std::cerr << "SYCL exception caught: " << e.what() << std::endl;
            FAIL() << "Failed to initialize SYCL device";
        }
    }

    void TearDown() override {
        // Cleanup
    }

    // Helper function to generate random point clouds
    void generateRandomPoints(std::shared_ptr<sycl_points::PointContainerCPU> points, size_t num_points, float range) {
        std::uniform_real_distribution<float> dist(-range, range);

        for (size_t i = 0; i < num_points; ++i) {
            (*points)[i] = sycl_points::PointType(dist(gen), dist(gen), dist(gen), 1.0f);
        }
    }

    // Helper function to compare radius search results
    void compareRadiusResults(const sycl_points::algorithms::knn::KNNResult& kdtree_result,
                              const sycl_points::algorithms::knn::KNNResult& bruteforce_result, size_t max_k,
                              float epsilon = 1e-4f) {
        ASSERT_EQ(kdtree_result.query_size, bruteforce_result.query_size);
        ASSERT_EQ(kdtree_result.k, bruteforce_result.k);

        for (size_t i = 0; i < kdtree_result.query_size; ++i) {
            std::vector<std::pair<float, int32_t>> sorted_kdtree;
            std::vector<std::pair<float, int32_t>> sorted_bruteforce;

            // Create pairs of indices and distances
            for (size_t j = 0; j < max_k; ++j) {
                sorted_kdtree.push_back({(*kdtree_result.distances)[i * max_k + j],
                                         (*kdtree_result.indices)[i * max_k + j]});
                sorted_bruteforce.push_back({(*bruteforce_result.distances)[i * max_k + j],
                                             (*bruteforce_result.indices)[i * max_k + j]});
            }

            // Sort by distance (and by index in case of equal distances)
            auto comparator = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                if (a.first < b.first) {
                    return true;
                }
                if (a.first > b.first) {
                    return false;
                }
                return a.second < b.second;
            };

            std::sort(sorted_kdtree.begin(), sorted_kdtree.end(), comparator);
            std::sort(sorted_bruteforce.begin(), sorted_bruteforce.end(), comparator);

            // Compare results
            for (size_t j = 0; j < max_k; ++j) {
                if (sorted_bruteforce[j].second < 0) {
                    EXPECT_EQ(sorted_kdtree[j].second, -1) << "Missing invalid index at query " << i << ", slot " << j;
                    EXPECT_EQ(sorted_kdtree[j].first, std::numeric_limits<float>::max())
                        << "Missing invalid distance at query " << i << ", slot " << j;
                } else {
                    EXPECT_NEAR(sorted_kdtree[j].first, sorted_bruteforce[j].first, epsilon)
                        << "Distance mismatch at query " << i << ", neighbor " << j;

                    EXPECT_EQ(sorted_kdtree[j].second, sorted_bruteforce[j].second)
                        << "Index mismatch at query " << i << ", neighbor " << j << ": "
                        << sorted_kdtree[j].second << " vs " << sorted_bruteforce[j].second << " (distances: "
                        << sorted_kdtree[j].first << " vs " << sorted_bruteforce[j].first << ")";
                }
            }
        }
    }

    // Helper function to compute radius search results on host
    sycl_points::algorithms::knn::KNNResult bruteForceRadiusSearch(const sycl_points::PointCloudShared& queries,
                                                                   const sycl_points::PointCloudShared& targets,
                                                                   size_t max_k, float radius) {
        sycl_points::algorithms::knn::KNNResult result;
        result.allocate(*queue, queries.size(), max_k);

        const float radius_sq = radius * radius;
        const auto& query_points = *queries.points;
        const auto& target_points = *targets.points;

        for (size_t i = 0; i < queries.size(); ++i) {
            std::vector<std::pair<float, int32_t>> neighbors;
            neighbors.reserve(targets.size());
            for (size_t j = 0; j < targets.size(); ++j) {
                const auto diff = query_points[i] - target_points[j];
                const float dist_sq = diff.squaredNorm();
                if (dist_sq <= radius_sq) {
                    neighbors.emplace_back(dist_sq, static_cast<int32_t>(j));
                }
            }

            std::sort(neighbors.begin(), neighbors.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            for (size_t k = 0; k < max_k; ++k) {
                const size_t out_idx = i * max_k + k;
                if (k < neighbors.size()) {
                    (*result.distances)[out_idx] = neighbors[k].first;
                    (*result.indices)[out_idx] = neighbors[k].second;
                } else {
                    (*result.distances)[out_idx] = std::numeric_limits<float>::max();
                    (*result.indices)[out_idx] = -1;
                }
            }
        }

        return result;
    }

    // Helper function to compare KNN search results
    void compareKNNResults(const sycl_points::algorithms::knn::KNNResult& kdtree_result,
                           const sycl_points::algorithms::knn::KNNResult& bruteforce_result, size_t k,
                           float epsilon = 1e-4f) {
        ASSERT_EQ(kdtree_result.query_size, bruteforce_result.query_size);
        ASSERT_EQ(kdtree_result.k, bruteforce_result.k);

        for (size_t i = 0; i < kdtree_result.query_size; ++i) {
            std::vector<std::pair<float, int32_t>> sorted_kdtree;
            std::vector<std::pair<float, int32_t>> sorted_bruteforce;

            // Create pairs of indices and distances
            for (size_t j = 0; j < k; ++j) {
                sorted_kdtree.push_back({(*kdtree_result.distances)[i * k + j], (*kdtree_result.indices)[i * k + j]});
                sorted_bruteforce.push_back(
                    {(*bruteforce_result.distances)[i * k + j], (*bruteforce_result.indices)[i * k + j]});
            }

            // Sort by distance (and by index in case of equal distances)
            auto comparator = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                if (a.first < b.first) {
                    return true;
                }
                if (a.first > b.first) {
                    return false;
                }
                return a.second < b.second;
            };

            std::sort(sorted_kdtree.begin(), sorted_kdtree.end(), comparator);
            std::sort(sorted_bruteforce.begin(), sorted_bruteforce.end(), comparator);

            // Compare results
            for (size_t j = 0; j < k; ++j) {
                EXPECT_NEAR(sorted_kdtree[j].first, sorted_bruteforce[j].first, epsilon)
                    << "Distance mismatch at query " << i << ", neighbor " << j;

                EXPECT_EQ(sorted_kdtree[j].second, sorted_bruteforce[j].second)
                    << "Index mismatch at query " << i << ", neighbor " << j << ": " << sorted_kdtree[j].second
                    << " vs " << sorted_bruteforce[j].second << " (distances: " << sorted_kdtree[j].first << " vs "
                    << sorted_bruteforce[j].first << ")";
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
        auto bruteforce_result =
            sycl_points::algorithms::knn::knn_search_bruteforce(*queue, *query_cloud, *target_cloud, k);

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
            auto test_kdtree = sycl_points::algorithms::knn::KDTree::build(*queue, test_target);

            // Run kNN search with KDTree
            auto kdtree_result = test_kdtree->knn_search(test_query, k);

            // Run kNN search with brute force
            auto bruteforce_result =
                sycl_points::algorithms::knn::knn_search_bruteforce(*queue, test_query, test_target, k);

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
    auto single_kdtree = sycl_points::algorithms::knn::KDTree::build(*queue, single_target);

    // Run kNN search with KDTree
    auto kdtree_result = single_kdtree->knn_search(single_query, k);

    // Run kNN search with brute force
    auto bruteforce_result =
        sycl_points::algorithms::knn::knn_search_bruteforce(*queue, single_query, single_target, k);

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
        auto bruteforce_result =
            sycl_points::algorithms::knn::knn_search_bruteforce(*queue, *query_cloud, *target_cloud, k);

        // Compare results
        compareKNNResults(kdtree_result, bruteforce_result, k);

        std::cout << "Accuracy test passed for k=" << k << std::endl;
    }
}

// Performance test (large dataset)
TEST_F(KDTreeTest, PerformanceLargeDataset) {
    // This test only measures time, without verifying accuracy for large datasets
    const size_t large_target_size = 100000;  // Adjust as needed
    const size_t large_query_size = 100000;
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
    auto large_kdtree = sycl_points::algorithms::knn::KDTree::build(*queue, large_target);
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);

    // Time measurement - KDTree search
    auto kdtree_start = std::chrono::high_resolution_clock::now();
    auto kdtree_result = large_kdtree->knn_search(large_query, k);
    auto kdtree_end = std::chrono::high_resolution_clock::now();
    auto kdtree_duration = std::chrono::duration_cast<std::chrono::milliseconds>(kdtree_end - kdtree_start);

    // Time measurement - Brute force search
    auto bf_start = std::chrono::high_resolution_clock::now();
    auto bf_result = sycl_points::algorithms::knn::knn_search_bruteforce(*queue, large_query, large_target, k);
    auto bf_end = std::chrono::high_resolution_clock::now();
    auto bf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bf_end - bf_start);

    // Output results
    std::cout << "Performance test with " << large_target_size << " target points, " << large_query_size
              << " query points, k=" << k << ":\n";
    std::cout << "  KDTree build time: " << build_duration.count() << " ms\n";
    std::cout << "  KDTree search time: " << kdtree_duration.count() << " ms\n";
    std::cout << "  BruteForce search time: " << bf_duration.count() << " ms\n";
    std::cout << "  Speedup: " << (double)bf_duration.count() / kdtree_duration.count() << "x\n";

    // Also verify the results are correct
    compareKNNResults(kdtree_result, bf_result, k);
}

TEST_F(KDTreeTest, RemoveByFlags) {
    const size_t k = 10;
    const size_t target_size = 1000;

    // Generate point cloud
    sycl_points::PointCloudCPU target_cpu;
    target_cpu.points->resize(target_size);
    generateRandomPoints(target_cpu.points, target_size, point_range);

    auto test_target = sycl_points::PointCloudShared(*queue, target_cpu);
    auto test_kdtree = sycl_points::algorithms::knn::KDTree::build(*queue, test_target);

    // Initial search - all points find themselves
    auto initial_result = test_kdtree->knn_search(test_target, k);
    for (size_t i = 0; i < target_size; ++i) {
        EXPECT_FLOAT_EQ(0.0f, (*initial_result.distances)[i * k]);
        EXPECT_EQ(static_cast<int32_t>(i), (*initial_result.indices)[i * k]);
    }

    // Create removal flags - remove every 10th point
    sycl_points::shared_vector<uint8_t> flags(target_size, sycl_points::algorithms::filter::INCLUDE_FLAG, *queue->ptr);
    sycl_points::shared_vector<int32_t> indices(target_size, *queue->ptr);
    std::iota(indices.begin(), indices.end(), 0);
    int32_t count = 0;
    for (size_t i = 0; i < target_size; i += 10) {
        flags[i] = sycl_points::algorithms::filter::REMOVE_FLAG;
    }
    for (size_t i = 0; i < target_size; ++i) {
        indices[i] = (flags[i] == sycl_points::algorithms::filter::INCLUDE_FLAG) ? count++ : -1;
    }

    // Apply removal flags
    test_kdtree->remove_nodes_by_flags(flags, indices);

    // Remove points
    sycl_points::algorithms::filter::FilterByFlags filter_by_flags(*queue);
    auto removed_target = test_target;
    filter_by_flags.filter_by_flags(*removed_target.points, flags);

    // Search BruteForce with removed points
    const auto bf_result =
        sycl_points::algorithms::knn::knn_search_bruteforce(*queue, removed_target, removed_target, k);

    // Verify that removed points are not in results
    auto removed_result = test_kdtree->knn_search(removed_target, k);
    for (size_t i = 0; i < removed_result.query_size; ++i) {
        for (size_t j = 0; j < k; ++j) {
            const size_t result_idx = i * k + j;
            const int32_t point_idx = (*removed_result.indices)[result_idx];
            ASSERT_FLOAT_EQ(bf_result.distances->at(result_idx), removed_result.distances->at(result_idx))
                << "Mismatch in distances at query " << i << ", neighbor " << j;
        }
    }
}

TEST_F(KDTreeTest, RadiusSearchMatchesBruteForce) {
    const size_t max_k = 10;
    const float radius = 5.0f;

    sycl_points::algorithms::knn::KNNResult kdtree_result;
    kdtree->radius_search_async(*query_cloud, max_k, radius, kdtree_result).wait_and_throw();

    const auto bruteforce_result = bruteForceRadiusSearch(*query_cloud, *target_cloud, max_k, radius);

    compareRadiusResults(kdtree_result, bruteforce_result, max_k);
}

TEST_F(KDTreeTest, RadiusSearchFillsInvalidEntries) {
    const size_t max_k = 5;
    const float radius = 0.05f;

    sycl_points::PointCloudCPU target_cpu;
    target_cpu.points->resize(3);
    (*target_cpu.points)[0] = sycl_points::PointType(0.0f, 0.0f, 0.0f, 1.0f);
    (*target_cpu.points)[1] = sycl_points::PointType(10.0f, 0.0f, 0.0f, 1.0f);
    (*target_cpu.points)[2] = sycl_points::PointType(0.0f, 10.0f, 0.0f, 1.0f);

    sycl_points::PointCloudCPU query_cpu;
    query_cpu.points->resize(2);
    (*query_cpu.points)[0] = sycl_points::PointType(0.01f, 0.0f, 0.0f, 1.0f);
    (*query_cpu.points)[1] = sycl_points::PointType(20.0f, 20.0f, 0.0f, 1.0f);

    auto small_target = sycl_points::PointCloudShared(*queue, target_cpu);
    auto small_query = sycl_points::PointCloudShared(*queue, query_cpu);
    auto small_kdtree = sycl_points::algorithms::knn::KDTree::build(*queue, small_target);

    sycl_points::algorithms::knn::KNNResult kdtree_result;
    small_kdtree->radius_search_async(small_query, max_k, radius, kdtree_result).wait_and_throw();

    const auto bruteforce_result = bruteForceRadiusSearch(small_query, small_target, max_k, radius);
    compareRadiusResults(kdtree_result, bruteforce_result, max_k);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
