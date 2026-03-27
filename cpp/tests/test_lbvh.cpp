#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "sycl_points/algorithms/filter/voxel_downsampling.hpp"
#include "sycl_points/algorithms/knn/bruteforce.hpp"
#include "sycl_points/algorithms/knn/kdtree.hpp"
#include "sycl_points/algorithms/knn/lbvh.hpp"
#include "sycl_points/algorithms/knn/result.hpp"
#include "sycl_points/io/point_cloud_reader.hpp"
#include "sycl_points/points/point_cloud.hpp"
#include "sycl_points/utils/sycl_utils.hpp"

namespace {

void generateRandomPoints(std::shared_ptr<sycl_points::PointContainerCPU> points, size_t num_points, float range,
                          std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-range, range);
    for (size_t i = 0; i < num_points; ++i) {
        (*points)[i] = sycl_points::PointType(dist(gen), dist(gen), dist(gen), 1.0f);
    }
}

void compareKNNResults(const sycl_points::algorithms::knn::KNNResult& lhs,
                       const sycl_points::algorithms::knn::KNNResult& rhs, size_t k, float epsilon = 1e-4f) {
    ASSERT_EQ(lhs.query_size, rhs.query_size);
    ASSERT_EQ(lhs.k, rhs.k);

    for (size_t i = 0; i < lhs.query_size; ++i) {
        for (size_t j = 0; j < k; ++j) {
            const size_t offset = i * k + j;
            ASSERT_NEAR((*lhs.distances)[offset], (*rhs.distances)[offset], epsilon)
                << "Distance mismatch at query " << i << ", neighbor " << j;
            ASSERT_EQ((*lhs.indices)[offset], (*rhs.indices)[offset])
                << "Index mismatch at query " << i << ", neighbor " << j;
        }
    }
}

std::filesystem::path locateDataFile(const std::string& relative_path) {
    const std::vector<std::filesystem::path> candidates = {std::filesystem::path(relative_path),
                                                           std::filesystem::path("..") / relative_path,
                                                           std::filesystem::path("../..") / relative_path};
    for (const auto& c : candidates) {
        if (std::filesystem::exists(c)) return std::filesystem::canonical(c);
    }
    throw std::runtime_error("[locateDataFile] unable to locate: " + relative_path);
}

}  // namespace

// ============================================================================
//  Correctness tests
// ============================================================================

/// Verify LBVH kNN results match brute-force for a small random point cloud.
TEST(LBVHTest, CompareWithBruteForce) {
    try {
        const size_t num_target = 256;
        const size_t num_query  = 64;
        const size_t k          = 4;
        const float  range      = 10.0f;
        const float  leaf_size  = 0.1f;

        std::mt19937 gen(42);
        sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::PointCloudCPU target_cpu, query_cpu;
        target_cpu.points->resize(num_target);
        query_cpu.points->resize(num_query);
        generateRandomPoints(target_cpu.points, num_target, range, gen);
        generateRandomPoints(query_cpu.points, num_query, range, gen);

        sycl_points::PointCloudShared target_cloud(queue, target_cpu);
        sycl_points::PointCloudShared query_cloud(queue, query_cpu);

        auto lbvh   = sycl_points::algorithms::knn::LBVH::build(queue, target_cloud);
        auto result = lbvh->knn_search(query_cloud, k);
        auto brute  = sycl_points::algorithms::knn::knn_search_bruteforce(queue, query_cloud, target_cloud, k);

        ASSERT_EQ(result.query_size, num_query);
        ASSERT_EQ(result.k, k);
        compareKNNResults(result, brute, k);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception: " << e.what();
    }
}

/// Verify that every point in the target cloud finds itself as its nearest
/// neighbour (distance == 0, index == self) when querying the target against
/// itself.
TEST(LBVHTest, SelfQuery) {
    try {
        const size_t n     = 512;
        const size_t k     = 1;
        const float  range = 5.0f;

        std::mt19937 gen(99);
        sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::PointCloudCPU cpu;
        cpu.points->resize(n);
        generateRandomPoints(cpu.points, n, range, gen);

        sycl_points::PointCloudShared cloud(queue, cpu);
        auto lbvh   = sycl_points::algorithms::knn::LBVH::build(queue, cloud);
        auto result = lbvh->knn_search(cloud, k);

        for (size_t i = 0; i < n; ++i) {
            EXPECT_FLOAT_EQ(0.0f, (*result.distances)[i * k])
                << "Non-zero self-distance at index " << i;
            EXPECT_EQ(static_cast<int32_t>(i), (*result.indices)[i * k])
                << "Wrong self-index at index " << i;
        }
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception: " << e.what();
    }
}

/// Compare LBVH and KDTree results for several k values on a medium cloud.
TEST(LBVHTest, CompareWithKDTree) {
    try {
        const std::vector<size_t> k_values = {1, 10, 20, 30};
        const size_t num_target = 1024;
        const size_t num_query  = 256;
        const float  range      = 20.0f;

        std::mt19937 gen(2024);
        sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::PointCloudCPU target_cpu, query_cpu;
        target_cpu.points->resize(num_target);
        query_cpu.points->resize(num_query);
        generateRandomPoints(target_cpu.points, num_target, range, gen);
        generateRandomPoints(query_cpu.points, num_query, range, gen);

        sycl_points::PointCloudShared target_cloud(queue, target_cpu);
        sycl_points::PointCloudShared query_cloud(queue, query_cpu);

        auto kdtree = sycl_points::algorithms::knn::KDTree::build(queue, target_cloud);

        for (size_t k : k_values) {
            auto lbvh_result = sycl_points::algorithms::knn::LBVH::build(queue, target_cloud)->knn_search(query_cloud, k);
            auto kd_result   = kdtree->knn_search(query_cloud, k);

            ASSERT_EQ(lbvh_result.query_size, num_query) << "query_size mismatch for k=" << k;
            ASSERT_EQ(lbvh_result.k, k)                 << "k mismatch for k=" << k;
            compareKNNResults(lbvh_result, kd_result, k);
        }
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception: " << e.what();
    }
}

/// Verify correct behaviour when the target cloud has exactly one point.
TEST(LBVHTest, SinglePointCloud) {
    try {
        sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::PointCloudCPU cpu;
        cpu.points->resize(1);
        (*cpu.points)[0] = sycl_points::PointType(1.0f, 2.0f, 3.0f, 1.0f);

        sycl_points::PointCloudShared cloud(queue, cpu);
        auto lbvh   = sycl_points::algorithms::knn::LBVH::build(queue, cloud);
        auto result = lbvh->knn_search(cloud, 1);

        ASSERT_EQ(result.query_size, 1u);
        ASSERT_EQ(result.k, 1u);
        EXPECT_FLOAT_EQ((*result.distances)[0], 0.0f);
        EXPECT_EQ((*result.indices)[0], 0);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception: " << e.what();
    }
}

/// Verify correct behaviour when the query cloud is empty.
TEST(LBVHTest, EmptyQueryCloud) {
    try {
        sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::PointCloudCPU target_cpu;
        target_cpu.points->resize(64);
        std::mt19937 gen(7);
        generateRandomPoints(target_cpu.points, 64, 5.0f, gen);

        sycl_points::PointCloudShared target_cloud(queue, target_cpu);
        sycl_points::PointCloudShared empty_queries(queue);
        empty_queries.points = std::make_shared<sycl_points::PointContainerShared>(0, *queue.ptr);

        auto lbvh   = sycl_points::algorithms::knn::LBVH::build(queue, target_cloud);
        auto result = lbvh->knn_search(empty_queries, 4);

        EXPECT_EQ(result.query_size, 0u);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception: " << e.what();
    }
}

/// Verify that all points in a collinear cloud are found correctly.
TEST(LBVHTest, CollinearPoints) {
    try {
        const size_t n = 128;
        const size_t k = 3;

        sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::PointCloudCPU cpu;
        cpu.points->resize(n);
        for (size_t i = 0; i < n; ++i) {
            (*cpu.points)[i] = sycl_points::PointType(static_cast<float>(i), 0.0f, 0.0f, 1.0f);
        }

        sycl_points::PointCloudShared cloud(queue, cpu);
        auto lbvh  = sycl_points::algorithms::knn::LBVH::build(queue, cloud);
        auto brute = sycl_points::algorithms::knn::knn_search_bruteforce(queue, cloud, cloud, k);
        auto result = lbvh->knn_search(cloud, k);

        compareKNNResults(result, brute, k);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception: " << e.what();
    }
}

// ============================================================================
//  Benchmark tests
// ============================================================================

/// Compare build and search times of LBVH vs KDTree on random clouds.
TEST(LBVHTest, BenchmarkAgainstKDTree) {
    try {
        using Clock = std::chrono::high_resolution_clock;

        const float  range        = 50.0f;
        const size_t random_seed  = 1337;
        const size_t bench_runs   = 20;

        const std::vector<std::pair<size_t, size_t>> cases = {{10000, 1000}, {100000, 10000}};
        const std::vector<size_t> k_values = {1, 10};

        sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        for (size_t ci = 0; ci < cases.size(); ++ci) {
            const auto [num_target, num_query] = cases[ci];
            std::mt19937 gen(random_seed + ci);

            sycl_points::PointCloudCPU target_cpu, query_cpu;
            target_cpu.points->resize(num_target);
            query_cpu.points->resize(num_query);
            generateRandomPoints(target_cpu.points, num_target, range, gen);
            generateRandomPoints(query_cpu.points, num_query, range, gen);

            sycl_points::PointCloudShared target_cloud(queue, target_cpu);
            sycl_points::PointCloudShared query_cloud(queue, query_cpu);

            for (size_t k : k_values) {
                // Warmup + correctness check
                auto lbvh_warm  = sycl_points::algorithms::knn::LBVH::build(queue, target_cloud)->knn_search(query_cloud, k);
                auto kd_warm    = sycl_points::algorithms::knn::KDTree::build(queue, target_cloud)->knn_search(query_cloud, k);
                compareKNNResults(lbvh_warm, kd_warm, k);

                double total_lbvh_build = 0, total_lbvh_search = 0;
                double total_kd_build   = 0, total_kd_search   = 0;

                for (size_t run = 0; run < bench_runs; ++run) {
                    auto t0 = Clock::now();
                    auto tree = sycl_points::algorithms::knn::LBVH::build(queue, target_cloud);
                    auto t1 = Clock::now();
                    auto res = tree->knn_search(query_cloud, k);
                    auto t2 = Clock::now();
                    total_lbvh_build  += std::chrono::duration<double, std::milli>(t1 - t0).count();
                    total_lbvh_search += std::chrono::duration<double, std::milli>(t2 - t1).count();

                    auto t3 = Clock::now();
                    auto kd = sycl_points::algorithms::knn::KDTree::build(queue, target_cloud);
                    auto t4 = Clock::now();
                    auto kr = kd->knn_search(query_cloud, k);
                    auto t5 = Clock::now();
                    total_kd_build  += std::chrono::duration<double, std::milli>(t4 - t3).count();
                    total_kd_search += std::chrono::duration<double, std::milli>(t5 - t4).count();
                }

                std::cout << "Benchmark case " << ci + 1
                          << ": target=" << num_target << " query=" << num_query << " k=" << k << "\n"
                          << "  LBVH  build=" << total_lbvh_build  / bench_runs << " ms"
                          << "  search=" << total_lbvh_search / bench_runs << " ms\n"
                          << "  KDTree build=" << total_kd_build   / bench_runs << " ms"
                          << "  search=" << total_kd_search  / bench_runs << " ms\n";
            }
        }
        SUCCEED();
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception: " << e.what();
    }
}

/// Benchmark with real PLY dataset files when available.
TEST(LBVHTest, BenchmarkWithDataset) {
    try {
        using Clock = std::chrono::high_resolution_clock;

        const std::vector<size_t> k_values = {1, 10, 20, 30};
        const size_t bench_runs = 50;
        const float  leaf_size  = 0.1f;

        sycl::device device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        const auto target_path = locateDataFile("data/target.ply");
        const auto query_path  = locateDataFile("data/source.ply");
        const auto target_cpu  = sycl_points::PointCloudReader::readFile(target_path.string());
        const auto query_cpu   = sycl_points::PointCloudReader::readFile(query_path.string());

        ASSERT_FALSE(target_cpu.points->empty()) << "Target cloud must not be empty";
        ASSERT_FALSE(query_cpu.points->empty())  << "Query cloud must not be empty";
        ASSERT_GE(target_cpu.points->size(), k_values.back());

        sycl_points::algorithms::filter::VoxelGrid vg(queue, leaf_size);

        sycl_points::PointCloudShared target_cloud(queue, target_cpu);
        sycl_points::PointCloudShared query_cloud(queue, query_cpu);
        sycl_points::PointCloudShared ds_target(queue), ds_query(queue);

        vg.downsampling(target_cloud, ds_target);
        vg.downsampling(query_cloud, ds_query);

        size_t case_idx = 0;
        for (const auto& [tgt, qry] : {std::make_pair(ds_target, ds_query),
                                        std::make_pair(target_cloud, ds_query),
                                        std::make_pair(ds_target, query_cloud),
                                        std::make_pair(target_cloud, query_cloud)}) {
            std::cout << "Dataset benchmark case " << ++case_idx << "\n";
            for (size_t k : k_values) {
                double total_lbvh_build = 0, total_lbvh_search = 0;
                double total_kd_build   = 0, total_kd_search   = 0;

                for (size_t run = 0; run < bench_runs; ++run) {
                    auto t0   = Clock::now();
                    auto lbvh = sycl_points::algorithms::knn::LBVH::build(queue, tgt);
                    auto t1   = Clock::now();
                    auto lr   = lbvh->knn_search(qry, k);
                    auto t2   = Clock::now();
                    ASSERT_EQ(lr.query_size, qry.size());
                    ASSERT_EQ(lr.k, k);
                    total_lbvh_build  += std::chrono::duration<double, std::milli>(t1 - t0).count();
                    total_lbvh_search += std::chrono::duration<double, std::milli>(t2 - t1).count();

                    auto t3 = Clock::now();
                    auto kd = sycl_points::algorithms::knn::KDTree::build(queue, tgt);
                    auto t4 = Clock::now();
                    auto kr = kd->knn_search(qry, k);
                    auto t5 = Clock::now();
                    ASSERT_EQ(kr.query_size, qry.size());
                    ASSERT_EQ(kr.k, k);
                    total_kd_build  += std::chrono::duration<double, std::milli>(t4 - t3).count();
                    total_kd_search += std::chrono::duration<double, std::milli>(t5 - t4).count();
                }

                std::cout << "  target=" << tgt.size() << " query=" << qry.size() << " k=" << k << "\n"
                          << "    LBVH  build=" << total_lbvh_build  / bench_runs << " ms"
                          << "  search=" << total_lbvh_search / bench_runs << " ms\n"
                          << "    KDTree build=" << total_kd_build   / bench_runs << " ms"
                          << "  search=" << total_kd_search  / bench_runs << " ms\n";
            }
        }
        SUCCEED();
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception: " << e.what();
    }
}
