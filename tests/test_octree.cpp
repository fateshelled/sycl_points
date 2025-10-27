#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sycl_points/algorithms/common/filter_by_flags.hpp>
#include <sycl_points/algorithms/knn/bruteforce.hpp>
#include <sycl_points/algorithms/knn/kdtree.hpp>
#include <sycl_points/algorithms/knn/octree.hpp>
#include <sycl_points/algorithms/knn/result.hpp>
#include <sycl_points/algorithms/voxel_downsampling.hpp>
#include <sycl_points/io/point_cloud_reader.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>
#include <tuple>
#include <vector>

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
                << "Distance mismatch at query " << i << ", neighbour " << j;
            ASSERT_EQ((*lhs.indices)[offset], (*rhs.indices)[offset])
                << "Index mismatch at query " << i << ", neighbour " << j;
        }
    }
}

std::filesystem::path locateDataFile(const std::string& relative_path) {
    const std::vector<std::filesystem::path> search_candidates = {std::filesystem::path(relative_path),
                                                                  std::filesystem::path("..") / relative_path,
                                                                  std::filesystem::path("../..") / relative_path};
    for (const auto& candidate : search_candidates) {
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::canonical(candidate);
        }
    }
    throw std::runtime_error("Unable to locate data file: " + relative_path);
}

}  // namespace

TEST(OctreeTest, CompareWithBruteForceInterfaceOnly) {
    try {
        const size_t num_target_points = 256;
        const size_t num_query_points = 64;
        const size_t k = 4;
        const float point_range = 10.0f;
        const float leaf_size = 0.1f;

        std::mt19937 gen(2024);

        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::PointCloudCPU target_cpu;
        target_cpu.points->resize(num_target_points);
        generateRandomPoints(target_cpu.points, num_target_points, point_range, gen);

        sycl_points::PointCloudCPU query_cpu;
        query_cpu.points->resize(num_query_points);
        generateRandomPoints(query_cpu.points, num_query_points, point_range, gen);

        sycl_points::PointCloudShared target_cloud(queue, target_cpu);
        sycl_points::PointCloudShared query_cloud(queue, query_cpu);

        auto octree = sycl_points::algorithms::knn::Octree::build(queue, target_cloud, leaf_size);
        auto octree_result = octree->knn_search(query_cloud, k);
        auto bruteforce_result =
            sycl_points::algorithms::knn::knn_search_bruteforce(queue, query_cloud, target_cloud, k);

        compareKNNResults(octree_result, bruteforce_result, k);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(OctreeTest, RemoveByFlags) {
    try {
        const size_t target_size = 1024;
        const size_t k = 10;
        const float point_range = 10.0f;
        const float leaf_size = 0.1f;

        std::mt19937 gen(2025);

        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        sycl_points::PointCloudCPU target_cpu;
        target_cpu.points->resize(target_size);
        generateRandomPoints(target_cpu.points, target_size, point_range, gen);

        sycl_points::PointCloudShared target_cloud(queue, target_cpu);
        auto octree = sycl_points::algorithms::knn::Octree::build(queue, target_cloud, leaf_size);

        auto initial_result = octree->knn_search(target_cloud, k);
        for (size_t i = 0; i < target_size; ++i) {
            EXPECT_FLOAT_EQ(0.0f, (*initial_result.distances)[i * k]);
            EXPECT_EQ(static_cast<int32_t>(i), (*initial_result.indices)[i * k]);
        }

        sycl_points::shared_vector<uint8_t> flags(target_size, sycl_points::algorithms::filter::INCLUDE_FLAG, *queue.ptr);
        sycl_points::shared_vector<int32_t> indices(target_size, *queue.ptr);
        std::iota(indices.begin(), indices.end(), 0);

        int32_t compact_index = 0;
        for (size_t i = 0; i < target_size; ++i) {
            if (i % 7 == 0) {
                flags[i] = sycl_points::algorithms::filter::REMOVE_FLAG;
            }
            indices[i] = (flags[i] == sycl_points::algorithms::filter::INCLUDE_FLAG) ? compact_index++ : -1;
        }

        octree->remove_nodes_by_flags(flags, indices);

        sycl_points::algorithms::filter::FilterByFlags filter(queue);
        auto filtered_cloud = target_cloud;
        filter.filter_by_flags(*filtered_cloud.points, flags);

        const auto brute_force =
            sycl_points::algorithms::knn::knn_search_bruteforce(queue, filtered_cloud, filtered_cloud, k);
        const auto updated_result = octree->knn_search(filtered_cloud, k);

        for (size_t query_idx = 0; query_idx < updated_result.query_size; ++query_idx) {
            for (size_t neighbor = 0; neighbor < k; ++neighbor) {
                const size_t offset = query_idx * k + neighbor;
                ASSERT_FLOAT_EQ(brute_force.distances->at(offset), updated_result.distances->at(offset))
                    << "Mismatch in distances at query " << query_idx << ", neighbor " << neighbor;
                ASSERT_EQ(brute_force.indices->at(offset), updated_result.indices->at(offset))
                    << "Mismatch in indices at query " << query_idx << ", neighbor " << neighbor;
            }
        }
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(OctreeTest, BenchmarkOctreeAgainstKDTree) {
    try {
        using Clock = std::chrono::high_resolution_clock;

        const float point_range = 50.0f;
        const float leaf_size = 0.1f;
        const size_t random_seed = 1337;
        const size_t benchmark_runs = 20;

        const std::vector<std::pair<size_t, size_t>> benchmark_cases = {
            {10000, 1000},
            {100000, 10000},
        };

        const std::vector<size_t> k_values = {1, 10};

        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        for (size_t case_index = 0; case_index < benchmark_cases.size(); ++case_index) {
            const auto [num_target_points, num_query_points] = benchmark_cases[case_index];
            std::mt19937 gen(random_seed + static_cast<unsigned int>(case_index));

            sycl_points::PointCloudCPU target_cpu;
            target_cpu.points->resize(num_target_points);
            generateRandomPoints(target_cpu.points, num_target_points, point_range, gen);

            sycl_points::PointCloudCPU query_cpu;
            query_cpu.points->resize(num_query_points);
            generateRandomPoints(query_cpu.points, num_query_points, point_range, gen);

            sycl_points::PointCloudShared target_cloud(queue, target_cpu);
            sycl_points::PointCloudShared query_cloud(queue, query_cpu);

            auto run_octree_iteration = [&](size_t knn_k) {
                const auto build_start = Clock::now();
                auto tree = sycl_points::algorithms::knn::Octree::build(queue, target_cloud, leaf_size);
                const auto build_end = Clock::now();
                const auto search_start = Clock::now();
                auto result = tree->knn_search(query_cloud, knn_k);
                const auto search_end = Clock::now();

                const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
                const double search_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();

                return std::make_tuple(build_ms, search_ms, result);
            };

            auto run_kdtree_iteration = [&](size_t knn_k) {
                const auto build_start = Clock::now();
                auto tree = sycl_points::algorithms::knn::KDTree::build(queue, target_cloud);
                const auto build_end = Clock::now();
                const auto search_start = Clock::now();
                auto result = tree->knn_search(query_cloud, knn_k);
                const auto search_end = Clock::now();

                const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
                const double search_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();

                return std::make_tuple(build_ms, search_ms, result);
            };

            for (const size_t k : k_values) {
                const auto warmup_octree = run_octree_iteration(k);
                const auto warmup_kdtree = run_kdtree_iteration(k);

                const auto& warmup_octree_result = std::get<2>(warmup_octree);
                const auto& warmup_kdtree_result = std::get<2>(warmup_kdtree);

                compareKNNResults(warmup_octree_result, warmup_kdtree_result, k);

                double total_octree_build_ms = 0.0;
                double total_octree_search_ms = 0.0;
                for (size_t run = 0; run < benchmark_runs; ++run) {
                    auto [build_ms, search_ms, result] = run_octree_iteration(k);
                    compareKNNResults(result, warmup_kdtree_result, k);
                    total_octree_build_ms += build_ms;
                    total_octree_search_ms += search_ms;
                }

                double total_kdtree_build_ms = 0.0;
                double total_kdtree_search_ms = 0.0;
                for (size_t run = 0; run < benchmark_runs; ++run) {
                    auto [build_ms, search_ms, result] = run_kdtree_iteration(k);
                    compareKNNResults(result, warmup_kdtree_result, k);
                    total_kdtree_build_ms += build_ms;
                    total_kdtree_search_ms += search_ms;
                }

                const double average_octree_build_ms = total_octree_build_ms / static_cast<double>(benchmark_runs);
                const double average_octree_search_ms = total_octree_search_ms / static_cast<double>(benchmark_runs);
                const double average_kdtree_build_ms = total_kdtree_build_ms / static_cast<double>(benchmark_runs);
                const double average_kdtree_search_ms = total_kdtree_search_ms / static_cast<double>(benchmark_runs);

                std::cout << "Benchmark case " << case_index + 1 << ": target_points=" << num_target_points
                          << ", query_points=" << num_query_points << ", k=" << k << std::endl;
                std::cout << "  Average Octree build time over " << benchmark_runs
                          << " runs: " << average_octree_build_ms << " ms" << std::endl;
                std::cout << "  Average Octree search time over " << benchmark_runs
                          << " runs: " << average_octree_search_ms << " ms" << std::endl;
                std::cout << "  Average KDTree build time over " << benchmark_runs
                          << " runs: " << average_kdtree_build_ms << " ms" << std::endl;
                std::cout << "  Average KDTree search time over " << benchmark_runs
                          << " runs: " << average_kdtree_search_ms << " ms" << std::endl;
            }
        }

        SUCCEED();
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(OctreeTest, BenchmarkOctreeWithDatasetTargets) {
    try {
        using Clock = std::chrono::high_resolution_clock;

        const std::vector<size_t> k_values = {1, 10, 20, 30};
        const size_t benchmark_runs = 50;
        const float leaf_size = 0.1f;

        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        const auto target_path = locateDataFile("data/target.ply");
        const auto query_path = locateDataFile("data/source.ply");
        const auto target_cpu = sycl_points::PointCloudReader::readFile(target_path.string());
        const auto query_cpu = sycl_points::PointCloudReader::readFile(query_path.string());

        ASSERT_FALSE(target_cpu.points->empty()) << "Target point cloud must not be empty";
        ASSERT_FALSE(query_cpu.points->empty()) << "Query point cloud must not be empty";

        ASSERT_GE(target_cpu.points->size(), k_values.back())
            << "Dataset contains fewer points than required for the largest k";

        sycl_points::algorithms::filter::VoxelGrid vg(queue, leaf_size);

        sycl_points::PointCloudShared target_cloud(queue, target_cpu);
        sycl_points::PointCloudShared downsampled_target_cloud(queue);
        sycl_points::PointCloudShared query_cloud(queue, query_cpu);
        sycl_points::PointCloudShared downsampled_query_cloud(queue);

        vg.downsampling(target_cloud, downsampled_target_cloud);
        vg.downsampling(query_cloud, downsampled_query_cloud);

        auto run_octree_iteration = [&](size_t knn_k, const sycl_points::PointCloudShared& target,
                                        const sycl_points::PointCloudShared& query) {
            const auto build_start = Clock::now();
            auto tree = sycl_points::algorithms::knn::Octree::build(queue, target, leaf_size);
            const auto build_end = Clock::now();
            const auto search_start = Clock::now();
            auto result = tree->knn_search(query, knn_k);
            const auto search_end = Clock::now();

            const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
            const double search_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();

            return std::make_tuple(build_ms, search_ms, result);
        };

        auto run_kdtree_iteration = [&](size_t knn_k, const sycl_points::PointCloudShared& target,
                                        const sycl_points::PointCloudShared& query) {
            const auto build_start = Clock::now();
            auto tree = sycl_points::algorithms::knn::KDTree::build(queue, target);
            const auto build_end = Clock::now();
            const auto search_start = Clock::now();
            auto result = tree->knn_search(query, knn_k);
            const auto search_end = Clock::now();

            const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
            const double search_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();

            return std::make_tuple(build_ms, search_ms, result);
        };

        size_t case_index = 0;
        for (const auto& [target, query] : {std::make_pair(downsampled_target_cloud, downsampled_query_cloud),  //
                                            std::make_pair(target_cloud, downsampled_query_cloud),              //
                                            std::make_pair(downsampled_target_cloud, query_cloud),              //
                                            std::make_pair(target_cloud, query_cloud)}) {
            std::cout << "Benchmark case " << ++case_index << std::endl;
            for (const size_t k : k_values) {
                const auto warmup_octree = run_octree_iteration(k, target, query);
                const auto warmup_kdtree = run_kdtree_iteration(k, target, query);

                ASSERT_EQ(std::get<2>(warmup_octree).query_size, query.size());
                ASSERT_EQ(std::get<2>(warmup_octree).k, k);
                ASSERT_EQ(std::get<2>(warmup_kdtree).query_size, query.size());
                ASSERT_EQ(std::get<2>(warmup_kdtree).k, k);

                double total_octree_build_ms = 0.0;
                double total_octree_search_ms = 0.0;
                for (size_t run = 0; run < benchmark_runs; ++run) {
                    auto [build_ms, search_ms, result] = run_octree_iteration(k, target, query);
                    ASSERT_EQ(result.query_size, query.size());
                    ASSERT_EQ(result.k, k);
                    total_octree_build_ms += build_ms;
                    total_octree_search_ms += search_ms;
                }

                double total_kdtree_build_ms = 0.0;
                double total_kdtree_search_ms = 0.0;
                for (size_t run = 0; run < benchmark_runs; ++run) {
                    auto [build_ms, search_ms, result] = run_kdtree_iteration(k, target, query);
                    ASSERT_EQ(result.query_size, query.size());
                    ASSERT_EQ(result.k, k);
                    total_kdtree_build_ms += build_ms;
                    total_kdtree_search_ms += search_ms;
                }

                const double average_octree_build_ms = total_octree_build_ms / static_cast<double>(benchmark_runs);
                const double average_octree_search_ms = total_octree_search_ms / static_cast<double>(benchmark_runs);
                const double average_kdtree_build_ms = total_kdtree_build_ms / static_cast<double>(benchmark_runs);
                const double average_kdtree_search_ms = total_kdtree_search_ms / static_cast<double>(benchmark_runs);

                std::cout << "  Dataset benchmark: target_points=" << target.size() << ", query_points=" << query.size()
                          << ", k=" << k << std::endl;
                std::cout << "    Average Octree build time over " << benchmark_runs
                          << " runs: " << average_octree_build_ms << " ms" << std::endl;
                std::cout << "    Average Octree search time over " << benchmark_runs
                          << " runs: " << average_octree_search_ms << " ms" << std::endl;
                std::cout << "    Average KDTree build time over " << benchmark_runs
                          << " runs: " << average_kdtree_build_ms << " ms" << std::endl;
                std::cout << "    Average KDTree search time over " << benchmark_runs
                          << " runs: " << average_kdtree_search_ms << " ms" << std::endl;
            }
        }

        SUCCEED();
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

TEST(OctreeTest, DynamicUpdatesFollowDesign) {
    try {
        sycl::device device = sycl::device(sycl_points::sycl_utils::device_selector::default_selector_v);
        sycl_points::sycl_utils::DeviceQueue queue(device);

        const size_t k = 3;
        std::vector<sycl_points::PointType> target_points = {
            sycl_points::PointType(-1.0f, -1.0f, -1.0f, 1.0f),
            sycl_points::PointType(1.0f, -1.0f, 0.5f, 1.0f),
            sycl_points::PointType(-0.5f, 1.0f, 1.0f, 1.0f),
            sycl_points::PointType(0.75f, 0.75f, -0.75f, 1.0f),
        };

        auto make_shared_cloud = [&](const std::vector<sycl_points::PointType>& points) {
            sycl_points::PointCloudCPU cpu_cloud;
            cpu_cloud.points->resize(points.size());
            for (size_t i = 0; i < points.size(); ++i) {
                (*cpu_cloud.points)[i] = points[i];
            }
            return sycl_points::PointCloudShared(queue, cpu_cloud);
        };

        std::vector<sycl_points::PointType> query_points = {
            sycl_points::PointType(0.0f, 0.0f, 0.0f, 1.0f),
            sycl_points::PointType(-0.5f, 0.75f, 0.25f, 1.0f),
        };

        sycl_points::PointCloudCPU query_cpu;
        query_cpu.points->resize(query_points.size());
        for (size_t i = 0; i < query_points.size(); ++i) {
            (*query_cpu.points)[i] = query_points[i];
        }
        sycl_points::PointCloudShared query_cloud(queue, query_cpu);

        auto tree = sycl_points::algorithms::knn::Octree::build(queue, make_shared_cloud(target_points), 0.25f, 2);

        auto verify_against_brute_force = [&](std::vector<sycl_points::PointType>& reference_points) {
            reference_points = tree->snapshot_points();
            auto snapshot_ids = tree->snapshot_ids();
            auto reference_cloud = make_shared_cloud(reference_points);
            auto octree_result = tree->knn_search(query_cloud, k);
            auto brute_result =
                sycl_points::algorithms::knn::knn_search_bruteforce(queue, query_cloud, reference_cloud, k);

            for (size_t idx = 0; idx < brute_result.indices->size(); ++idx) {
                const int32_t compact_index = (*brute_result.indices)[idx];
                if (compact_index >= 0 && static_cast<size_t>(compact_index) < snapshot_ids.size()) {
                    (*brute_result.indices)[idx] = snapshot_ids[static_cast<size_t>(compact_index)];
                } else {
                    (*brute_result.indices)[idx] = -1;
                }
            }

            compareKNNResults(octree_result, brute_result, k, 1e-3f);
        };

        verify_against_brute_force(target_points);

        const sycl_points::PointType inserted_point(1.5f, 1.5f, 1.5f, 1.0f);
        tree->insert(inserted_point);
        target_points.push_back(inserted_point);
        EXPECT_EQ(tree->size(), target_points.size());
        verify_against_brute_force(target_points);

        EXPECT_TRUE(tree->remove(inserted_point, 1e-4f));
        target_points.pop_back();
        EXPECT_EQ(tree->size(), target_points.size());
        verify_against_brute_force(target_points);

        sycl_points::algorithms::knn::Octree::BoundingBox remove_region{};
        remove_region.min_bounds = Eigen::Vector3f(-1.1f, -1.1f, -1.1f);
        remove_region.max_bounds = Eigen::Vector3f(0.0f, 0.0f, 0.25f);
        const size_t removed_count = tree->delete_box(remove_region);
        target_points.erase(std::remove_if(target_points.begin(), target_points.end(),
                                           [&](const auto& point) { return remove_region.contains(point); }),
                            target_points.end());
        EXPECT_EQ(removed_count, 1u);
        EXPECT_EQ(tree->size(), target_points.size());
        verify_against_brute_force(target_points);

        const auto radius_hits = tree->radius_search(sycl_points::PointType(-0.4f, 0.9f, 0.9f, 1.0f), 0.5f);
        EXPECT_FALSE(radius_hits.empty());
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}
