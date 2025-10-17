#include <gtest/gtest.h>

#include <random>
#include <sycl_points/algorithms/knn_search.hpp>
#include <sycl_points/algorithms/octree.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/sycl_utils.hpp>

namespace {

void generateRandomPoints(std::shared_ptr<sycl_points::PointContainerCPU> points, size_t num_points, float range,
                          std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-range, range);
    for (size_t i = 0; i < num_points; ++i) {
        (*points)[i] = sycl_points::PointType(dist(gen), dist(gen), dist(gen), 1.0f);
    }
}

void compareKNNResults(const sycl_points::algorithms::knn_search::KNNResult& lhs,
                       const sycl_points::algorithms::knn_search::KNNResult& rhs, size_t k, float epsilon = 1e-4f) {
    ASSERT_EQ(lhs.query_size, rhs.query_size);
    ASSERT_EQ(lhs.k, rhs.k);

    for (size_t i = 0; i < lhs.query_size; ++i) {
        for (size_t j = 0; j < k; ++j) {
            const size_t offset = i * k + j;
            EXPECT_NEAR((*lhs.distances)[offset], (*rhs.distances)[offset], epsilon)
                << "Distance mismatch at query " << i << ", neighbour " << j;
            EXPECT_EQ((*lhs.indices)[offset], (*rhs.indices)[offset])
                << "Index mismatch at query " << i << ", neighbour " << j;
        }
    }
}

}  // namespace

TEST(OctreeTest, CompareWithBruteForceInterfaceOnly) {
    try {
        const size_t num_target_points = 256;
        const size_t num_query_points = 64;
        const size_t k = 4;
        const float point_range = 10.0f;

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

        auto octree = sycl_points::algorithms::octree::Octree::build(queue, target_cloud, 0.5f);
        auto octree_result = octree->knn_search(query_cloud, k);
        auto bruteforce_result =
            sycl_points::algorithms::knn_search::knn_search_bruteforce(queue, query_cloud, target_cloud, k);

        compareKNNResults(octree_result, bruteforce_result, k);
    } catch (const sycl::exception& e) {
        FAIL() << "SYCL exception caught: " << e.what();
    }
}

