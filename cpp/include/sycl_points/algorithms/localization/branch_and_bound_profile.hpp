#pragma once

#include <cstddef>
#include <vector>

namespace sycl_points::algorithms::localization {

/// @brief Timing and candidate counters for one search level.
/// @note The score time and the matrix build time are part of the parent flush.
struct BranchAndBoundLevelProfile {
    size_t evaluated = 0;          ///< Candidates that the KDTree scored.
    size_t queued = 0;             ///< Candidates that passed the score gate.
    size_t branched = 0;           ///< Expanded nodes at this level.
    size_t pruned = 0;             ///< Candidates removed at this level by the score bound.
    size_t leaves = 0;             ///< Accepted leaves at this level.
    double score_ms = 0.0;         ///< KDTree scoring time at this level.
    double matrix_build_ms = 0.0;  ///< Transform matrix build time at this level.
    double branch_ms = 0.0;        ///< Child expansion time at this level.
};

/// @brief Detailed timing and counters for one branch-and-bound search.
/// @note The phase times do not overlap. The flush breakdown overlaps the flush phases.
struct BranchAndBoundProfile {
    // Non-overlapping phases. The sum is close to total_ms.
    double prepare_ms = 0.0;             ///< Input checks, angular info, and grid setup.
    double initial_enumerate_ms = 0.0;   ///< Candidate generation for the initial grid.
    double initial_flush_ms = 0.0;       ///< Scoring of the initial grid.
    double search_flush_ms = 0.0;        ///< Scoring during the best-first search.
    double search_branch_ms = 0.0;       ///< Child expansion during the search.
    double search_bookkeeping_ms = 0.0;  ///< Queue pop, prune, and leaf update during the search.
    double finalize_ms = 0.0;            ///< Result assembly.

    // Breakdown of all flush calls. These intervals overlap the phase times above.
    double flush_total_ms = 0.0;
    double flush_partition_ms = 0.0;     ///< Grouping of the stock by level.
    double flush_matrix_build_ms = 0.0;  ///< Transform matrix build.
    double flush_score_wait_ms = 0.0;    ///< Score kernel and wait.
    double flush_queue_ms = 0.0;         ///< Score store and queue push.

    // Counters.
    size_t initial_candidates = 0;
    size_t flush_calls = 0;
    size_t score_batches = 0;  ///< Number of KDTree kernel launches.
    size_t branch_calls = 0;
    size_t pruned_candidates = 0;
    size_t queued_candidates = 0;
    size_t leaf_candidates = 0;
    size_t max_queue_size = 0;

    double total_ms = 0.0;  ///< Full call time, including the input checks.

    /// @brief Per-level statistics. The index is the level number.
    std::vector<BranchAndBoundLevelProfile> levels;
};

}  // namespace sycl_points::algorithms::localization
