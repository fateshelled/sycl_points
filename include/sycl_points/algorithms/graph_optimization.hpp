#pragma once

#include <map>
#include <set>
#include <sycl_points/algorithms/knn_search.hpp>
#include <sycl_points/algorithms/registration.hpp>
#include <sycl_points/points/point_cloud.hpp>
#include <sycl_points/utils/eigen_utils.hpp>
#include <unordered_map>
#include <vector>

namespace sycl_points {
namespace algorithms {
namespace graph_slam {

/// @brief Pose change threshold for constraint invalidation
struct PoseChangeThreshold {
    float translation_threshold = 1e-2f;   ///< Translation change threshold in meters (default: 1 cm)
    float rotation_threshold = 0.001745f;  ///< Rotation change threshold in radians (default: 0.1 degrees)
};

/// @brief Graph node containing point cloud and pose information
struct GraphNode {
    uint32_t id;                        ///< Unique node identifier
    PointCloudShared::Ptr point_cloud;  ///< Associated point cloud data
    knn_search::KDTree::Ptr kdtree;     ///< Precomputed KDTree for correspondence search
    Eigen::Isometry3f pose;             ///< Current optimized 6DOF pose
    Eigen::Isometry3f initial_pose;     ///< Initial pose estimate
    double timestamp;                   ///< Observation timestamp
    uint32_t priority = 0;              ///< Priority for node removal (lower = higher priority)
    bool is_keyframe = false;           ///< Whether this node is marked as a keyframe

    GraphNode() : timestamp(0.0) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Edge constraint with information matrix and Jacobians
struct EdgeConstraint {
    Eigen::Matrix<float, 6, 6> information_matrix =
        Eigen::Matrix<float, 6, 6>::Identity();  ///< Information matrix (inverse covariance)
    Eigen::Matrix<float, 6, 1> residual = Eigen::Matrix<float, 6, 1>::Zero();  ///< Constraint residual vector
    Eigen::Matrix<float, 6, 6> J_from = Eigen::Matrix<float, 6, 6>::Zero();    ///< Jacobian w.r.t. source node pose
    Eigen::Matrix<float, 6, 6> J_to = Eigen::Matrix<float, 6, 6>::Zero();      ///< Jacobian w.r.t. target node pose
    float error = 0.0f;                                                        ///< Constraint error value
    uint32_t inlier = 0;                                                       ///< Number of inlier correspondences

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Graph edge representing constraints between nodes
struct GraphEdge {
    uint32_t from_node_id;
    uint32_t to_node_id;
    bool enabled = true;

    /// @brief Edge type for different constraint scenarios
    enum class EdgeType {
        SEQUENTIAL,      ///< Consecutive frames with strict thresholds and high weight
        LOOP_CLOSURE,    ///< Loop closure constraints with robust estimation and lower weight
        SUBMAP_TO_SCAN,  ///< Submap to scan matching with adaptive correspondence distance
        CUSTOM           ///< User-defined parameters
    } type = EdgeType::SEQUENTIAL;

    registration::RegistrationParams gicp_params;
    float weight = 1.0f;
    PoseChangeThreshold pose_threshold;

    /// @brief Computation state for efficient caching and invalidation
    enum class ComputationState {
        UNCOMPUTED,  ///< Constraint not computed yet
        COMPUTED,    ///< Constraint computed and cached, ready for optimization
        INVALID,     ///< Constraint invalidated due to significant pose changes
        FAILED       ///< GICP matching failed (insufficient inliers, etc.)
    } state = ComputationState::UNCOMPUTED;

    EdgeConstraint cached_constraint;

    // Poses when constraint was computed (for invalidation check)
    Eigen::Isometry3f from_pose_when_computed;
    Eigen::Isometry3f to_pose_when_computed;

    /// @brief Default constructor with default parameters
    GraphEdge() { set_default_params(); }

    /// @brief Constructor with specific edge type and corresponding parameters
    /// @param edge_type Type of edge constraint (SEQUENTIAL, LOOP_CLOSURE, etc.)
    GraphEdge(EdgeType edge_type) : type(edge_type) { set_params_for_type(edge_type); }

    /// @brief Check if edge constraint has been computed and cached
    /// @return True if constraint is computed and valid
    bool is_computed() const { return state == ComputationState::COMPUTED; }

    /// @brief Check if edge constraint needs recomputation
    /// @return True if constraint is uncomputed or invalidated
    bool needs_recomputation() const {
        return state == ComputationState::UNCOMPUTED || state == ComputationState::INVALID;
    }

    /// @brief Check if poses have changed beyond threshold since last computation
    /// @param from_pose Current pose of the source node
    /// @param to_pose Current pose of the target node
    /// @return True if poses have changed significantly and constraint should be recomputed
    bool is_pose_changed(const Eigen::Isometry3f& from_pose, const Eigen::Isometry3f& to_pose) const {
        if (state != ComputationState::COMPUTED) return false;

        const auto from_diff = from_pose.inverse() * from_pose_when_computed;
        const float from_trans_change = from_diff.translation().norm();
        const float from_rot_change = std::acos(std::clamp((from_diff.linear().trace() - 1.0f) / 2.0f, -1.0f, 1.0f));

        const auto to_diff = to_pose.inverse() * to_pose_when_computed;
        const float to_trans_change = to_diff.translation().norm();
        const float to_rot_change = std::acos(std::clamp((to_diff.linear().trace() - 1.0f) / 2.0f, -1.0f, 1.0f));

        return from_trans_change > pose_threshold.translation_threshold ||
               from_rot_change > pose_threshold.rotation_threshold ||
               to_trans_change > pose_threshold.translation_threshold ||
               to_rot_change > pose_threshold.rotation_threshold;
    }

private:
    void set_default_params() {
        gicp_params.max_iterations = 10;
        gicp_params.max_correspondence_distance = 1.0f;
        gicp_params.translation_eps = 1e-3f;
        gicp_params.rotation_eps = 1e-3f;
        gicp_params.robust_loss = registration::RobustLossType::NONE;
        pose_threshold = {1e-3f, 1e-3f};
        weight = 1.0f;
    }

    void set_params_for_type(EdgeType edge_type) {
        set_default_params();
        switch (edge_type) {
            case EdgeType::SEQUENTIAL:
                gicp_params.max_correspondence_distance = 0.5f;
                pose_threshold = {1e-3f, 5e-4f};
                weight = 1.0f;
                break;

            case EdgeType::LOOP_CLOSURE:
                gicp_params.max_correspondence_distance = 2.0f;
                gicp_params.robust_loss = registration::RobustLossType::HUBER;
                gicp_params.robust_scale = 0.5f;
                pose_threshold = {5e-3f, 2e-3f};
                weight = 0.5f;
                break;

            case EdgeType::SUBMAP_TO_SCAN:
                gicp_params.max_correspondence_distance = 1.5f;
                gicp_params.adaptive_correspondence_distance = true;
                gicp_params.inlier_ratio = 0.6f;
                pose_threshold = {2e-3f, 1e-3f};
                weight = 0.8f;
                break;

            case EdgeType::CUSTOM:
                break;
        }
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Graph SLAM pose optimization
class PoseGraph {
public:
    using Ptr = std::shared_ptr<PoseGraph>;

    /// @brief Constructor
    /// @param queue SYCL device queue for computation
    PoseGraph(const sycl_utils::DeviceQueue& queue) : queue_(queue) {}

    /// @brief Add node to graph
    /// @param node_id Unique identifier for the node
    /// @param cloud Point cloud data associated with this node
    /// @param initial_pose Initial 6DOF pose estimate for this node
    /// @param timestamp Time stamp of the observation (default: 0.0)
    /// @param priority Priority for node removal (lower = higher priority, default: 0)
    /// @param is_keyframe Whether this node is a keyframe (default: false)
    void add_node(uint32_t node_id, PointCloudShared::Ptr cloud, const Eigen::Isometry3f& initial_pose,
                  double timestamp = 0.0, uint32_t priority = 0, bool is_keyframe = false) {
        if (this->nodes_.count(node_id) > 0) {
            throw std::runtime_error("Node ID " + std::to_string(node_id) + " already exists");
        }

        GraphNode node;
        node.id = node_id;
        node.point_cloud = cloud;
        node.kdtree = knn_search::KDTree::build(this->queue_, *cloud);
        node.pose = initial_pose;
        node.initial_pose = initial_pose;
        node.timestamp = timestamp;
        node.priority = priority;
        node.is_keyframe = is_keyframe;

        this->nodes_[node_id] = std::move(node);
        mark_node_edges_for_recomputation(node_id);
    }

    /// @brief Add edge between nodes with predefined parameters
    /// @param from_id Source node ID
    /// @param to_id Target node ID
    /// @param type Type of edge constraint (default: SEQUENTIAL)
    void add_edge(uint32_t from_id, uint32_t to_id, GraphEdge::EdgeType type = GraphEdge::EdgeType::SEQUENTIAL) {
        if (this->nodes_.count(from_id) == 0 || this->nodes_.count(to_id) == 0) {
            throw std::runtime_error("Invalid node ID in edge: " + std::to_string(from_id) + " -> " +
                                     std::to_string(to_id));
        }

        GraphEdge edge(type);
        edge.from_node_id = from_id;
        edge.to_node_id = to_id;

        this->edges_.push_back(edge);
        this->edges_need_computation_.insert(this->edges_.size() - 1);
    }

    /// @brief Add edge between nodes with custom parameters
    /// @param from_id Source node ID
    /// @param to_id Target node ID
    /// @param params Custom GICP registration parameters for this edge
    /// @param weight Weight factor for this constraint in optimization (default: 1.0)
    void add_edge(uint32_t from_id, uint32_t to_id, const registration::RegistrationParams& params,
                  float weight = 1.0f) {
        if (nodes_.count(from_id) == 0 || nodes_.count(to_id) == 0) {
            throw std::runtime_error("Invalid node ID in edge");
        }

        GraphEdge edge;
        edge.from_node_id = from_id;
        edge.to_node_id = to_id;
        edge.gicp_params = params;
        edge.weight = weight;
        edge.type = GraphEdge::EdgeType::CUSTOM;

        this->edges_.push_back(edge);
        this->edges_need_computation_.insert(this->edges_.size() - 1);
    }

    /// @brief Remove node and all associated edges from graph
    /// @param node_id ID of the node to remove
    /// @return True if node was successfully removed, false if node not found
    bool remove_node(uint32_t node_id) {
        if (nodes_.count(node_id) == 0) return false;

        this->edges_.erase(std::remove_if(this->edges_.begin(), this->edges_.end(),
                                          [node_id](const GraphEdge& edge) {
                                              return edge.from_node_id == node_id || edge.to_node_id == node_id;
                                          }),
                           this->edges_.end());

        this->nodes_.erase(node_id);
        return true;
    }

    /// @brief Execute graph SLAM optimization to refine all node poses
    /// @param global_params Optimization parameters including max iterations, convergence thresholds, and verbosity
    /// @return Registration result containing convergence status, final error, iteration count, and inlier count
    registration::RegistrationResult optimize(const registration::RegistrationParams& global_params) {
        registration::RegistrationResult result;

        if (this->nodes_.empty()) return result;

        const auto reference_node_id = get_reference_node_id();
        if (global_params.verbose) {
            std::cout << "Reference node: " << reference_node_id << std::endl;
        }

        const PoseChangeThreshold global_threshold{1e-3f, 1e-3f};

        for (size_t iter = 0; iter < global_params.max_iterations; ++iter) {
            // 1. Update invalidated constraints due to pose changes
            update_invalidated_constraints();

            // 2. Compute required edge constraints
            compute_required_constraints();

            // 3. Build global linear system
            const auto [H_global, b_global] = build_global_system_fixed_reference();

            if (H_global.rows() == 0) {
                if (global_params.verbose) {
                    std::cout << "No movable nodes in system" << std::endl;
                }
                break;
            }

            if (std::abs(H_global.determinant()) < 1e-12f) {
                if (global_params.verbose) {
                    std::cout << "System is singular at iteration " << iter << std::endl;
                }
                break;
            }

            // 4. Update poses (excluding reference node)
            const auto delta = solve_linear_system(H_global, b_global);
            update_movable_poses(delta, reference_node_id);

            // 5. Convergence check
            float max_trans_change = 0.0f, max_rot_change = 0.0f;
            const size_t movable_count = count_movable_nodes(reference_node_id);

            for (size_t i = 0; i < movable_count; ++i) {
                const auto delta_i = delta.segment<6>(i * 6);
                max_trans_change = std::max(max_trans_change, delta_i.tail<3>().norm());
                max_rot_change = std::max(max_rot_change, delta_i.head<3>().norm());
            }

            result.converged =
                max_trans_change < global_params.translation_eps && max_rot_change < global_params.rotation_eps;

            if (global_params.verbose) {
                std::cout << "Iteration " << iter << ": "
                          << "max_trans=" << max_trans_change << ", "
                          << "max_rot=" << max_rot_change << std::endl;
            }

            result.iterations = iter;

            if (result.converged) {
                break;
            }
        }

        result.error = compute_total_error();
        result.inlier = count_total_inliers();

        return result;
    }

    /// @brief Get optimized pose of a specific node
    /// @param node_id ID of the node to query
    /// @return 6DOF pose as SE(3) transformation matrix
    /// @throws std::runtime_error if node_id is not found
    Eigen::Isometry3f get_pose(uint32_t node_id) const {
        auto it = this->nodes_.find(node_id);
        if (it == this->nodes_.end()) {
            throw std::runtime_error("Node not found: " + std::to_string(node_id));
        }
        return it->second.pose;
    }

    /// @brief Get point cloud associated with a specific node
    /// @param node_id ID of the node to query
    /// @return Shared pointer to the point cloud data
    /// @throws std::runtime_error if node_id is not found
    PointCloudShared::Ptr get_node_cloud(uint32_t node_id) const {
        auto it = this->nodes_.find(node_id);
        if (it == this->nodes_.end()) {
            throw std::runtime_error("Node not found: " + std::to_string(node_id));
        }
        return it->second.point_cloud;
    }

    /// @brief Get all optimized poses in the graph
    /// @return Map from node ID to corresponding 6DOF pose
    std::map<int, Eigen::Isometry3f> get_all_poses() const {
        std::map<int, Eigen::Isometry3f> poses;
        for (const auto& [id, node] : this->nodes_) {
            poses[id] = node.pose;
        }
        return poses;
    }

    /// @brief Get total number of nodes in the graph
    /// @return Number of nodes currently in the graph
    size_t node_count() const { return this->nodes_.size(); }

    /// @brief Get total number of edges in the graph
    /// @return Number of edges currently in the graph
    size_t edge_count() const { return this->edges_.size(); }

    /// @brief Check if a node exists in the graph
    /// @param node_id ID of the node to check
    /// @return True if node exists, false otherwise
    bool has_node(uint32_t node_id) const { return this->nodes_.count(node_id) > 0; }

private:
    sycl_utils::DeviceQueue queue_;
    std::unordered_map<uint32_t, GraphNode> nodes_;
    std::vector<GraphEdge> edges_;

    std::set<size_t> edges_need_computation_;

    uint32_t get_reference_node_id() const {
        if (this->nodes_.empty()) return std::numeric_limits<uint32_t>::max();

        uint32_t min_id = std::numeric_limits<uint32_t>::max();
        for (const auto& [id, node] : this->nodes_) {
            min_id = std::min(min_id, id);
        }
        return min_id;
    }

    size_t count_movable_nodes(uint32_t reference_node_id) const {
        size_t count = 0;
        for (const auto& [id, node] : this->nodes_) {
            if (id != reference_node_id) count++;
        }
        return count;
    }

    /// @brief Compute SE(3) Adjoint matrix: [R, [t]×R; 0, R]
    Eigen::Matrix<float, 6, 6> adjoint(const Eigen::Isometry3f& T) const {
        Eigen::Matrix<float, 6, 6> Ad = Eigen::Matrix<float, 6, 6>::Zero();
        const Eigen::Matrix3f R = T.linear();
        const Eigen::Vector3f t = T.translation();

        Ad.block<3, 3>(0, 0) = R;
        Ad.block<3, 3>(3, 3) = R;
        Ad.block<3, 3>(0, 3) = eigen_utils::lie::skew(t) * R;

        return Ad;
    }

    /// @brief Build global system with fixed reference node
    std::pair<Eigen::MatrixXf, Eigen::VectorXf> build_global_system_fixed_reference() {
        const auto reference_node_id = get_reference_node_id();
        const auto num_movable_nodes = count_movable_nodes(reference_node_id);

        if (num_movable_nodes == 0) {
            return {Eigen::MatrixXf::Zero(0, 0), Eigen::VectorXf::Zero(0)};
        }

        size_t system_size = num_movable_nodes * 6;
        Eigen::MatrixXf H_global = Eigen::MatrixXf::Zero(system_size, system_size);
        Eigen::VectorXf b_global = Eigen::VectorXf::Zero(system_size);

        // Map node IDs to indices (excluding reference node)
        std::map<uint32_t, size_t> node_id_to_index;
        size_t index = 0;
        for (const auto& [id, node] : this->nodes_) {
            if (id != reference_node_id) {
                node_id_to_index[id] = index++;
            }
        }

        for (const auto& edge : this->edges_) {
            if (!edge.enabled || edge.state != GraphEdge::ComputationState::COMPUTED) continue;

            const bool from_is_reference = (edge.from_node_id == reference_node_id);
            const bool to_is_reference = (edge.to_node_id == reference_node_id);

            if (from_is_reference && to_is_reference) continue;

            const auto& constraint = edge.cached_constraint;
            const float w = edge.weight;

            const auto& info_matrix = constraint.information_matrix;
            const auto& residual = constraint.residual;
            const auto& J_from = constraint.J_from;
            const auto& J_to = constraint.J_to;

            if (from_is_reference) {
                const size_t to_idx = node_id_to_index[edge.to_node_id];
                const size_t to_offset = to_idx * 6;

                H_global.block<6, 6>(to_offset, to_offset) += w * (J_to.transpose() * info_matrix * J_to);
                b_global.segment<6>(to_offset) += w * (J_to.transpose() * info_matrix * residual);

            } else if (to_is_reference) {
                const size_t from_idx = node_id_to_index[edge.from_node_id];
                const size_t from_offset = from_idx * 6;

                H_global.block<6, 6>(from_offset, from_offset) += w * (J_from.transpose() * info_matrix * J_from);
                b_global.segment<6>(from_offset) += w * (J_from.transpose() * info_matrix * residual);

            } else {
                const size_t from_idx = node_id_to_index[edge.from_node_id];
                const size_t to_idx = node_id_to_index[edge.to_node_id];

                const size_t from_offset = from_idx * 6;
                const size_t to_offset = to_idx * 6;

                // Hessian blocks: H = J^T * Λ * J
                const auto H_from_from = J_from.transpose() * info_matrix * J_from;
                const auto H_to_to = J_to.transpose() * info_matrix * J_to;
                const auto H_from_to = J_from.transpose() * info_matrix * J_to;
                const auto H_to_from = J_to.transpose() * info_matrix * J_from;

                H_global.block<6, 6>(from_offset, from_offset) += w * H_from_from;
                H_global.block<6, 6>(to_offset, to_offset) += w * H_to_to;
                H_global.block<6, 6>(from_offset, to_offset) += w * H_from_to;
                H_global.block<6, 6>(to_offset, from_offset) += w * H_to_from;

                // Gradient: b = J^T * Λ * e
                const auto b_from = J_from.transpose() * info_matrix * residual;
                const auto b_to = J_to.transpose() * info_matrix * residual;

                b_global.segment<6>(from_offset) += w * b_from;
                b_global.segment<6>(to_offset) += w * b_to;
            }
        }

        return {H_global, b_global};
    }

    /// @brief Update poses on SE(3) manifold (excluding reference node)
    void update_movable_poses(const Eigen::VectorXf& delta, uint32_t reference_node_id) {
        std::map<uint32_t, size_t> node_id_to_index;
        size_t index = 0;
        for (const auto& [id, node] : this->nodes_) {
            if (id != reference_node_id) {
                node_id_to_index[id] = index++;
            }
        }

        for (auto& [id, node] : this->nodes_) {
            if (id == reference_node_id) continue;

            const size_t idx = node_id_to_index[id];
            const auto delta_i = delta.segment<6>(idx * 6);

            node.pose = node.pose * eigen_utils::lie::se3_exp(delta_i);
        }
    }

    void mark_node_edges_for_recomputation(uint32_t node_id) {
        for (size_t i = 0; i < this->edges_.size(); ++i) {
            const auto& edge = this->edges_[i];
            if (edge.from_node_id == node_id || edge.to_node_id == node_id) {
                this->edges_need_computation_.insert(i);
                this->edges_[i].state = GraphEdge::ComputationState::UNCOMPUTED;
            }
        }
    }

    void update_invalidated_constraints() {
        for (size_t i = 0; i < this->edges_.size(); ++i) {
            auto& edge = this->edges_[i];

            if (edge.state != GraphEdge::ComputationState::COMPUTED) continue;

            const auto& current_from_pose = this->nodes_[edge.from_node_id].pose;
            const auto& current_to_pose = this->nodes_[edge.to_node_id].pose;

            if (edge.is_pose_changed(current_from_pose, current_to_pose)) {
                edge.state = GraphEdge::ComputationState::INVALID;
                edges_need_computation_.insert(i);
            }
        }
    }

    void compute_required_constraints() {
        if (this->edges_need_computation_.empty()) return;

        for (size_t edge_idx : this->edges_need_computation_) {
            auto& edge = this->edges_[edge_idx];

            if (!edge.enabled) continue;

            try {
                edge.cached_constraint = compute_edge_constraint(edge);
                edge.state = GraphEdge::ComputationState::COMPUTED;
                edge.from_pose_when_computed = this->nodes_[edge.from_node_id].pose;
                edge.to_pose_when_computed = this->nodes_[edge.to_node_id].pose;

            } catch (const std::exception& e) {
                std::cerr << "Edge computation failed: " << e.what() << std::endl;
                edge.state = GraphEdge::ComputationState::FAILED;
            }
        }

        this->edges_need_computation_.clear();
    }

    /// @brief Compute edge constraint using relative pose
    /// Constraint: e = log(T_obs^(-1) * T_pred) where T_pred = T_from^(-1) * T_to
    EdgeConstraint compute_edge_constraint(const GraphEdge& edge) {
        const auto& from_node = this->nodes_[edge.from_node_id];
        const auto& to_node = this->nodes_[edge.to_node_id];

        // Initial guess from current node poses
        const Eigen::Isometry3f relative_pose_initial = from_node.pose.inverse() * to_node.pose;

        if (edge.gicp_params.verbose) {
            std::cout << "Computing constraint for edge " << edge.from_node_id << " -> " << edge.to_node_id
                      << std::endl;
            std::cout << "  Initial relative pose: " << relative_pose_initial.translation().transpose() << std::endl;
        }

        // Execute GICP registration
        auto registration = std::make_shared<registration::RegistrationGICP>(this->queue_, edge.gicp_params);

        const auto gicp_result = registration->align(*from_node.point_cloud, *to_node.point_cloud, *to_node.kdtree,
                                               relative_pose_initial.matrix());

        if (edge.gicp_params.verbose) {
            std::cout << "  GICP result: " << gicp_result.T.translation().transpose() << std::endl;
            std::cout << "  Error: " << gicp_result.error << ", Inliers: " << gicp_result.inlier << std::endl;
        }

        // Observed relative pose from GICP
        const Eigen::Isometry3f T_obs = Eigen::Isometry3f(gicp_result.T.matrix());

        // Predicted relative pose from current node poses
        const Eigen::Isometry3f T_pred = from_node.pose.inverse() * to_node.pose;

        // Residual: e = log(T_obs^(-1) * T_pred)
        const Eigen::Isometry3f error_transform = T_obs.inverse() * T_pred;
        const Eigen::Matrix<float, 6, 1> residual = eigen_utils::lie::se3_log(error_transform);

        // Jacobians for constraint e = log(T_obs^(-1) * T_from^(-1) * T_to)
        // J_from = ∂e/∂ξ_from = -Ad(T_pred)
        // J_to = ∂e/∂ξ_to = I_6×6
        const Eigen::Matrix<float, 6, 6> J_from = -adjoint(T_pred);
        const Eigen::Matrix<float, 6, 6> J_to = Eigen::Matrix<float, 6, 6>::Identity();

        // Information matrix from GICP Hessian
        Eigen::Matrix<float, 6, 6> information_matrix = gicp_result.H;

        // Ensure positive definiteness
        if (std::abs(information_matrix.determinant()) < 1e-12f) {
            information_matrix += 1e-6f * Eigen::Matrix<float, 6, 6>::Identity();
        }

        EdgeConstraint constraint;
        constraint.information_matrix = information_matrix;
        constraint.residual = residual;
        constraint.J_from = J_from;
        constraint.J_to = J_to;
        constraint.error = 0.5f * residual.transpose() * information_matrix * residual;
        constraint.inlier = gicp_result.inlier;

        return constraint;
    }

    Eigen::VectorXf solve_linear_system(const Eigen::MatrixXf& H, const Eigen::VectorXf& b) const {
        return H.ldlt().solve(-b);
    }

    float compute_total_error() const {
        float total_error = 0.0f;
        for (const auto& edge : this->edges_) {
            if (edge.enabled && edge.state == GraphEdge::ComputationState::COMPUTED) {
                total_error += edge.weight * edge.cached_constraint.error;
            }
        }
        return total_error;
    }

    size_t count_total_inliers() const {
        size_t total_inliers = 0;
        for (const auto& edge : this->edges_) {
            if (edge.enabled && edge.state == GraphEdge::ComputationState::COMPUTED) {
                total_inliers += edge.cached_constraint.inlier;
            }
        }
        return total_inliers;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace graph_slam
}  // namespace algorithms
}  // namespace sycl_points