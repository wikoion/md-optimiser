// Core headers from the OR-Tools Constraint Programming (CP-SAT) solver
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"

#include <unordered_set>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdint>

// Bring OR-Tools namespace into scope
using namespace operations_research;
namespace sat = operations_research::sat;

using namespace std;

namespace std {
    template <>
    struct hash<std::pair<int, int>> {
        std::size_t operator()(const std::pair<int, int>& p) const noexcept {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };
}

extern "C" {

// Describes a scalable group of machines with a type, size, and max count
struct MachineDeployment {
    const char* name;
    double cpu;
    double memory;
    int max_scale_out;
};

// Describes a single workload unit that needs placement
struct Pod {
    double cpu;
    double memory;
    const int* affinity_peers;
    const int* affinity_rules;
    int affinity_count;
    const int* soft_peers;
    const double* soft_affinities;
    int soft_affinity_count;
    int current_md_assignment; // Index of MD this pod is currently placed on (-1 if unknown)
};

enum SolverStatus {
    UNKNOWN = 0,
    MODEL_INVALID = 1,
    FEASIBLE = 2,
    INFEASIBLE = 3,
    OPTIMAL = 4
};

struct SolverResult {
    bool success;
    double objective;
    int status_code;
    double solve_time_secs;
    double current_state_cost;  // Cost of current state before optimization
    bool already_optimal;       // True if improvement doesn't meet threshold
    bool used_beam_search;      // True if beam search was used (hint or fallback)
    bool beam_search_fallback;  // True if pure beam search fallback was used
    int cpsat_attempts;         // Number of CP-SAT attempts made
    int best_attempt;           // Which attempt produced the result
    int unplaced_pods;          // Number of pods that couldn't be placed
};
}

class Optimiser {
private:
   int num_mds;
   const MachineDeployment* mds;
   int num_pods;
   const Pod* pods;
   const int* allowed_matrix;
   sat::CpModelBuilder* model;
   vector<int> slots_per_md;
   vector<vector<sat::BoolVar>> slot_used;
   vector<vector<vector<sat::BoolVar>>> pod_slot_assignment;
   
   // Resource tracking for waste calculation
   vector<sat::LinearExpr> md_cpu_assigned;
   vector<sat::LinearExpr> md_mem_assigned;
   vector<sat::LinearExpr> md_nodes_used;

public:
    explicit Optimiser(
       int num_mds, const MachineDeployment* mds,
       int num_pods, const Pod* pods,
       const int* allowed_matrix,
       sat::CpModelBuilder* model
    ) : num_mds(num_mds), mds(mds), num_pods(num_pods), pods(pods), allowed_matrix(allowed_matrix),
        model(model), slots_per_md(num_mds), slot_used(num_mds), pod_slot_assignment(num_pods),
        md_cpu_assigned(num_mds), md_mem_assigned(num_mds), md_nodes_used(num_mds)
    {}

    const vector<int>& GetSlotsPerMd() const { return slots_per_md; };
    const vector<vector<sat::BoolVar>>& GetSlotUsed() const { return slot_used; };
    const vector<vector<vector<sat::BoolVar>>>& GetPodSlotAssignment() const { return pod_slot_assignment; };

    void InitSlotsUsed() {
        for (int i = 0; i < num_mds; ++i) {
            int max_slots = mds[i].max_scale_out;
            slots_per_md[i] = max_slots;
            slot_used[i].resize(max_slots);
            for (int j = 0; j < max_slots; ++j) {
                slot_used[i][j] = model->NewBoolVar();
            }
        }
    };

    // init_pod_slot_assignment_matrix creates a zeroed 3d matrix/tensor of pods(int) to slots_used 
    // (mds int to md slots boolvar)
    void InitPodSlotAssignment() {
        for (int i = 0; i < num_pods; ++i) {
            pod_slot_assignment[i].resize(num_mds);
            for (int j = 0; j < num_mds; ++j) {
                if (!allowed_matrix[i * num_mds + j]) continue;
                int slots = slots_per_md[j];
                pod_slot_assignment[i][j].resize(slots);
                for (int k = 0; k < slots; ++k) {
                    pod_slot_assignment[i][j][k] = model->NewBoolVar();
                }
            }
        }
    };
    
    // Track resource assignments for waste calculation
    void TrackResourceAssignments(int md_idx, int slot_idx, 
                                   const sat::LinearExpr& cpu_sum, 
                                   const sat::LinearExpr& mem_sum) {
        md_cpu_assigned[md_idx] += cpu_sum;
        md_mem_assigned[md_idx] += mem_sum;
    }
    
    // Set the number of nodes used for an MD
    void SetNodesUsed(int md_idx, const sat::LinearExpr& nodes_used) {
        md_nodes_used[md_idx] = nodes_used;
    }
    
    // Calculate waste objective for all MDs
    sat::LinearExpr CalculateWasteObjective() {
        sat::LinearExpr total_waste;
        
        for (int j = 0; j < num_mds; ++j) {
            // Calculate provisioned resources (nodes_used * capacity)
            sat::LinearExpr cpu_provisioned = md_nodes_used[j] * static_cast<int>(mds[j].cpu * 100);
            sat::LinearExpr mem_provisioned = md_nodes_used[j] * static_cast<int>(mds[j].memory * 100);
            
            // Calculate waste (provisioned - assigned)
            // Note: md_cpu_assigned and md_mem_assigned are already scaled by 100
            sat::LinearExpr cpu_waste = cpu_provisioned - md_cpu_assigned[j];
            sat::LinearExpr mem_waste = mem_provisioned - md_mem_assigned[j];
            
            // Add normalized waste to total
            // We normalize by dividing by capacity, but since we're in integer domain,
            // we keep the scale factor
            total_waste += cpu_waste + mem_waste;
        }
        
        return total_waste;
    }
    
    // Build combined objective with waste and plugin scores
    sat::LinearExpr BuildCombinedObjective(const double* plugin_scores, 
                                           const sat::LinearExpr& soft_penalty,
                                           double waste_weight = 0.7) {
        // Calculate waste objective
        sat::LinearExpr waste_objective = CalculateWasteObjective();
        
        // Calculate plugin score objective
        sat::LinearExpr plugin_objective;
        for (int j = 0; j < num_mds; ++j) {
            // Original logic: higher plugin score = lower cost
            int weight = static_cast<int>((1.0 - plugin_scores[j]) * 1000.0) + 1;
            for (int k = 0; k < slots_per_md[j]; ++k) {
                plugin_objective += slot_used[j][k] * weight;
            }
        }
        
        // Treat waste and plugin scores equally
        // Both get the same weight scaling for fair comparison
        int objective_scale = 1000;
        
        sat::LinearExpr combined_objective;
        combined_objective += waste_objective * objective_scale;  // Waste minimization
        combined_objective += plugin_objective * objective_scale;  // Plugin scores
        combined_objective += soft_penalty * 250;  // Soft affinity penalties
        
        return combined_objective;
    }

};

// ============================================================================
// BEAM SEARCH HEURISTIC IMPLEMENTATION
// ============================================================================

// Structure to track pod assignment in beam search algorithm
struct BeamAssignment {
    int md;
    int slot;
};

// Structure to track slot resource usage
struct SlotState {
    bool used;
    double cpu_remaining;
    double mem_remaining;
};

// Helper: Calculate waste for placing a pod in a slot
double CalculateSlotWaste(const MachineDeployment& md, const SlotState& slot_state, const Pod& pod) {
    double remaining_cpu = slot_state.cpu_remaining - pod.cpu;
    double remaining_mem = slot_state.mem_remaining - pod.memory;

    // Normalize waste by capacity
    double cpu_waste_pct = remaining_cpu / md.cpu;
    double mem_waste_pct = remaining_mem / md.memory;

    return cpu_waste_pct + mem_waste_pct;
}

// Helper: Check if pod has anti-affinity conflict with already placed pods
bool HasAntiAffinityConflict(
    int pod_idx,
    int md_idx,
    int slot_idx,
    const Pod* pods,
    const vector<BeamAssignment>& assignments
) {
    const Pod& pod = pods[pod_idx];

    for (int i = 0; i < pod.affinity_count; ++i) {
        int peer_idx = pod.affinity_peers[i];
        int rule = pod.affinity_rules[i];

        if (rule == -1) {  // Anti-affinity
            // Check if peer is already placed
            if (peer_idx >= 0 && peer_idx < static_cast<int>(assignments.size())) {
                const BeamAssignment& peer_assignment = assignments[peer_idx];
                if (peer_assignment.md == md_idx && peer_assignment.slot == slot_idx) {
                    return true;  // Conflict found
                }
            }
        }
    }

    return false;
}

// Helper: Check if pod has colocate affinity preference with already placed pods
bool HasColocateAffinityPreference(
    int pod_idx,
    int md_idx,
    int slot_idx,
    const Pod* pods,
    const vector<BeamAssignment>& assignments
) {
    const Pod& pod = pods[pod_idx];

    for (int i = 0; i < pod.affinity_count; ++i) {
        int peer_idx = pod.affinity_peers[i];
        int rule = pod.affinity_rules[i];

        if (rule == 1) {  // Colocate
            // Check if peer is already placed in this slot
            if (peer_idx >= 0 && peer_idx < static_cast<int>(assignments.size())) {
                const BeamAssignment& peer_assignment = assignments[peer_idx];
                if (peer_assignment.md == md_idx && peer_assignment.slot == slot_idx) {
                    return true;  // Preference match
                }
            }
        }
    }

    return false;
}

// Helper: Calculate soft affinity penalty for placing pod in slot
double CalculateSoftAffinityPenalty(
    int pod_idx,
    int md_idx,
    int slot_idx,
    const Pod* pods,
    const vector<BeamAssignment>& assignments
) {
    const Pod& pod = pods[pod_idx];
    double penalty = 0.0;

    for (int i = 0; i < pod.soft_affinity_count; ++i) {
        int peer_idx = pod.soft_peers[i];
        double affinity = pod.soft_affinities[i];

        if (peer_idx >= 0 && peer_idx < static_cast<int>(assignments.size())) {
            const BeamAssignment& peer_assignment = assignments[peer_idx];

            if (peer_assignment.md != -1) {  // Peer is placed
                bool same_slot = (peer_assignment.md == md_idx && peer_assignment.slot == slot_idx);

                if (affinity > 0) {
                    // Prefer colocate
                    if (!same_slot) {
                        penalty += std::abs(affinity) * 100.0;
                    }
                } else if (affinity < 0) {
                    // Prefer spread
                    if (same_slot) {
                        penalty += std::abs(affinity) * 100.0;
                    }
                }
            }
        }
    }

    return penalty;
}

// Helper: Count valid placements for a pod (for difficulty scoring)
int CountValidPlacements(int pod_idx, const int* allowed_matrix, int num_mds) {
    int count = 0;
    for (int j = 0; j < num_mds; ++j) {
        if (allowed_matrix[pod_idx * num_mds + j]) {
            count++;
        }
    }
    return count;
}

// Helper: Count hard affinity constraints for a pod (for difficulty scoring)
int CountHardAffinityConstraints(const Pod& pod) {
    return pod.affinity_count;
}

// Helper: Sort pods by difficulty (most constrained first)
vector<int> SortPodsByDifficulty(
    const Pod* pods,
    int num_pods,
    const int* allowed_matrix,
    int num_mds
) {
    struct PodDifficulty {
        int pod_idx;
        double difficulty;
    };

    vector<PodDifficulty> difficulties;
    difficulties.reserve(num_pods);

    for (int i = 0; i < num_pods; ++i) {
        int valid_placements = CountValidPlacements(i, allowed_matrix, num_mds);
        int affinity_count = CountHardAffinityConstraints(pods[i]);
        double size = pods[i].cpu + pods[i].memory;

        // Weight: affinity > constraints > size
        double difficulty = 10000.0 * affinity_count +
                           100.0 / std::max(valid_placements, 1) +
                           size;

        difficulties.push_back({i, difficulty});
    }

    // Sort by difficulty descending
    std::sort(difficulties.begin(), difficulties.end(),
              [](const PodDifficulty& a, const PodDifficulty& b) {
                  return a.difficulty > b.difficulty;
              });

    vector<int> pod_order;
    pod_order.reserve(num_pods);
    for (const auto& pd : difficulties) {
        pod_order.push_back(pd.pod_idx);
    }

    return pod_order;
}

// Helper: Sort MDs by score (best first)
vector<int> SortMDsByScore(const double* plugin_scores, int num_mds) {
    struct MDScore {
        int md_idx;
        double score;
    };

    vector<MDScore> scores;
    scores.reserve(num_mds);

    for (int i = 0; i < num_mds; ++i) {
        scores.push_back({i, plugin_scores[i]});
    }

    // Sort by score descending (best first)
    std::sort(scores.begin(), scores.end(),
              [](const MDScore& a, const MDScore& b) {
                  return a.score > b.score;
              });

    vector<int> md_order;
    md_order.reserve(num_mds);
    for (const auto& ms : scores) {
        md_order.push_back(ms.md_idx);
    }

    return md_order;
}

// Helper: Build affinity groups (pods with colocate affinity)
vector<vector<int>> BuildAffinityGroups(const Pod* pods, int num_pods) {
    // Build adjacency list for colocate affinity
    vector<vector<int>> graph(num_pods);

    for (int i = 0; i < num_pods; ++i) {
        const Pod& pod = pods[i];
        for (int j = 0; j < pod.affinity_count; ++j) {
            int peer = pod.affinity_peers[j];
            int rule = pod.affinity_rules[j];

            if (rule == 1 && peer >= 0 && peer < num_pods) {  // Colocate
                graph[i].push_back(peer);
                graph[peer].push_back(i);
            }
        }
    }

    // Find connected components using BFS
    vector<bool> visited(num_pods, false);
    vector<vector<int>> groups;

    for (int i = 0; i < num_pods; ++i) {
        if (!visited[i] && !graph[i].empty()) {
            vector<int> group;
            vector<int> queue;
            queue.push_back(i);
            visited[i] = true;

            while (!queue.empty()) {
                int curr = queue.back();
                queue.pop_back();
                group.push_back(curr);

                for (int neighbor : graph[curr]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }

            if (group.size() > 1) {  // Only track multi-pod groups
                groups.push_back(group);
            }
        }
    }

    return groups;
}

// Helper: Try to place affinity group together
bool TryPlaceAffinityGroup(
    const vector<int>& group,
    const vector<int>& md_order,
    const MachineDeployment* mds,
    const Pod* pods,
    const int* allowed_matrix,
    int num_mds,
    vector<vector<SlotState>>& slot_usage,
    vector<BeamAssignment>& assignments
) {
    // Calculate total resources needed for group
    double total_cpu = 0.0;
    double total_mem = 0.0;
    for (int pod_idx : group) {
        total_cpu += pods[pod_idx].cpu;
        total_mem += pods[pod_idx].memory;
    }

    // PHASE 1: CLUSTER AUTOSCALER STRATEGY - Try ALL existing (used) slots first
    // This is the key fix: prefer packing into existing nodes before creating new ones
    int best_md = -1;
    int best_slot = -1;
    double best_score = -std::numeric_limits<double>::infinity();

    for (int md_idx = 0; md_idx < num_mds; ++md_idx) {
        const MachineDeployment& md = mds[md_idx];

        // Check if all pods in group are allowed on this MD
        bool all_allowed = true;
        for (int pod_idx : group) {
            if (!allowed_matrix[pod_idx * num_mds + md_idx]) {
                all_allowed = false;
                break;
            }
        }
        if (!all_allowed) continue;

        // Try all EXISTING (used) slots in this MD
        for (int slot_idx = 0; slot_idx < md.max_scale_out; ++slot_idx) {
            if (!slot_usage[md_idx][slot_idx].used) {
                continue;  // Skip empty slots in Phase 1
            }

            // Check if group fits in remaining capacity
            if (total_cpu <= slot_usage[md_idx][slot_idx].cpu_remaining &&
                total_mem <= slot_usage[md_idx][slot_idx].mem_remaining) {

                // Score this placement - PREFER FULLER NODES (like Cluster Autoscaler)
                double remaining_cpu = slot_usage[md_idx][slot_idx].cpu_remaining - total_cpu;
                double remaining_mem = slot_usage[md_idx][slot_idx].mem_remaining - total_mem;

                // Calculate utilization after placing group
                double cpu_used_pct = 1.0 - (remaining_cpu / md.cpu);
                double mem_used_pct = 1.0 - (remaining_mem / md.memory);
                double avg_utilization = (cpu_used_pct + mem_used_pct) / 2.0;

                // Higher score = better fit (fuller node)
                double score = avg_utilization * 10000.0;  // 0-10000 range

                // Small bonus for better plugin scores (but utilization dominates)
                score += (md_idx < static_cast<int>(md_order.size()) ?
                         (1000.0 / (1.0 + md_order[md_idx])) : 0.0);

                if (score > best_score) {
                    best_score = score;
                    best_md = md_idx;
                    best_slot = slot_idx;
                }
            }
        }
    }

    // If we found an existing slot with capacity, use it
    if (best_md != -1) {
        for (int pod_idx : group) {
            assignments[pod_idx] = {best_md, best_slot};
        }
        slot_usage[best_md][best_slot].cpu_remaining -= total_cpu;
        slot_usage[best_md][best_slot].mem_remaining -= total_mem;
        return true;
    }

    // PHASE 2: No existing slot found - create new node
    // Try MDs in preferred order
    for (int md_idx : md_order) {
        const MachineDeployment& md = mds[md_idx];

        // Check if all pods in group are allowed on this MD
        bool all_allowed = true;
        for (int pod_idx : group) {
            if (!allowed_matrix[pod_idx * num_mds + md_idx]) {
                all_allowed = false;
                break;
            }
        }
        if (!all_allowed) continue;

        // Check if group fits in MD capacity
        if (total_cpu > md.cpu || total_mem > md.memory) {
            continue;
        }

        // Find first empty slot in this MD
        for (int slot_idx = 0; slot_idx < md.max_scale_out; ++slot_idx) {
            if (slot_usage[md_idx][slot_idx].used) {
                continue;  // Skip used slots in Phase 2
            }

            // Found empty slot - create new node here
            for (int pod_idx : group) {
                assignments[pod_idx] = {md_idx, slot_idx};
            }

            slot_usage[md_idx][slot_idx].used = true;
            slot_usage[md_idx][slot_idx].cpu_remaining -= total_cpu;
            slot_usage[md_idx][slot_idx].mem_remaining -= total_mem;

            return true;
        }
    }

    return false;
}

// Main beam search placement algorithm result
struct BeamResult {
    vector<BeamAssignment> assignments;
    vector<vector<bool>> slots_used;
    bool all_placed;
    int unplaced_count;
};

// Helper: Calculate optimal MD selection based on resource requirements
vector<int> CalculateOptimalMDOrder(
    const MachineDeployment* mds,
    int num_mds,
    const Pod* pods,
    int num_pods,
    const double* plugin_scores,
    const int* allowed_matrix
) {
    // Calculate total resource requirements
    double total_cpu = 0.0;
    double total_mem = 0.0;
    for (int i = 0; i < num_pods; ++i) {
        total_cpu += pods[i].cpu;
        total_mem += pods[i].memory;
    }

    // QUICK WIN #1 IMPROVEMENT: Calculate actual average MD capacity instead of hardcoded assumptions
    double avg_md_cpu = 0.0;
    double avg_md_mem = 0.0;
    int viable_md_count = 0;

    for (int j = 0; j < num_mds; ++j) {
        // Only consider MDs that can host at least one pod
        bool can_place_any = false;
        for (int i = 0; i < num_pods; ++i) {
            if (allowed_matrix[i * num_mds + j] &&
                pods[i].cpu <= mds[j].cpu &&
                pods[i].memory <= mds[j].memory) {
                can_place_any = true;
                break;
            }
        }
        if (can_place_any) {
            avg_md_cpu += mds[j].cpu;
            avg_md_mem += mds[j].memory;
            viable_md_count++;
        }
    }

    if (viable_md_count > 0) {
        avg_md_cpu /= viable_md_count;
        avg_md_mem /= viable_md_count;
    } else {
        avg_md_cpu = 16.0;  // Fallback
        avg_md_mem = 32.0;
    }

    // Estimate minimum nodes needed (using actual MD averages)
    int estimated_nodes = std::max(
        static_cast<int>(std::ceil(total_cpu / avg_md_cpu)),
        static_cast<int>(std::ceil(total_mem / avg_md_mem))
    );
    estimated_nodes = std::max(3, estimated_nodes);  // At least 3 for anti-affinity

    // Calculate target resources per node
    double target_cpu_per_node = total_cpu / estimated_nodes;
    double target_mem_per_node = total_mem / estimated_nodes;

    // Score each MD based on:
    // 1. How well it matches target resources (minimize waste)
    // 2. Plugin score (prefer higher scoring MDs)
    // 3. Availability (can it fit the workload)
    struct MDFitScore {
        int md_idx;
        double score;
    };

    vector<MDFitScore> md_scores;
    for (int j = 0; j < num_mds; ++j) {
        const MachineDeployment& md = mds[j];

        // Check if any pods can be placed on this MD
        bool can_place_any = false;
        for (int i = 0; i < num_pods; ++i) {
            if (allowed_matrix[i * num_mds + j] &&
                pods[i].cpu <= md.cpu &&
                pods[i].memory <= md.memory) {
                can_place_any = true;
                break;
            }
        }

        if (!can_place_any) {
            continue;  // Skip MDs that can't host any pods
        }

        // Calculate fitness score
        // Lower waste = better fit to target
        double cpu_waste_ratio = std::abs(md.cpu - target_cpu_per_node) / target_cpu_per_node;
        double mem_waste_ratio = std::abs(md.memory - target_mem_per_node) / target_mem_per_node;
        double waste_score = cpu_waste_ratio + mem_waste_ratio;

        // Prefer MDs that are slightly larger than target (can fit more pods)
        double size_bonus = 0.0;
        if (md.cpu >= target_cpu_per_node && md.memory >= target_mem_per_node) {
            size_bonus = -0.5;  // Bonus for being able to accommodate target
        }

        // Incorporate plugin score (higher = better)
        double plugin_weight = plugin_scores[j] * 2.0;

        // Combined score (lower is better for waste, but we want high plugin scores)
        double combined_score = waste_score + size_bonus - plugin_weight;

        md_scores.push_back({j, combined_score});
    }

    // Sort by combined score (lower is better)
    std::sort(md_scores.begin(), md_scores.end(),
              [](const MDFitScore& a, const MDFitScore& b) {
                  return a.score < b.score;
              });

    vector<int> md_order;
    md_order.reserve(md_scores.size());
    for (const auto& ms : md_scores) {
        md_order.push_back(ms.md_idx);
    }

    return md_order;
}


// ============================================================================
// BEAM SEARCH GREEDY - Enhanced backtracking bin-packing
// ============================================================================

// SearchState: Represents a partial solution during beam search
struct SearchState {
    vector<BeamAssignment> assignments;
    vector<vector<SlotState>> slot_usage;
    double waste_score;
    int nodes_used;
    int pods_placed;

    // Default constructor (needed for vector::resize)
    SearchState() : waste_score(0.0), nodes_used(0), pods_placed(0) {}

    SearchState(int num_pods, int num_mds, const MachineDeployment* mds)
        : assignments(num_pods, {-1, -1}),
          slot_usage(num_mds),
          waste_score(0.0),
          nodes_used(0),
          pods_placed(0) {
        for (int i = 0; i < num_mds; ++i) {
            slot_usage[i].resize(mds[i].max_scale_out);
            for (int j = 0; j < mds[i].max_scale_out; ++j) {
                slot_usage[i][j] = {false, mds[i].cpu, mds[i].memory};
            }
        }
    }

    // Clone state for branching
    SearchState Clone() const {
        SearchState copy = *this;
        return copy;
    }

    // Calculate current waste in the state
    double CalculateWaste(const MachineDeployment* mds, int num_mds,
                         const double* plugin_scores) const {
        double total_waste = 0.0;
        int total_nodes = 0;

        for (int md_idx = 0; md_idx < num_mds; ++md_idx) {
            for (int slot_idx = 0; slot_idx < mds[md_idx].max_scale_out; ++slot_idx) {
                if (slot_usage[md_idx][slot_idx].used) {
                    total_nodes++;
                    // Waste = unused resources + cost penalty
                    double cpu_waste = slot_usage[md_idx][slot_idx].cpu_remaining;
                    double mem_waste = slot_usage[md_idx][slot_idx].mem_remaining;
                    total_waste += cpu_waste + mem_waste;

                    // Add machine cost (inverse of plugin score represents cost)
                    total_waste += (10000.0 - plugin_scores[md_idx]);
                }
            }
        }

        // CRITICAL: Heavy penalty for node count (favor fewer nodes over less waste)
        // This aligns with Cluster Autoscaler's bin-packing philosophy
        total_waste += total_nodes * 100000.0;

        return total_waste;
    }
};

// Count nodes already in use for a specific MD type
int CountNodesForMD(const SearchState& state, int md_idx,
                   const MachineDeployment* mds) {
    int count = 0;
    for (int slot_idx = 0; slot_idx < mds[md_idx].max_scale_out; ++slot_idx) {
        if (state.slot_usage[md_idx][slot_idx].used) {
            count++;
        }
    }
    return count;
}

// Check if pod can fit in any existing node of this MD type
bool CanFitInExistingNode(int pod_idx, int md_idx, const SearchState& state,
                          const Pod* pods, const MachineDeployment* mds) {
    for (int slot_idx = 0; slot_idx < mds[md_idx].max_scale_out; ++slot_idx) {
        if (state.slot_usage[md_idx][slot_idx].used) {
            if (pods[pod_idx].cpu <= state.slot_usage[md_idx][slot_idx].cpu_remaining &&
                pods[pod_idx].memory <= state.slot_usage[md_idx][slot_idx].mem_remaining) {
                return true;
            }
        }
    }
    return false;
}

// Select top N MD candidates for placing a pod (dynamic narrowing)
vector<int> SelectTopMDsForPod(int pod_idx, const SearchState& state,
                               const MachineDeployment* mds, int num_mds,
                               const Pod* pods, const double* plugin_scores,
                               const int* allowed_matrix, int limit) {
    vector<pair<int, double>> md_scores;

    for (int md_idx = 0; md_idx < num_mds; ++md_idx) {
        // Check basic feasibility
        if (!allowed_matrix[pod_idx * num_mds + md_idx]) continue;
        if (pods[pod_idx].cpu > mds[md_idx].cpu ||
            pods[pod_idx].memory > mds[md_idx].memory) continue;

        // CRITICAL: Check if MD has ANY capacity (existing node with space OR ability to create new node)
        bool can_fit_existing = CanFitInExistingNode(pod_idx, md_idx, state, pods, mds);
        bool can_create_new = false;
        for (int slot_idx = 0; slot_idx < mds[md_idx].max_scale_out; ++slot_idx) {
            if (!state.slot_usage[md_idx][slot_idx].used) {
                can_create_new = true;
                break;
            }
        }

        // Skip this MD if it has no capacity at all
        if (!can_fit_existing && !can_create_new) continue;

        double score = 0.0;

        // 1. STRONGEST PREFERENCE: Can fit in existing node of this MD type
        if (can_fit_existing) {
            score += 50000.0;  // Massive bonus for consolidation

            // Additional bonus based on how full it would make the node
            for (int slot_idx = 0; slot_idx < mds[md_idx].max_scale_out; ++slot_idx) {
                if (state.slot_usage[md_idx][slot_idx].used) {
                    if (pods[pod_idx].cpu <= state.slot_usage[md_idx][slot_idx].cpu_remaining &&
                        pods[pod_idx].memory <= state.slot_usage[md_idx][slot_idx].mem_remaining) {
                        double cpu_util = pods[pod_idx].cpu / mds[md_idx].cpu;
                        double mem_util = pods[pod_idx].memory / mds[md_idx].memory;
                        score += (cpu_util + mem_util) * 1000.0;
                        break;
                    }
                }
            }
        }

        // 2. MODERATE PREFERENCE: Already have nodes of this MD type (create new node)
        int existing_nodes = CountNodesForMD(state, md_idx, mds);
        if (existing_nodes > 0) {
            score += 10000.0 / (1.0 + existing_nodes);  // Diminishing returns
        }

        // 3. Resource fit quality (minimize waste if creating new node)
        double cpu_fit = pods[pod_idx].cpu / mds[md_idx].cpu;
        double mem_fit = pods[pod_idx].memory / mds[md_idx].memory;
        double fit_score = std::min(cpu_fit, mem_fit);  // 0-1, higher = better fit
        score += fit_score * 500.0;

        // 4. Plugin score bonus (cost/preference)
        score += plugin_scores[md_idx] * 5.0;

        md_scores.push_back({md_idx, score});
    }

    // Sort by score descending
    sort(md_scores.begin(), md_scores.end(),
         [](const pair<int,double>& a, const pair<int,double>& b) {
             return a.second > b.second;
         });

    // Return top N MD indices
    vector<int> result;
    for (int i = 0; i < std::min(limit, (int)md_scores.size()); ++i) {
        result.push_back(md_scores[i].first);
    }
    return result;
}

// Try to place pod on existing node
bool TryPlaceOnExistingNode(int pod_idx, int md_idx, SearchState& state,
                            const MachineDeployment* mds, const Pod* pods) {
    int best_slot = -1;
    double best_util = -1.0;

    // Find best existing slot (prefer fuller nodes)
    for (int slot_idx = 0; slot_idx < mds[md_idx].max_scale_out; ++slot_idx) {
        if (state.slot_usage[md_idx][slot_idx].used) {
            if (pods[pod_idx].cpu <= state.slot_usage[md_idx][slot_idx].cpu_remaining &&
                pods[pod_idx].memory <= state.slot_usage[md_idx][slot_idx].mem_remaining) {

                // Calculate utilization after placing
                double cpu_util = 1.0 - ((state.slot_usage[md_idx][slot_idx].cpu_remaining - pods[pod_idx].cpu) / mds[md_idx].cpu);
                double mem_util = 1.0 - ((state.slot_usage[md_idx][slot_idx].mem_remaining - pods[pod_idx].memory) / mds[md_idx].memory);
                double avg_util = (cpu_util + mem_util) / 2.0;

                if (avg_util > best_util) {
                    best_util = avg_util;
                    best_slot = slot_idx;
                }
            }
        }
    }

    if (best_slot != -1) {
        // Place pod
        state.assignments[pod_idx] = {md_idx, best_slot};
        state.slot_usage[md_idx][best_slot].cpu_remaining -= pods[pod_idx].cpu;
        state.slot_usage[md_idx][best_slot].mem_remaining -= pods[pod_idx].memory;
        state.pods_placed++;
        return true;
    }

    return false;
}

// Try to place pod on new node
bool TryPlaceOnNewNode(int pod_idx, int md_idx, SearchState& state,
                      const MachineDeployment* mds, const Pod* pods) {
    // Find first empty slot
    for (int slot_idx = 0; slot_idx < mds[md_idx].max_scale_out; ++slot_idx) {
        if (!state.slot_usage[md_idx][slot_idx].used) {
            // Create new node
            state.assignments[pod_idx] = {md_idx, slot_idx};
            state.slot_usage[md_idx][slot_idx].used = true;
            state.slot_usage[md_idx][slot_idx].cpu_remaining -= pods[pod_idx].cpu;
            state.slot_usage[md_idx][slot_idx].mem_remaining -= pods[pod_idx].memory;
            state.nodes_used++;
            state.pods_placed++;
            return true;
        }
    }

    return false;  // MD is full
}

// Main beam search algorithm
BeamResult RunBeamSearch(
    const MachineDeployment* mds,
    int num_mds,
    const Pod* pods,
    int num_pods,
    const double* plugin_scores,
    const int* allowed_matrix,
    int beam_width = 3,
    int md_candidates = 5
) {
    // Build affinity groups (pods that MUST be placed together)
    vector<vector<int>> affinity_groups = BuildAffinityGroups(pods, num_pods);

    // Calculate MD order for affinity group placement
    vector<int> md_order = CalculateOptimalMDOrder(mds, num_mds, pods, num_pods,
                                                     plugin_scores, allowed_matrix);

    // Initialize beam with empty state
    vector<SearchState> beam;
    beam.push_back(SearchState(num_pods, num_mds, mds));

    // PHASE 1: Place affinity groups using best-fit strategy
    // (Beam search doesn't help here since affinity groups must be placed together)
    for (const auto& group : affinity_groups) {
        bool placed = TryPlaceAffinityGroup(group, md_order, mds, pods, allowed_matrix,
                                           num_mds, beam[0].slot_usage, beam[0].assignments);
        if (placed) {
            for (int pod_idx : group) {
                if (beam[0].assignments[pod_idx].md != -1) {
                    beam[0].pods_placed++;
                    // Update used flag
                    int md = beam[0].assignments[pod_idx].md;
                    int slot = beam[0].assignments[pod_idx].slot;
                    beam[0].slot_usage[md][slot].used = true;
                }
            }
        }
    }

    // PHASE 2: Beam search for remaining pods
    // Sort pods by difficulty (largest, most constrained first)
    vector<int> pod_order = SortPodsByDifficulty(pods, num_pods, allowed_matrix, num_mds);

    // Optional debug file (disabled by default for performance)
    // std::ofstream debug("/tmp/beam_debug_cpp.txt");
    // debug << "Starting beam search with " << num_pods << " pods" << std::endl;
    // debug << "Beam width: " << beam_width << ", MD candidates: " << md_candidates << std::endl;
    // debug << "Affinity groups placed: " << affinity_groups.size() << std::endl;
    // debug << "Pods already placed: " << beam[0].pods_placed << std::endl;

    // Process each pod in order
    for (int i = 0; i < (int)pod_order.size(); ++i) {
        int pod_idx = pod_order[i];

        // Skip pods already placed by affinity group logic
        if (beam[0].assignments[pod_idx].md != -1) {
            continue;
        }

        vector<SearchState> next_beam;

        // debug << "\n=== Pod " << i << "/" << num_pods << " (idx=" << pod_idx << ") ===" << std::endl;
        // debug << "Current beam size: " << beam.size() << std::endl;

        // For each state in current beam
        for (size_t beam_idx = 0; beam_idx < beam.size(); ++beam_idx) {
            const SearchState& state = beam[beam_idx];

            // CLUSTER AUTOSCALER STRATEGY: Try ALL existing nodes first (across all MDs)
            vector<pair<int, int>> existing_options; // (md_idx, slot_idx) pairs
            vector<double> existing_scores;

            for (int md_idx = 0; md_idx < num_mds; ++md_idx) {
                // Basic feasibility
                if (!allowed_matrix[pod_idx * num_mds + md_idx]) continue;
                if (pods[pod_idx].cpu > mds[md_idx].cpu ||
                    pods[pod_idx].memory > mds[md_idx].memory) continue;

                // Check all existing slots in this MD
                for (int slot_idx = 0; slot_idx < mds[md_idx].max_scale_out; ++slot_idx) {
                    if (!state.slot_usage[md_idx][slot_idx].used) continue;

                    // Can pod fit?
                    if (pods[pod_idx].cpu > state.slot_usage[md_idx][slot_idx].cpu_remaining ||
                        pods[pod_idx].memory > state.slot_usage[md_idx][slot_idx].mem_remaining) continue;

                    // Score: prefer fuller nodes
                    double cpu_util = 1.0 - ((state.slot_usage[md_idx][slot_idx].cpu_remaining - pods[pod_idx].cpu) / mds[md_idx].cpu);
                    double mem_util = 1.0 - ((state.slot_usage[md_idx][slot_idx].mem_remaining - pods[pod_idx].memory) / mds[md_idx].memory);
                    double score = (cpu_util + mem_util) / 2.0 * 10000.0 + plugin_scores[md_idx] * 10.0;

                    existing_options.push_back({md_idx, slot_idx});
                    existing_scores.push_back(score);
                }
            }

            // debug << "  Beam state " << beam_idx << ": " << existing_options.size() << " existing slots, ";

            // If we found existing slots, create states for top ones
            if (!existing_options.empty()) {
                // Sort by score (best first)
                vector<int> indices(existing_options.size());
                for (size_t k = 0; k < indices.size(); ++k) indices[k] = k;
                sort(indices.begin(), indices.end(),
                     [&](int a, int b) { return existing_scores[a] > existing_scores[b]; });

                // Create states for top md_candidates existing placements
                int limit = std::min(md_candidates, (int)existing_options.size());
                for (int k = 0; k < limit; ++k) {
                    int idx = indices[k];
                    int md_idx = existing_options[idx].first;
                    int slot_idx = existing_options[idx].second;

                    SearchState new_state = state.Clone();
                    new_state.assignments[pod_idx] = {md_idx, slot_idx};
                    new_state.slot_usage[md_idx][slot_idx].cpu_remaining -= pods[pod_idx].cpu;
                    new_state.slot_usage[md_idx][slot_idx].mem_remaining -= pods[pod_idx].memory;
                    new_state.pods_placed++;
                    new_state.waste_score = new_state.CalculateWaste(mds, num_mds, plugin_scores);
                    next_beam.push_back(new_state);
                }
                // debug << "used top " << limit << " existing" << std::endl;
            } else {
                // No existing slots - need to create new nodes
                // Get top MD choices for creating new nodes
                vector<int> md_choices = SelectTopMDsForPod(pod_idx, state, mds, num_mds,
                                                            pods, plugin_scores,
                                                            allowed_matrix, md_candidates);

                // debug << md_choices.size() << " new node options" << std::endl;

                for (int md_idx : md_choices) {
                    SearchState new_state = state.Clone();
                    bool placed = TryPlaceOnNewNode(pod_idx, md_idx, new_state, mds, pods);

                    if (placed) {
                        new_state.waste_score = new_state.CalculateWaste(mds, num_mds, plugin_scores);
                        next_beam.push_back(new_state);
                    }
                }
            }
        }

        // debug << "Next beam size: " << next_beam.size() << std::endl;

        // If no placements found, return failure
        if (next_beam.empty()) {
            // debug << "FAILURE: Beam became empty at pod " << i << "/" << num_pods << std::endl;
            // debug.close();
            BeamResult failure;
            failure.assignments.resize(num_pods, {-1, -1});
            failure.slots_used.resize(num_mds);
            for (int j = 0; j < num_mds; ++j) {
                failure.slots_used[j].resize(mds[j].max_scale_out, false);
            }
            failure.all_placed = false;
            failure.unplaced_count = num_pods - i;
            return failure;
        }

        // Keep only top beam_width states (lowest waste)
        sort(next_beam.begin(), next_beam.end(),
             [](const SearchState& a, const SearchState& b) {
                 return a.waste_score < b.waste_score;
             });

        if ((int)next_beam.size() > beam_width) {
            next_beam.resize(beam_width);
        }

        beam = next_beam;
    }

    // debug << "\n=== SUCCESS: All pods placed ===" << std::endl;
    // debug.close();

    // Return best state from final beam
    if (beam.empty()) {
        BeamResult failure;
        failure.assignments.resize(num_pods, {-1, -1});
        failure.slots_used.resize(num_mds);
        for (int j = 0; j < num_mds; ++j) {
            failure.slots_used[j].resize(mds[j].max_scale_out, false);
        }
        failure.all_placed = false;
        failure.unplaced_count = num_pods;
        return failure;
    }

    // Convert best SearchState to BeamResult
    const SearchState& best = beam[0];
    BeamResult result;
    result.assignments = best.assignments;
    result.slots_used.resize(num_mds);

    // Convert slot_usage to slots_used (used flags)
    for (int md_idx = 0; md_idx < num_mds; ++md_idx) {
        result.slots_used[md_idx].resize(mds[md_idx].max_scale_out);
        for (int slot_idx = 0; slot_idx < mds[md_idx].max_scale_out; ++slot_idx) {
            result.slots_used[md_idx][slot_idx] = best.slot_usage[md_idx][slot_idx].used;
        }
    }

    result.all_placed = (best.pods_placed == num_pods);
    result.unplaced_count = num_pods - best.pods_placed;

    return result;
}

// Helper: Calculate solution cost (must match BuildCombinedObjective)
double CalculateSolutionCost(
    const vector<BeamAssignment>& assignments,
    const vector<vector<bool>>& slots_used,
    const MachineDeployment* mds,
    int num_mds,
    const Pod* pods,
    int num_pods,
    const double* plugin_scores
) {
    double total_waste = 0.0;
    double plugin_cost = 0.0;
    double soft_affinity_penalty = 0.0;

    // Calculate waste per MD
    for (int j = 0; j < num_mds; ++j) {
        int nodes_used = 0;
        for (bool used : slots_used[j]) {
            if (used) nodes_used++;
        }

        if (nodes_used == 0) continue;

        // Calculate assigned resources
        double cpu_assigned = 0.0;
        double mem_assigned = 0.0;
        for (int i = 0; i < num_pods; ++i) {
            if (assignments[i].md == j) {
                cpu_assigned += pods[i].cpu * 100.0;
                mem_assigned += pods[i].memory * 100.0;
            }
        }

        // Calculate provisioned resources
        double cpu_provisioned = nodes_used * mds[j].cpu * 100.0;
        double mem_provisioned = nodes_used * mds[j].memory * 100.0;

        // Waste
        double cpu_waste = cpu_provisioned - cpu_assigned;
        double mem_waste = mem_provisioned - mem_assigned;
        total_waste += cpu_waste + mem_waste;
    }

    // Calculate plugin score cost
    for (int j = 0; j < num_mds; ++j) {
        int nodes_used = 0;
        for (bool used : slots_used[j]) {
            if (used) nodes_used++;
        }
        double weight = (1.0 - plugin_scores[j]) * 1000.0 + 1.0;
        plugin_cost += nodes_used * weight;
    }

    // Calculate soft affinity penalties
    std::unordered_set<std::pair<int, int>> soft_seen;
    for (int i = 0; i < num_pods; ++i) {
        const Pod& pod = pods[i];
        if (pod.soft_affinity_count == 0) continue;

        for (int p = 0; p < pod.soft_affinity_count; ++p) {
            int peer = pod.soft_peers[p];
            if (peer < 0 || peer >= num_pods) continue;

            auto pair = std::minmax(i, peer);
            if (!soft_seen.insert(pair).second) continue;

            double affinity = pod.soft_affinities[p];
            if (affinity == 0.0) continue;

            int rule = (affinity > 0) ? 1 : -1;
            int scaled_weight = static_cast<int>(std::abs(affinity) * 1000.0);

            bool same_slot = (assignments[i].md == assignments[peer].md &&
                            assignments[i].slot == assignments[peer].slot);

            bool penalty_applies = false;
            if (rule == 1) {
                // Colocate: penalty if not same slot
                penalty_applies = !same_slot;
            } else {
                // Spread: penalty if same slot
                penalty_applies = same_slot;
            }

            if (penalty_applies) {
                soft_affinity_penalty += scaled_weight;
            }
        }
    }

    // Combined objective (must match C++)
    int objective_scale = 1000;
    double total_cost = (total_waste * objective_scale) +
                       (plugin_cost * objective_scale) +
                       (soft_affinity_penalty * 250);

    return total_cost;
}

// Helper: Convert beam result to hint matrix format
vector<int> ConvertBeamToHintMatrix(
    const BeamResult& beam_result,
    const MachineDeployment* mds,
    int num_mds,
    int num_pods
) {
    // Calculate total hint size
    int total_size = 0;
    for (int j = 0; j < num_mds; ++j) {
        total_size += num_pods * mds[j].max_scale_out;
    }

    vector<int> hint(total_size, 0);

    int offset = 0;
    for (int i = 0; i < num_pods; ++i) {
        for (int j = 0; j < num_mds; ++j) {
            for (int k = 0; k < mds[j].max_scale_out; ++k) {
                if (beam_result.assignments[i].md == j &&
                    beam_result.assignments[i].slot == k) {
                    hint[offset] = 1;
                }
                offset++;
            }
        }
    }

    return hint;
}

// Helper: Copy beam result to output arrays
void CopyBeamResultToOutput(
    const BeamResult& beam_result,
    const MachineDeployment* mds,
    int num_mds,
    int num_pods,
    int* out_slot_assignments,
    uint8_t* out_slots_used
) {
    // Copy assignments
    for (int i = 0; i < num_pods; ++i) {
        out_slot_assignments[i * 2] = beam_result.assignments[i].md;
        out_slot_assignments[i * 2 + 1] = beam_result.assignments[i].slot;
    }

    // Copy slots_used
    int offset = 0;
    for (int j = 0; j < num_mds; ++j) {
        for (int k = 0; k < mds[j].max_scale_out; ++k) {
            out_slots_used[offset++] = beam_result.slots_used[j][k] ? 1 : 0;
        }
    }
}

// Helper: Calculate current state cost
double CalculateCurrentStateCost(
    const MachineDeployment* mds,
    int num_mds,
    const Pod* pods,
    int num_pods,
    const double* plugin_scores
) {
    double current_state_cost = 0.0;
    std::vector<bool> md_currently_used(num_mds, false);

    // Mark which MDs are currently being used
    for (int i = 0; i < num_pods; ++i) {
        int current_md = pods[i].current_md_assignment;
        if (current_md >= 0 && current_md < num_mds) {
            md_currently_used[current_md] = true;
        }
    }

    // Calculate current cost based on waste from current pod placements
    for (int j = 0; j < num_mds; ++j) {
        if (md_currently_used[j]) {
            // Calculate current resource usage for this MD
            double cpu_used = 0.0, mem_used = 0.0;
            for (int i = 0; i < num_pods; ++i) {
                if (pods[i].current_md_assignment == j) {
                    cpu_used += pods[i].cpu * 100.0;
                    mem_used += pods[i].memory * 100.0;
                }
            }

            // Calculate provisioned resources (assuming 1 node currently)
            double cpu_provisioned = mds[j].cpu * 100.0;
            double mem_provisioned = mds[j].memory * 100.0;

            // Calculate waste
            double cpu_waste = cpu_provisioned - cpu_used;
            double mem_waste = mem_provisioned - mem_used;

            // Add plugin score component
            int plugin_weight = static_cast<int>((1.0 - plugin_scores[j]) * 1000.0) + 1;

            current_state_cost += (cpu_waste + mem_waste) * 1000.0 + plugin_weight;
        }
    }

    return current_state_cost;
}

// Structure for CP-SAT solver result
struct CPSATResult {
    bool success;
    double objective;
    int status_code;
    double solve_time_secs;
    vector<BeamAssignment> assignments;
    vector<vector<bool>> slots_used;
};

// Helper: Run CP-SAT solver for one attempt
CPSATResult RunCPSATSolverAttempt(
    const MachineDeployment* mds,
    int num_mds,
    const Pod* pods,
    int num_pods,
    const double* plugin_scores,
    const int* allowed_matrix,
    const int* hint,
    int timeout_secs
) {
    CPSATResult result;
    result.success = false;
    result.objective = 0.0;
    result.status_code = 0;
    result.solve_time_secs = 0.0;

    sat::CpModelBuilder model;
    Optimiser optimiser = Optimiser(num_mds, mds, num_pods, pods, allowed_matrix, &model);

    optimiser.InitSlotsUsed();
    optimiser.InitPodSlotAssignment();

    auto slots_per_md = optimiser.GetSlotsPerMd();
    auto pod_slot_assignment = optimiser.GetPodSlotAssignment();
    auto slot_used = optimiser.GetSlotUsed();

    // Add hard affinity constraints
    std::unordered_set<std::pair<int, int>> affinity_seen;
    for (int i = 0; i < num_pods; ++i) {
        const Pod& pod = pods[i];
        for (int p = 0; p < pod.affinity_count; ++p) {
            int other = pod.affinity_peers[p];
            int rule = pod.affinity_rules[p];
            if (other < 0 || other >= num_pods || i > other || rule == 0) continue;
            auto pair = std::minmax(i, other);
            if (!affinity_seen.insert(pair).second) continue;
            for (int j = 0; j < num_mds; ++j) {
                if (!allowed_matrix[i * num_mds + j] || !allowed_matrix[other * num_mds + j]) continue;
                int slots = slots_per_md[j];
                for (int k = 0; k < slots; ++k) {
                    if (rule == 1) {
                        model.AddEquality(pod_slot_assignment[i][j][k], pod_slot_assignment[other][j][k]);
                    } else {
                        model.AddBoolOr({pod_slot_assignment[i][j][k].Not(), pod_slot_assignment[other][j][k].Not()});
                    }
                }
            }
        }
    }

    // Add hints if provided
    if (hint != nullptr) {
        int offset = 0;
        for (int i = 0; i < num_pods; ++i) {
            for (int j = 0; j < num_mds; ++j) {
                for (int k = 0; k < slots_per_md[j]; ++k) {
                    if (hint[offset] == 1 && allowed_matrix[i * num_mds + j]) {
                        model.AddHint(pod_slot_assignment[i][j][k], 1);
                    }
                    offset++;
                }
            }
        }
    }

    // Each pod must be assigned to exactly one slot
    for (int i = 0; i < num_pods; ++i) {
        std::vector<sat::BoolVar> choices;
        for (int j = 0; j < num_mds; ++j) {
            if (!allowed_matrix[i * num_mds + j]) continue;
            for (int k = 0; k < slots_per_md[j]; ++k) {
                choices.push_back(pod_slot_assignment[i][j][k]);
            }
        }
        model.AddEquality(sat::LinearExpr::Sum(choices), 1);
    }

    // Resource constraints and slot tracking
    for (int j = 0; j < num_mds; ++j) {
        int slots = slots_per_md[j];
        sat::LinearExpr used_sum;
        for (int k = 0; k < slots; ++k) {
            sat::LinearExpr cpu_sum, mem_sum, assigned;
            for (int i = 0; i < num_pods; ++i) {
                if (!allowed_matrix[i * num_mds + j]) continue;
                cpu_sum += pod_slot_assignment[i][j][k] * static_cast<int>(pods[i].cpu * 100);
                mem_sum += pod_slot_assignment[i][j][k] * static_cast<int>(pods[i].memory * 100);
                assigned += pod_slot_assignment[i][j][k];
            }
            model.AddLessOrEqual(cpu_sum, static_cast<int>(mds[j].cpu * 100));
            model.AddLessOrEqual(mem_sum, static_cast<int>(mds[j].memory * 100));
            model.AddGreaterThan(assigned, 0).OnlyEnforceIf(slot_used[j][k]);
            model.AddEquality(assigned, 0).OnlyEnforceIf(slot_used[j][k].Not());
            used_sum += slot_used[j][k];

            optimiser.TrackResourceAssignments(j, k, cpu_sum, mem_sum);
        }
        model.AddLessOrEqual(used_sum, mds[j].max_scale_out);
        optimiser.SetNodesUsed(j, used_sum);

        for (int k = 0; k + 1 < slots; ++k) {
            model.AddImplication(slot_used[j][k + 1], slot_used[j][k]);
        }
    }

    // Soft affinity penalties
    std::unordered_set<std::pair<int, int>> soft_seen;
    sat::LinearExpr total_penalty;
    for (int i = 0; i < num_pods; ++i) {
        const Pod& pod = pods[i];
        if (pod.soft_affinity_count == 0) continue;
        for (int p = 0; p < pod.soft_affinity_count; ++p) {
            int peer = pod.soft_peers[p];
            if (peer < 0 || peer >= num_pods) continue;
            auto pair = std::minmax(i, peer);
            if (!soft_seen.insert(pair).second) continue;
            double affinity = pod.soft_affinities[p];
            if (affinity == 0.0) continue;
            int rule = (affinity > 0) ? 1 : -1;
            int scaled_weight = static_cast<int>(std::abs(affinity) * 1000.0);
            for (int j = 0; j < num_mds; ++j) {
                if (!allowed_matrix[i * num_mds + j] || !allowed_matrix[peer * num_mds + j]) continue;
                int slots = slots_per_md[j];
                for (int k = 0; k < slots; ++k) {
                    sat::BoolVar penalty = model.NewBoolVar();
                    if (rule == 1) {
                        model.AddBoolOr({pod_slot_assignment[i][j][k].Not(), pod_slot_assignment[peer][j][k].Not()}).OnlyEnforceIf(penalty);
                        model.AddBoolAnd({pod_slot_assignment[i][j][k], pod_slot_assignment[peer][j][k]}).OnlyEnforceIf(penalty.Not());
                    } else {
                        model.AddBoolAnd({pod_slot_assignment[i][j][k], pod_slot_assignment[peer][j][k]}).OnlyEnforceIf(penalty);
                        model.AddBoolOr({pod_slot_assignment[i][j][k].Not(), pod_slot_assignment[peer][j][k].Not()}).OnlyEnforceIf(penalty.Not());
                    }
                    total_penalty += penalty * scaled_weight;
                }
            }
        }
    }

    // Build objective
    sat::LinearExpr objective = optimiser.BuildCombinedObjective(plugin_scores, total_penalty, 0.7);
    model.Minimize(objective);

    // Solve
    sat::Model cp_model;
    sat::SatParameters parameters;
    parameters.set_max_time_in_seconds(static_cast<double>(timeout_secs));
    parameters.set_random_seed(42);
    parameters.set_randomize_search(false);
    cp_model.Add(sat::NewSatParameters(parameters));

    const sat::CpSolverResponse response = sat::SolveCpModel(model.Build(), &cp_model);

    result.status_code = static_cast<int>(response.status());
    result.success = response.status() == sat::CpSolverStatus::OPTIMAL ||
                     response.status() == sat::CpSolverStatus::FEASIBLE;
    result.solve_time_secs = response.wall_time();
    result.objective = response.objective_value();

    if (result.success) {
        // Extract assignments
        result.assignments.resize(num_pods, {-1, -1});
        for (int i = 0; i < num_pods; ++i) {
            for (int j = 0; j < num_mds; ++j) {
                if (!allowed_matrix[i * num_mds + j]) continue;
                for (int k = 0; k < slots_per_md[j]; ++k) {
                    if (sat::SolutionBooleanValue(response, pod_slot_assignment[i][j][k])) {
                        result.assignments[i] = {j, k};
                        goto next_pod;
                    }
                }
            }
        next_pod:;
        }

        // Extract slots_used
        result.slots_used.resize(num_mds);
        for (int j = 0; j < num_mds; ++j) {
            result.slots_used[j].resize(slots_per_md[j]);
            for (int k = 0; k < slots_per_md[j]; ++k) {
                result.slots_used[j][k] = sat::SolutionBooleanValue(response, slot_used[j][k]);
            }
        }
    }

    return result;
}

// ============================================================================
// END BEAM SEARCH HEURISTIC IMPLEMENTATION
// ============================================================================

extern "C" {

__attribute__((visibility("default")))
SolverResult OptimisePlacement(
    const MachineDeployment* mds, int num_mds,
    const Pod* pods, int num_pods,
    const double* plugin_scores,
    const int* allowed_matrix,
    const int* initial_assignment,
    int* out_slot_assignments,
    uint8_t* out_slots_used,
    const int* max_runtime_secs,
    const double* improvement_threshold,
    const bool* score_only,
    const int* max_attempts,
    const bool* use_greedy_hint,
    const int* greedy_hint_attempt,
    const bool* fallback_to_greedy,
    const bool* greedy_only
) {
    // Initialize result
    SolverResult result = {false, 0.0, 0, 0.0, 0.0, false, false, false, 0, 0, 0};

    // Parse configuration with defaults
    int num_attempts = (max_attempts != nullptr) ? *max_attempts : 1;
    int timeout = (max_runtime_secs != nullptr) ? *max_runtime_secs : 15;
    bool do_greedy_hint = (use_greedy_hint != nullptr) ? *use_greedy_hint : false;
    int greedy_hint_at = (greedy_hint_attempt != nullptr) ? *greedy_hint_attempt : 2;
    bool do_fallback = (fallback_to_greedy != nullptr) ? *fallback_to_greedy : false;
    bool only_greedy = (greedy_only != nullptr) ? *greedy_only : false;
    double threshold = (improvement_threshold != nullptr) ? *improvement_threshold : 0.0;

    // Calculate current state cost
    double current_state_cost = CalculateCurrentStateCost(mds, num_mds, pods, num_pods, plugin_scores);
    result.current_state_cost = current_state_cost;

    // Score-only mode: return current state cost without optimization
    if (score_only != nullptr && *score_only) {
        result.success = true;
        result.objective = current_state_cost;
        result.status_code = static_cast<int>(FEASIBLE);
        result.solve_time_secs = 0.0;
        result.already_optimal = false;
        result.used_beam_search = false;
        result.beam_search_fallback = false;
        result.cpsat_attempts = 0;
        result.best_attempt = 0;
        result.unplaced_pods = 0;

        // Set all output arrays to empty
        for (int i = 0; i < num_pods; ++i) {
            out_slot_assignments[i * 2] = -1;
            out_slot_assignments[i * 2 + 1] = -1;
        }

        int offset = 0;
        for (int j = 0; j < num_mds; ++j) {
            for (int k = 0; k < mds[j].max_scale_out; ++k) {
                out_slots_used[offset++] = 0;
            }
        }

        return result;
    }

    // Beam-search-only mode
    if (only_greedy) {
        BeamResult beam_result = RunBeamSearch(mds, num_mds, pods, num_pods,
                                               plugin_scores, allowed_matrix,
                                               3, 3);  // beam_width=3, md_candidates=3

        if (!beam_result.all_placed) {
            result.success = false;
            result.unplaced_pods = beam_result.unplaced_count;
            result.used_beam_search = true;
            result.beam_search_fallback = true;
            result.cpsat_attempts = 0;
            return result;
        }

        double beam_score = CalculateSolutionCost(beam_result.assignments,
                                                  beam_result.slots_used,
                                                  mds, num_mds, pods, num_pods,
                                                  plugin_scores);

        // If current_state_cost is 0 or very small (pods unplaced), any valid placement is acceptable
        bool accept_solution = false;
        if (current_state_cost < 1.0) {
            // No current placement - any valid beam search solution is acceptable
            accept_solution = true;
        } else {
            double improvement_pct = (current_state_cost - beam_score) / current_state_cost * 100.0;
            accept_solution = (beam_score < current_state_cost && improvement_pct >= threshold);
        }

        if (accept_solution) {
            CopyBeamResultToOutput(beam_result, mds, num_mds, num_pods,
                                   out_slot_assignments, out_slots_used);
            result.success = true;
            result.objective = beam_score;
            result.status_code = static_cast<int>(FEASIBLE);
            result.already_optimal = false;
            result.used_beam_search = true;
            result.beam_search_fallback = true;
            result.cpsat_attempts = 0;
            result.best_attempt = 0;
            result.unplaced_pods = 0;
            return result;
        } else if (beam_score < current_state_cost) {
            // Improves but not enough
            CopyBeamResultToOutput(beam_result, mds, num_mds, num_pods,
                                   out_slot_assignments, out_slots_used);
            result.success = true;
            result.objective = beam_score;
            result.status_code = static_cast<int>(FEASIBLE);
            result.already_optimal = true;
            result.used_beam_search = true;
            result.beam_search_fallback = true;
            result.cpsat_attempts = 0;
            result.best_attempt = 0;
            result.unplaced_pods = 0;
            return result;
        } else {
            result.success = false;
            result.objective = beam_score;
            result.used_beam_search = true;
            result.beam_search_fallback = true;
            result.cpsat_attempts = 0;
            return result;
        }
    }

    // Main optimization loop with CP-SAT
    vector<int> beam_hint_matrix;
    bool beam_hint_generated = false;
    double best_score = std::numeric_limits<double>::infinity();
    CPSATResult best_cpsat_result;
    bool have_best = false;

    for (int attempt = 1; attempt <= num_attempts; ++attempt) {
        const int* hint_to_use = nullptr;

        // Determine hint strategy
        if (initial_assignment != nullptr) {
            // User-provided hint always takes precedence
            hint_to_use = initial_assignment;
        } else if (do_greedy_hint && attempt == greedy_hint_at && !beam_hint_generated) {
            // Generate beam search hint on specified attempt
            BeamResult beam_result = RunBeamSearch(mds, num_mds, pods, num_pods,
                                                   plugin_scores, allowed_matrix,
                                                   3, 3);  // beam_width=3, md_candidates=3
            if (beam_result.all_placed) {
                beam_hint_matrix = ConvertBeamToHintMatrix(beam_result, mds, num_mds, num_pods);
                hint_to_use = beam_hint_matrix.data();
                beam_hint_generated = true;
                result.used_beam_search = true;
                std::cerr << "Using beam search hint for attempt " << attempt << std::endl;
            }
        }

        // Run CP-SAT solver
        CPSATResult cpsat_result = RunCPSATSolverAttempt(mds, num_mds, pods, num_pods,
                                                          plugin_scores, allowed_matrix,
                                                          hint_to_use, timeout);

        result.cpsat_attempts = attempt;

        if (cpsat_result.success) {
            // Calculate actual score using our cost function
            double score = CalculateSolutionCost(cpsat_result.assignments,
                                                 cpsat_result.slots_used,
                                                 mds, num_mds, pods, num_pods,
                                                 plugin_scores);

            // Track best solution
            if (score < best_score) {
                best_score = score;
                best_cpsat_result = cpsat_result;
                have_best = true;
            }

            // Check if solution meets criteria
            bool accept_solution = false;
            if (current_state_cost < 1.0) {
                // No current placement - any valid solution is acceptable
                accept_solution = true;
            } else {
                double improvement_pct = (current_state_cost - score) / current_state_cost * 100.0;
                accept_solution = (score < current_state_cost && improvement_pct >= threshold);
            }

            if (accept_solution) {
                // Solution meets criteria - return immediately
                CopyBeamResultToOutput(BeamResult{
                    cpsat_result.assignments,
                    cpsat_result.slots_used,
                    true,
                    0
                }, mds, num_mds, num_pods, out_slot_assignments, out_slots_used);

                result.success = true;
                result.objective = score;
                result.status_code = cpsat_result.status_code;
                result.solve_time_secs = cpsat_result.solve_time_secs;
                result.already_optimal = false;
                result.best_attempt = attempt;
                result.unplaced_pods = 0;
                return result;
            }
        }
    }

    // All CP-SAT attempts completed - check if we have a best solution
    bool have_acceptable_best = false;
    double best_improvement_pct = 0.0;

    if (have_best) {
        if (current_state_cost < 1.0) {
            // No current placement - any valid solution is acceptable
            have_acceptable_best = true;
        } else {
            best_improvement_pct = (current_state_cost - best_score) / current_state_cost * 100.0;
            have_acceptable_best = (best_score < current_state_cost);
        }
    }

    if (have_acceptable_best) {

        CopyBeamResultToOutput(BeamResult{
            best_cpsat_result.assignments,
            best_cpsat_result.slots_used,
            true,
            0
        }, mds, num_mds, num_pods, out_slot_assignments, out_slots_used);

        result.success = true;
        result.objective = best_score;
        result.status_code = best_cpsat_result.status_code;
        result.solve_time_secs = best_cpsat_result.solve_time_secs;
        result.already_optimal = (best_improvement_pct < threshold);
        result.best_attempt = num_attempts;
        result.unplaced_pods = 0;
        return result;
    }

    // No valid CP-SAT solution found - try beam search fallback
    if (do_fallback) {
        BeamResult beam_result = RunBeamSearch(mds, num_mds, pods, num_pods,
                                               plugin_scores, allowed_matrix,
                                               3, 3);  // beam_width=3, md_candidates=3

        if (beam_result.all_placed) {
            double beam_score = CalculateSolutionCost(beam_result.assignments,
                                                      beam_result.slots_used,
                                                      mds, num_mds, pods, num_pods,
                                                      plugin_scores);

            // If current_state_cost is 0 or very small (pods unplaced), any valid placement is acceptable
            bool accept_solution = false;
            if (current_state_cost < 1.0) {
                // No current placement - any valid beam search solution is acceptable
                accept_solution = true;
            } else {
                double improvement_pct = (current_state_cost - beam_score) / current_state_cost * 100.0;
                accept_solution = (beam_score < current_state_cost && improvement_pct >= threshold);
            }

            if (accept_solution) {
                CopyBeamResultToOutput(beam_result, mds, num_mds, num_pods,
                                       out_slot_assignments, out_slots_used);
                result.success = true;
                result.objective = beam_score;
                result.status_code = static_cast<int>(FEASIBLE);
                result.already_optimal = false;
                result.used_beam_search = true;
                result.beam_search_fallback = true;
                result.best_attempt = 0;
                result.unplaced_pods = 0;
                return result;
            } else if (beam_score < current_state_cost) {
                // Improves but doesn't meet threshold
                CopyBeamResultToOutput(beam_result, mds, num_mds, num_pods,
                                       out_slot_assignments, out_slots_used);
                result.success = true;
                result.objective = beam_score;
                result.status_code = static_cast<int>(FEASIBLE);
                result.already_optimal = true;
                result.used_beam_search = true;
                result.beam_search_fallback = true;
                result.best_attempt = 0;
                result.unplaced_pods = 0;
                return result;
            }
        }
    }

    // No solution found that improves current state
    result.success = false;
    return result;
}

// ============================================================================
// Beam Search API - Enhanced greedy with limited backtracking
// ============================================================================

__attribute__((visibility("default")))
SolverResult OptimisePlacementBeamSearch(
    const MachineDeployment* mds, int num_mds,
    const Pod* pods, int num_pods,
    const double* plugin_scores,
    const int* allowed_matrix,
    int* out_slot_assignments,
    uint8_t* out_slots_used,
    const int* beam_width,        // Beam width (default: 3)
    const int* md_candidates      // MD candidates per pod (default: 5)
) {
    {
        std::ofstream debug("/tmp/beam_cpp_entry.txt", std::ios::app);
        debug << ">>> OptimisePlacementBeamSearch ENTERED: " << num_pods << " pods, " << num_mds << " MDs" << std::endl;
    }

    SolverResult result = {false, 0.0, 0, 0.0, 0.0, false, false, false, 0, 0, 0};

    // Parse parameters with defaults
    int width = (beam_width != nullptr) ? *beam_width : 3;
    int candidates = (md_candidates != nullptr) ? *md_candidates : 5;

    // Validate parameters
    if (width < 1) width = 1;
    if (width > 10) width = 10;  // Limit max beam width to prevent explosion
    if (candidates < 1) candidates = 1;
    if (candidates > 20) candidates = 20;  // Limit max candidates

    auto start = std::chrono::high_resolution_clock::now();

    // Run beam search
    BeamResult beam_result = RunBeamSearch(mds, num_mds, pods, num_pods,
                                           plugin_scores, allowed_matrix,
                                           width, candidates);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    {
        std::ofstream debug("/tmp/beam_cpp_entry.txt", std::ios::app);
        debug << ">>> all_placed=" << beam_result.all_placed << ", unplaced_count=" << beam_result.unplaced_count << std::endl;
    }

    // Check if all pods placed
    if (!beam_result.all_placed) {
        std::ofstream debug("/tmp/beam_cpp_entry.txt", std::ios::app);
        debug << ">>> EARLY RETURN: not all placed" << std::endl;
        result.success = false;
        result.unplaced_pods = beam_result.unplaced_count;
        result.used_beam_search = true;
        result.solve_time_secs = elapsed.count();
        return result;
    }

    {
        std::ofstream debug("/tmp/beam_cpp_entry.txt", std::ios::app);
        debug << ">>> SUCCESS PATH" << std::endl;
    }

    // Calculate score
    double score = CalculateSolutionCost(beam_result.assignments,
                                         beam_result.slots_used,
                                         mds, num_mds, pods, num_pods,
                                         plugin_scores);

    // Copy result to output
    CopyBeamResultToOutput(beam_result, mds, num_mds, num_pods,
                           out_slot_assignments, out_slots_used);

    result.success = true;
    result.objective = score;
    result.status_code = static_cast<int>(FEASIBLE);
    result.solve_time_secs = elapsed.count();
    result.used_beam_search = true;
    result.cpsat_attempts = 0;
    result.best_attempt = 1;
    result.unplaced_pods = 0;

    return result;
}

}  // extern "C"
