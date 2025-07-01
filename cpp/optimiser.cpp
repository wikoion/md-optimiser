// Core headers from the OR-Tools Constraint Programming (CP-SAT) solver
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_utils.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/util/sorted_interval_list.h"

#include <unordered_set>
#include <utility>
#include <vector>
#include <cmath>
#include <string>
#include <iostream>

// Bring OR-Tools namespace into scope
using namespace operations_research;
namespace sat = operations_research::sat;

namespace std {
    template <>
    struct hash<std::pair<int, int>> {
        std::size_t operator()(const std::pair<int, int>& p) const noexcept {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };
}

// Make symbols accessible to C code / Go FFI (via cgo)
extern "C" {

// ------------------------
// Domain structs (shared with Go)
// ------------------------

// Describes a scalable group of machines with a type, size, and max count
struct MachineDeployment {
    const char* name;       // Human-readable name (e.g., "m5.large")
    double cpu;             // CPU per machine (e.g., 2.0 vCPU)
    double memory;          // Memory per machine (e.g., 4.0 GB)
    int max_scale_out;      // Max number of machines that can be added
};

// Describes a single workload unit that needs placement
struct Pod {
    double cpu;             // CPU required (e.g., 0.5 vCPU)
    double memory;          // Memory required (e.g., 1.0 GB)

    // Hard affinity rules: same or different node as listed pod indices
    const int* affinity_peers;  // sparse array of pod indices (only peer pod indices)
    const int* affinity_rules;  // matches above array for lookups, 1 = same node, -1 = different
    int affinity_count;

    // Soft preference: +0.8 = strong colocate preference, -0.8 = strong spread preference
    const int* soft_peers;          // sparse array of pod indices (only peer pod indices)
    const double* soft_affinities;  // matches above array for lookups, 0.x = same node, -0.x = different, 0 = none
    int soft_affinity_count;
};

// Enum for interpreting the solver's return status
enum SolverStatus {
    UNKNOWN = 0,
    MODEL_INVALID = 1,
    FEASIBLE = 2,
    INFEASIBLE = 3,
    OPTIMAL = 4
};

// Holds the result of the solver call (success flag, objective score, etc.)
struct SolverResult {
    bool success;           // Whether a valid solution was found
    double objective;       // Final objective value (lower = better)
    int status_code;        // SolverStatus enum as int
    double solve_time_secs; // Time the solver spent (wall time)
};

// ------------------------
// Optimization function (entrypoint)
// ------------------------

__attribute__((visibility("default")))
SolverResult OptimisePlacement(
    const MachineDeployment* mds, int num_mds,          // Array of deployments
    const Pod* pods, int num_pods,                      // Array of pods
    const double* plugin_scores,                        // Weight per deployment (0.0 - 1.0)
    const int* allowed_matrix,                          // 2D matrix: pod x deployment allowed (bool)
    const int* initial_assignment,                      // Optional: initial suggested pod-to-deployment
    int* out_assignments,                               // Output: per-pod selected deployment index
    int* out_nodes_used                                 // Output: per-deployment number of nodes used
) {
    SolverResult result;

    // ------------------------
    // 1. Define CP-SAT model
    // ------------------------

    sat::CpModelBuilder model;

    // Decision variable matrix: x[i][j] = true if pod i is assigned to deployment j
    std::vector<std::vector<sat::BoolVar>> x(num_pods, std::vector<sat::BoolVar>(num_mds));
    for (int i = 0; i < num_pods; ++i) {
        for (int j = 0; j < num_mds; ++j) {
            x[i][j] = model.NewBoolVar();  // Create boolean var

            // Enforce hard constraint: only allow assignments in the allowed matrix
            if (!allowed_matrix[i * num_mds + j]) {
                model.AddEquality(x[i][j], 0);  // x[i][j] must be false
            }
        }
    }

    // ------------------------
    // 2. Hard Affinity Constraints
    // ------------------------

    std::unordered_set<std::pair<int, int>> affinity_seen;

    for (int i = 0; i < num_pods; ++i) {
        const Pod& pod = pods[i];

        for (int p = 0; p < pod.affinity_count; ++p) {
            int other = pod.affinity_peers[p];
            int rule = pod.affinity_rules[p];

            if (other < 0 || other >= num_pods) continue;  // invalid index
            if (i >= other || rule == 0) continue;  // skip duplicates or no-ops

            auto pair = std::minmax(i, other);
            if (!affinity_seen.insert(pair).second) continue;  // already added

            for (int j = 0; j < num_mds; ++j) {
                if (rule == 1) {
                    // Must colocate
                    model.AddEquality(x[i][j], x[other][j]);
                } else if (rule == -1) {
                    // Must NOT colocate
                    model.AddBoolOr({x[i][j].Not(), x[other][j].Not()});
                }
            }
        }
    }

    // ------------------------
    // 3. Provide hint (optional)
    // ------------------------

    // A hint suggests a good starting point for the solver (not mandatory)
    if (initial_assignment != nullptr) {
        for (int i = 0; i < num_pods; ++i) {
            int hint_md = initial_assignment[i];
            if (hint_md >= 0 && hint_md < num_mds) {
                model.AddHint(x[i][hint_md], 1);  // Suggest placing pod i on deployment hint_md
            }
        }
    }

    // ------------------------
    // 4. Assignment constraint
    // ------------------------

    // Each pod must be assigned to exactly one deployment
    for (int i = 0; i < num_pods; ++i) {
        std::vector<sat::BoolVar> choices;
        for (int j = 0; j < num_mds; ++j) {
            choices.push_back(x[i][j]);
        }
        model.AddEquality(sat::LinearExpr::Sum(choices), 1);
    }

    // ------------------------
    // 5. Capacity constraints
    // ------------------------

    // For each deployment, compute how many nodes are needed and ensure total resource fits
    std::vector<sat::IntVar> nodes_used(num_mds);
    std::vector<sat::BoolVar> used_flag(num_mds);  // NEW: usage indicators

    for (int j = 0; j < num_mds; ++j) {
        // 1. Create IntVar for node count
        nodes_used[j] = model.NewIntVar(Domain(0, mds[j].max_scale_out));

        // 2. Accumulate total resource demand for this MD
        sat::LinearExpr cpu_sum;
        sat::LinearExpr mem_sum;
        sat::LinearExpr assigned_pods;

        for (int i = 0; i < num_pods; ++i) {
            cpu_sum += x[i][j] * static_cast<int>(pods[i].cpu * 100);
            mem_sum += x[i][j] * static_cast<int>(pods[i].memory * 100);
            assigned_pods += x[i][j];  // Count pods assigned to MD j
        }

        // 3. Add capacity constraints
        model.AddLessOrEqual(cpu_sum, nodes_used[j] * static_cast<int>(mds[j].cpu * 100));
        model.AddLessOrEqual(mem_sum, nodes_used[j] * static_cast<int>(mds[j].memory * 100));

        // 4. Add usage flag and binding constraints
        used_flag[j] = model.NewBoolVar();  // Create BoolVar indicating MD usage

        // If any pods assigned to j → used_flag[j] = true
        model.AddGreaterThan(assigned_pods, 0).OnlyEnforceIf(used_flag[j]);
        model.AddEquality(assigned_pods, 0).OnlyEnforceIf(used_flag[j].Not());

        // Link used_flag to nodes_used
        model.AddGreaterThan(nodes_used[j], 0).OnlyEnforceIf(used_flag[j]);
        model.AddEquality(nodes_used[j], 0).OnlyEnforceIf(used_flag[j].Not());
    }

    // ------------------------
    // 6. Soft Affinity Constraints
    // ------------------------
    sat::LinearExpr total_penalty;
    for (int i = 0; i < num_pods; ++i) {
        const Pod& pod = pods[i];

        for (int p = 0; p < pod.soft_affinity_count; ++p) {
            int peer = pod.soft_peers[p];
            double affinity = pod.soft_affinities[p];

            if (affinity == 0.0) continue;

            int rule = (affinity > 0) ? 1 : -1;
            int scaled_weight = static_cast<int>(std::abs(affinity) * 1000.0);

            for (int j = 0; j < num_mds; ++j) {
                sat::BoolVar penalty = model.NewBoolVar();

                if (rule == 1) {
                    model.AddBoolOr({x[i][j].Not(), x[peer][j].Not()}).OnlyEnforceIf(penalty);
                    model.AddBoolAnd({x[i][j], x[peer][j]}).OnlyEnforceIf(penalty.Not());
                } else {
                    model.AddBoolAnd({x[i][j], x[peer][j]}).OnlyEnforceIf(penalty);
                    model.AddBoolOr({x[i][j].Not(), x[peer][j].Not()}).OnlyEnforceIf(penalty.Not());
                }

                total_penalty += penalty * scaled_weight;
            }
        }
    }


    // ------------------------
    // 7. Objective function
    // ------------------------

    // The goal is to minimize the total weighted cost of nodes used
    // Weights are derived from plugin_scores: higher scores = more desirable = lower weight
    sat::LinearExpr objective;
    for (int j = 0; j < num_mds; ++j) {
        int weight = static_cast<int>((1.0 - plugin_scores[j]) * 1000.0);  // inverse weighting
        objective += nodes_used[j] * weight;
    }

    double affinity_factor = 0.25;
    objective += static_cast<int>(affinity_factor * 1000.0) * total_penalty;
    model.Minimize(objective);


    // ------------------------
    // 8. Solve
    // ------------------------

    const sat::CpSolverResponse response = sat::Solve(model.Build());

    // Parse solver status
    result.status_code = static_cast<int>(response.status());
    result.success = response.status() == sat::CpSolverStatus::OPTIMAL ||
                     response.status() == sat::CpSolverStatus::FEASIBLE;
    result.solve_time_secs = response.wall_time();
    result.objective = response.objective_value();

    // No feasible solution found
    if (!result.success) {
        std::cerr << "No solution found.\n";
        return result;
    }

    // ------------------------
    // 9. Extract solution
    // ------------------------

    // Output: for each pod, which deployment it's assigned to
    for (int i = 0; i < num_pods; ++i) {
        for (int j = 0; j < num_mds; ++j) {
            if (sat::SolutionBooleanValue(response, x[i][j])) {
                out_assignments[i] = j;
                break;
            }
        }
    }

    // Output: for each deployment, how many nodes are used
    for (int j = 0; j < num_mds; ++j) {
        int used = static_cast<int>(sat::SolutionIntegerValue(response, nodes_used[j]));
        out_nodes_used[j] = used;
    }

    return result;
}

}  // extern "C"
