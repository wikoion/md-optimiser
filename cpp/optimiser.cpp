// Core headers from the OR-Tools Constraint Programming (CP-SAT) solver
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"

#include <unordered_set>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
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

public:
    explicit Optimiser(
       int num_mds, const MachineDeployment* mds,
       int num_pods, const Pod* pods,
       const int* allowed_matrix,
       sat::CpModelBuilder* model
    ) : num_mds(num_mds), mds(mds), num_pods(num_pods), pods(pods), allowed_matrix(allowed_matrix),
        model(model), slots_per_md(num_mds), slot_used(num_mds), pod_slot_assignment(num_pods)
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

};

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
    const double* improvement_threshold
) {
    SolverResult result;
    sat::CpModelBuilder model;

    Optimiser optimiser = Optimiser(num_mds, mds, num_pods, pods, allowed_matrix, &model);


    optimiser.InitSlotsUsed();
    optimiser.InitPodSlotAssignment();
    
    auto slots_per_md = optimiser.GetSlotsPerMd();
    auto pod_slot_assignment = optimiser.GetPodSlotAssignment();
    auto slot_used = optimiser.GetSlotUsed();

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

    if (initial_assignment != nullptr) {
        int offset = 0;
        for (int i = 0; i < num_pods; ++i) {
            for (int j = 0; j < num_mds; ++j) {
                for (int k = 0; k < slots_per_md[j]; ++k) {
                    if (initial_assignment[offset] == 1 && allowed_matrix[i * num_mds + j]) {
                        model.AddHint(pod_slot_assignment[i][j][k], 1);
                    }
                    offset++;
                }
            }
        }
    }

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
        }
        model.AddLessOrEqual(used_sum, mds[j].max_scale_out);
        for (int k = 0; k + 1 < slots; ++k) {
            model.AddImplication(slot_used[j][k + 1], slot_used[j][k]);
        }
    }

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

    sat::LinearExpr objective;
    for (int j = 0; j < num_mds; ++j) {
        int weight = static_cast<int>((1.0 - plugin_scores[j]) * 1000.0) + 1;
        int slots = slots_per_md[j];
        for (int k = 0; k < slots; ++k) {
            objective += slot_used[j][k] * weight;
        }
    }
    objective += static_cast<int>(0.25 * 1000.0) * total_penalty;
    
    // Calculate current state cost based on where pods are currently assigned
    double current_state_cost = 0.0;
    std::vector<bool> md_currently_used(num_mds, false);
    
    // Mark which MDs are currently being used
    for (int i = 0; i < num_pods; ++i) {
        int current_md = pods[i].current_md_assignment;
        if (current_md >= 0 && current_md < num_mds) {
            md_currently_used[current_md] = true;
        }
    }
    
    // Calculate current cost based on which MDs are currently used
    for (int j = 0; j < num_mds; ++j) {
        if (md_currently_used[j]) {
            int weight = static_cast<int>((1.0 - plugin_scores[j]) * 1000.0) + 1;
            current_state_cost += weight; // Cost of using this MD (1 node)
        }
    }
    result.current_state_cost = current_state_cost;
    
    model.Minimize(objective);

    sat::Model cp_model;
    sat::SatParameters parameters;
    if (max_runtime_secs != nullptr) {
        parameters.set_max_time_in_seconds(static_cast<double>(*max_runtime_secs));
    }
    parameters.set_random_seed(42);
    parameters.set_randomize_search(false);
    
    cp_model.Add(sat::NewSatParameters(parameters));

    const sat::CpSolverResponse response = sat::SolveCpModel(model.Build(), &cp_model);

    result.status_code = static_cast<int>(response.status());
    result.success = response.status() == sat::CpSolverStatus::OPTIMAL ||
                     response.status() == sat::CpSolverStatus::FEASIBLE;
    result.solve_time_secs = response.wall_time();
    result.objective = response.objective_value();

    if (!result.success) {
        std::cerr << "No solution found.\n";
        result.already_optimal = false; // Can't be optimal if no solution found
        return result;
    }
    
    // Check if improvement meets threshold
    result.already_optimal = false;
    if (improvement_threshold != nullptr && current_state_cost > 0.0) {
        double improvement_percentage = (current_state_cost - result.objective) / current_state_cost * 100.0;
        if (improvement_percentage < *improvement_threshold) {
            result.already_optimal = true;
            // Still return the solution, but mark it as not meeting threshold
        }
    }

    for (int i = 0; i < num_pods; ++i) {
        for (int j = 0; j < num_mds; ++j) {
            if (!allowed_matrix[i * num_mds + j]) continue;
            for (int k = 0; k < slots_per_md[j]; ++k) {
                if (sat::SolutionBooleanValue(response, pod_slot_assignment[i][j][k])) {
                    out_slot_assignments[i * 2] = j;
                    out_slot_assignments[i * 2 + 1] = k;
                    goto assigned;
                }
            }
        }
    assigned:;
    }

    int offset = 0;
    for (int j = 0; j < num_mds; ++j) {
        for (int k = 0; k < slots_per_md[j]; ++k) {
            out_slots_used[offset++] = sat::SolutionBooleanValue(response, slot_used[j][k]) ? 1 : 0;
        }
    }

    return result;
    }

}  // extern "C"
