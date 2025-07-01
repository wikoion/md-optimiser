// Core headers from the OR-Tools Constraint Programming (CP-SAT) solver
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_utils.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/util/sorted_interval_list.h"

#include <absl/container/flat_hash_set.h>
#include <utility>
#include <vector>
#include <cmath>
#include <string>
#include <iostream>

// Bring OR-Tools namespace into scope
using namespace operations_research;
namespace sat = operations_research::sat;


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
};

__attribute__((visibility("default")))
SolverResult OptimisePlacement(
    const MachineDeployment* mds, int num_mds,
    const Pod* pods, int num_pods,
    const double* plugin_scores,
    const int* allowed_matrix,
    const int* initial_assignment,
    int* out_assignments,
    int* out_nodes_used,
    const int* max_runtime_secs
) {
    SolverResult result;
    sat::CpModelBuilder model;

    std::vector<int> slots_per_md(num_mds);
    std::vector<std::vector<sat::BoolVar>> x(num_pods);
    std::vector<std::vector<int>> x_index(num_pods);
    std::vector<std::vector<sat::BoolVar>> slot_used(num_mds);
    std::vector<int> slot_base(num_mds + 1, 0);
    std::vector<sat::IntVar> assignment(num_pods);

    for (int j = 0; j < num_mds; ++j) {
        int max_slots = std::min(mds[j].max_scale_out, num_pods);
        slots_per_md[j] = max_slots;
        slot_used[j].resize(max_slots);
        for (int k = 0; k < max_slots; ++k) {
            slot_used[j][k] = model.NewBoolVar();
        }
        slot_base[j + 1] = slot_base[j] + max_slots;
    }
    int total_slots = slot_base[num_mds];

    for (int i = 0; i < num_pods; ++i) {
        assignment[i] = model.NewIntVar(sat::Domain(0, total_slots - 1));
    }

    for (int i = 0; i < num_pods; ++i) {
        std::vector<sat::BoolVar> vars;
        std::vector<int> indices;
        for (int j = 0; j < num_mds; ++j) {
            if (!allowed_matrix[i * num_mds + j]) continue;
            for (int k = 0; k < slots_per_md[j]; ++k) {
                vars.push_back(model.NewBoolVar());
                indices.push_back(slot_base[j] + k);
            }
        }
        x[i] = std::move(vars);
        x_index[i] = std::move(indices);
        model.AddEquality(sat::LinearExpr::Sum(x[i]), 1);
        model.AddEquality(sat::LinearExpr::ScalProd(x[i], x_index[i]), assignment[i]);
    }

    absl::flat_hash_set<std::pair<int, int>> affinity_seen;
    for (int i = 0; i < num_pods; ++i) {
        const Pod& pod = pods[i];
        for (int p = 0; p < pod.affinity_count; ++p) {
            int other = pod.affinity_peers[p];
            int rule = pod.affinity_rules[p];
            if (other < 0 || other >= num_pods || i >= other || rule == 0) continue;
            auto pair = std::minmax(i, other);
            if (!affinity_seen.insert(pair).second) continue;
            if (rule == 1) {
                model.AddEquality(assignment[i], assignment[other]);
            } else {
                model.AddNotEqual(assignment[i], assignment[other]);
            }
        }
    }

    if (initial_assignment != nullptr) {
        for (int i = 0; i < num_pods; ++i) {
            int hint_md = initial_assignment[i];
            if (hint_md >= 0 && hint_md < num_mds && allowed_matrix[i * num_mds + hint_md] && slots_per_md[hint_md] > 0) {
                model.AddHint(assignment[i], slot_base[hint_md]);
            }
        }
    }



    for (int j = 0; j < num_mds; ++j) {
        int slots = slots_per_md[j];
        sat::LinearExpr used_sum;
        for (int k = 0; k < slots; ++k) {
            int global = slot_base[j] + k;
            sat::LinearExpr cpu_sum, mem_sum, assigned;
            for (int i = 0; i < num_pods; ++i) {
                for (size_t v = 0; v < x_index[i].size(); ++v) {
                    if (x_index[i][v] == global) {
                        cpu_sum += x[i][v] * static_cast<int>(pods[i].cpu * 100);
                        mem_sum += x[i][v] * static_cast<int>(pods[i].memory * 100);
                        assigned += x[i][v];
                    }
                }
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

    absl::flat_hash_set<std::pair<int, int>> soft_seen;
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
            int scaled_weight = static_cast<int>(std::abs(affinity) * 100);
            sat::BoolVar penalty = model.NewBoolVar();
            if (rule == 1) {
                model.AddEquality(assignment[i], assignment[peer]).OnlyEnforceIf(penalty.Not());
                model.AddNotEqual(assignment[i], assignment[peer]).OnlyEnforceIf(penalty);
            } else {
                model.AddNotEqual(assignment[i], assignment[peer]).OnlyEnforceIf(penalty.Not());
                model.AddEquality(assignment[i], assignment[peer]).OnlyEnforceIf(penalty);
            }
            total_penalty += penalty * scaled_weight;
        }
    }

    sat::LinearExpr objective;
    for (int j = 0; j < num_mds; ++j) {
        int weight = static_cast<int>((1.0 - plugin_scores[j]) * 100) + 1;
        sat::LinearExpr md_used;
        for (int k = 0; k < slots_per_md[j]; ++k) {
            md_used += slot_used[j][k];
        }
        objective += md_used * weight;
    }
    objective += total_penalty;
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
        return result;
    }

    for (int i = 0; i < num_pods; ++i) {
        int64_t slot = sat::SolutionIntegerValue(response, assignment[i]);
        int md = 0;
        while (md + 1 < num_mds && slot >= slot_base[md + 1]) ++md;
        out_assignments[i] = md;
    }

    for (int j = 0; j < num_mds; ++j) {
        int count = 0;
        for (int k = 0; k < slots_per_md[j]; ++k) {
            if (sat::SolutionBooleanValue(response, slot_used[j][k])) {
                ++count;
            }
        }
        out_nodes_used[j] = count;
    }

    return result;
}

}  // extern "C"
