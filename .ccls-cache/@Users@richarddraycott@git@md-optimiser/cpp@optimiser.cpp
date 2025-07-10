// Core headers from the OR-Tools Constraint Programming (CP-SAT) solver
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"

#include <unordered_set>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>

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
}

// init_slots_used_matrix creates a zeroed 2d matrix of mds(ints) to slots(boolvars) 
void init_slots_used_matrix(
   int num_mds, const MachineDeployment* mds,
   sat::CpModelBuilder* model,
   vector<int>& slots_per_md,
   vector<vector<sat::BoolVar>>& slot_used
) {
    for (int i = 0; i < num_mds; ++i) {
        int max_slots = mds[i].max_scale_out;
        slots_per_md.at(i) = max_slots;
        slot_used.at(i).resize(max_slots);
        for (int j = 0; j < max_slots; ++j) {
            slot_used.at(i)[j] = model->NewBoolVar();
        }
    }
}

// init_pod_slot_assignment_matrix creates a zeroed 3d matrix/tensor of pods(int) to slots_used 
// (mds int to md slots boolvar)
void init_pod_slot_assignment_matrix(
   int num_mds, int num_pods,
   const int* allowed_matrix,
   sat::CpModelBuilder* model,
   vector<int>& slots_per_md,
   vector<vector<sat::BoolVar>>& slot_used,
   vector<vector<vector<sat::BoolVar>>>& pod_slot_assignment
) {
    for (int i = 0; i < num_pods; ++i) {
        pod_slot_assignment.at(i).resize(num_mds);
        for (int j = 0; j < num_mds; ++j) {
            if (!allowed_matrix[i * num_mds + j]) continue;
            int slots = slots_per_md.at(j);
            pod_slot_assignment.at(i)[j].resize(slots);
            for (int k = 0; k < slots; ++k) {
                pod_slot_assignment.at(i)[j][k] = model->NewBoolVar();
            }
        }
    }
}


void model_hard_pod_affinities(
   int num_mds, int num_pods,
   const Pod* pods,
   const int* allowed_matrix,
   sat::CpModelBuilder* model,
   vector<int>& slots_per_md,
   vector<vector<vector<sat::BoolVar>>>& pod_slot_assignment
) {
    std::unordered_set<std::pair<int, int>> affinity_seen;
    for (int i = 0; i < num_pods; ++i) {
        const Pod& pod = pods[i];

        for (int p = 0; p < pod.affinity_count; ++p) {
            int other = pod.affinity_peers[p];
            int rule = pod.affinity_rules[p];

            if (other < 0 || other >= num_pods || i >= other || rule == 0) continue;

            auto pair = std::minmax(i, other);

            if (!affinity_seen.insert(pair).second) continue;

            for (int j = 0; j < num_mds; ++j) {
                if (!allowed_matrix[i * num_mds + j] || !allowed_matrix[other * num_mds + j]) continue;
                int slots = slots_per_md[j];

                for (int k = 0; k < slots; ++k) {
                    if (rule == 1) {
                        model->AddEquality(
                                sat::LinearExpr(pod_slot_assignment[i][j][k]),
                                sat::LinearExpr(pod_slot_assignment[other][j][k])
                        );
                    } else {
                        model->AddBoolOr(absl::Span<const sat::BoolVar>(
                            {
                                pod_slot_assignment[i][j][k].Not(),
                                pod_slot_assignment[other][j][k].Not()
                            }
                        ));
                    }
                }
            }
        }
    }
}

extern "C" {
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
    std::vector<vector<vector<sat::BoolVar>>> pod_slot_assignment(num_pods);
    std::vector<vector<sat::BoolVar>> slot_used(num_mds);
    
    init_slots_used_matrix(num_mds, mds, &model, slots_per_md, slot_used);
    init_pod_slot_assignment_matrix(num_mds, num_pods, allowed_matrix, &model, slots_per_md, slot_used, pod_slot_assignment);
    model_hard_pod_affinities(num_mds, num_pods, pods, allowed_matrix, &model, slots_per_md, pod_slot_assignment);

    if (initial_assignment != nullptr) {
        for (int i = 0; i < num_pods; ++i) {
            int hint_md = initial_assignment[i];
            if (hint_md >= 0 && hint_md < num_mds && allowed_matrix[i * num_mds + hint_md] && slots_per_md[hint_md] > 0) {
                model.AddHint(pod_slot_assignment[i][hint_md][0], 1);
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
        absl::Span<const sat::BoolVar> span_choices(choices.data(), choices.size());
        model.AddEquality(sat::LinearExpr::Sum(span_choices), 1);
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
                assigned += sat::LinearExpr(pod_slot_assignment[i][j][k]);
            }
            model.AddLessOrEqual(cpu_sum, static_cast<int>(mds[j].cpu * 100));
            model.AddLessOrEqual(mem_sum, static_cast<int>(mds[j].memory * 100));
            model.AddGreaterThan(assigned, 0).OnlyEnforceIf(slot_used[j][k]);
            model.AddEquality(assigned, 0).OnlyEnforceIf(slot_used[j][k].Not());
            used_sum += sat::LinearExpr(slot_used[j][k]);
        }
        model.AddLessOrEqual(used_sum, mds[j].max_scale_out);
        for (int k = 0; k + 1 < slots; ++k) {
            model.AddImplication(slot_used[j][k + 1], slot_used[j][k]);
        }
    }

    std::unordered_set<std::pair<int, int>> soft_seen;
    using sat::operator*; 
    sat::LinearExpr total_penalty;
    for (int i = 0; i < num_pods; ++i) {
        const Pod& pod = pods[i];
        if (pod.soft_affinity_count == 0) continue;
        for (int p = 0; p < pod.soft_affinity_count; ++p) {
            int peer = pod.soft_peers[p];
            if (peer < 0 || peer >= num_pods) continue;
            std::pair<int, int> pair = std::minmax(i, peer);
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
                        model.AddBoolOr(absl::MakeConstSpan(
                            {
                                pod_slot_assignment[i][j][k].Not(),
                                pod_slot_assignment[peer][j][k].Not()
                            }
                        )).OnlyEnforceIf(penalty);

                        model.AddBoolOr(absl::MakeConstSpan(
                            {
                                pod_slot_assignment[i][j][k],
                                pod_slot_assignment[peer][j][k]
                            }
                        )).OnlyEnforceIf(penalty.Not());
                    } 

                    else {
                        model.AddBoolAnd(absl::MakeConstSpan(
                            {
                                pod_slot_assignment[i][j][k],
                                pod_slot_assignment[peer][j][k]
                            }
                        )).OnlyEnforceIf(penalty);
                        model.AddBoolAnd(absl::MakeConstSpan(
                            {
                                pod_slot_assignment[i][j][k].Not(),
                                pod_slot_assignment[peer][j][k].Not()
                            }
                        )).OnlyEnforceIf(penalty.Not());
                    }
                    total_penalty = sat::LinearExpr(penalty) * scaled_weight;
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
        for (int j = 0; j < num_mds; ++j) {
            if (!allowed_matrix[i * num_mds + j]) continue;
            for (int k = 0; k < slots_per_md[j]; ++k) {
                if (sat::SolutionBooleanValue(response, pod_slot_assignment[i][j][k])) {
                    out_assignments[i] = j;
                    goto assigned;
                }
            }
        }
    assigned:;
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
