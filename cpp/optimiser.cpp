#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_utils.h"
#include "ortools/util/sorted_interval_list.h"

#include <vector>
#include <cmath>
#include <string>
#include <iostream>

using namespace operations_research;
namespace sat = operations_research::sat;

extern "C" {

struct MachineDeployment {
    const char* name;
    double cpu;
    double memory;
    int max_scale_out;
};

struct Pod {
    double cpu;
    double memory;
};

__attribute__((visibility("default")))
void OptimisePlacement(
    const MachineDeployment* mds, int num_mds,
    const Pod* pods, int num_pods,
    const double* plugin_scores,
    int* out_assignments,
    int* out_nodes_used
) {
    sat::CpModelBuilder model;

    std::vector<std::vector<sat::BoolVar>> x(num_pods, std::vector<sat::BoolVar>(num_mds));
    for (int i = 0; i < num_pods; ++i) {
        for (int j = 0; j < num_mds; ++j) {
            x[i][j] = model.NewBoolVar();
        }
    }

    for (int i = 0; i < num_pods; ++i) {
        std::vector<sat::BoolVar> choices;
        for (int j = 0; j < num_mds; ++j) {
            choices.push_back(x[i][j]);
        }
        model.AddEquality(sat::LinearExpr::Sum(choices), 1);
    }

    std::vector<sat::IntVar> nodes_used(num_mds);
    for (int j = 0; j < num_mds; ++j) {
        nodes_used[j] = model.NewIntVar(Domain(0, mds[j].max_scale_out));

        sat::LinearExpr cpu_sum;
        sat::LinearExpr mem_sum;

        for (int i = 0; i < num_pods; ++i) {
            cpu_sum += x[i][j] * static_cast<int>(pods[i].cpu * 100);
            mem_sum += x[i][j] * static_cast<int>(pods[i].memory * 100);
        }

        model.AddLessOrEqual(cpu_sum, nodes_used[j] * static_cast<int>(mds[j].cpu * 100));
        model.AddLessOrEqual(mem_sum, nodes_used[j] * static_cast<int>(mds[j].memory * 100));
    }

    // Objective: weighted plugin score (Go provides score in [0,1], where higher = better)
    sat::LinearExpr objective;
    for (int j = 0; j < num_mds; ++j) {
        int weight = static_cast<int>((1.0 - plugin_scores[j]) * 1000.0);
        objective += nodes_used[j] * weight;
    }
    model.Minimize(objective);

    const sat::CpSolverResponse response = sat::Solve(model.Build());

    if (response.status() != sat::CpSolverStatus::OPTIMAL &&
        response.status() != sat::CpSolverStatus::FEASIBLE) {
        std::cerr << "No solution found.\n";
        return;
    }

    // Assign each pod
    for (int i = 0; i < num_pods; ++i) {
        for (int j = 0; j < num_mds; ++j) {
            if (sat::SolutionBooleanValue(response, x[i][j])) {
                out_assignments[i] = j;
                break;
            }
        }
    }

    // Report nodes used
    for (int j = 0; j < num_mds; ++j) {
        int used = static_cast<int>(sat::SolutionIntegerValue(response, nodes_used[j]));
        out_nodes_used[j] = used;
    }
}

} // extern "C"
