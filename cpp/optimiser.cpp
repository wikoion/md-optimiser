#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_utils.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/util/sorted_interval_list.h"

#include <vector>
#include <cmath>
#include <string>
#include <iostream>

using namespace operations_research;
namespace sat = operations_research::sat;

extern "C" {

// Structure describing a type of machine deployment (instance type)
struct MachineDeployment {
    const char* name;
    double cpu;           // CPU per node
    double memory;        // Memory per node
    int max_scale_out;    // Maximum number of nodes allowed
};

// Structure describing a pod to be scheduled
struct Pod {
    double cpu;
    double memory;
};

// C-callable optimization entry point
__attribute__((visibility("default")))
void OptimisePlacement(
    const MachineDeployment* mds, int num_mds,
    const Pod* pods, int num_pods,
    const double* plugin_scores,
    const int* allowed_matrix,
    const int* initial_assignment,
    int* out_assignments,
    int* out_nodes_used
) {
    // Create the constraint programming model
    sat::CpModelBuilder model;

    // Decision variables: x[i][j] is true if pod i is placed on deployment j
    std::vector<std::vector<sat::BoolVar>> x(num_pods, std::vector<sat::BoolVar>(num_mds));
    for (int i = 0; i < num_pods; ++i) {
        for (int j = 0; j < num_mds; ++j) {
            x[i][j] = model.NewBoolVar();
            // Disallow assignments that are not in the allowed matrix
            if (!allowed_matrix[i * num_mds + j]) {
                model.AddEquality(x[i][j], 0);
            }
        }
    }

    // Optional: provide a hint to the solver using an initial greedy assignment
    if (initial_assignment != nullptr) {
        for (int i = 0; i < num_pods; ++i) {
            int hint_md = initial_assignment[i];
            if (hint_md >= 0 && hint_md < num_mds) {
                model.AddHint(x[i][hint_md], 1);
            }
        }
    }

    // Constraint: each pod must be assigned to exactly one machine deployment
    for (int i = 0; i < num_pods; ++i) {
        std::vector<sat::BoolVar> choices;
        for (int j = 0; j < num_mds; ++j) {
            choices.push_back(x[i][j]);
        }
        model.AddEquality(sat::LinearExpr::Sum(choices), 1);
    }

    // Track how many nodes are used for each machine deployment
    std::vector<sat::IntVar> nodes_used(num_mds);
    for (int j = 0; j < num_mds; ++j) {
        nodes_used[j] = model.NewIntVar(Domain(0, mds[j].max_scale_out));

        sat::LinearExpr cpu_sum;
        sat::LinearExpr mem_sum;

        // Accumulate total CPU and memory demands of pods assigned to deployment j
        for (int i = 0; i < num_pods; ++i) {
            cpu_sum += x[i][j] * static_cast<int>(pods[i].cpu * 100);
            mem_sum += x[i][j] * static_cast<int>(pods[i].memory * 100);
        }

        // Constraint: total demand must fit within provisioned nodes
        model.AddLessOrEqual(cpu_sum, nodes_used[j] * static_cast<int>(mds[j].cpu * 100));
        model.AddLessOrEqual(mem_sum, nodes_used[j] * static_cast<int>(mds[j].memory * 100));
    }

    // Objective function: minimize weighted sum of nodes used across all deployments
    // Weights are derived from plugin scores (lower is better)
    sat::LinearExpr objective;
    for (int j = 0; j < num_mds; ++j) {
        int weight = static_cast<int>((1.0 - plugin_scores[j]) * 1000.0);
        objective += nodes_used[j] * weight;
    }
    model.Minimize(objective);

    // Solve the model
    const sat::CpSolverResponse response = sat::Solve(model.Build());

    // If the solver found no feasible solution, report and exit
    if (response.status() != sat::CpSolverStatus::OPTIMAL &&
        response.status() != sat::CpSolverStatus::FEASIBLE) {
        std::cerr << "No solution found.\n";
        return;
    }

    // Extract the pod-to-deployment assignments from the solution
    for (int i = 0; i < num_pods; ++i) {
        for (int j = 0; j < num_mds; ++j) {
            if (sat::SolutionBooleanValue(response, x[i][j])) {
                out_assignments[i] = j;
                break;
            }
        }
    }

    // Extract number of nodes used for each machine deployment
    for (int j = 0; j < num_mds; ++j) {
        int used = static_cast<int>(sat::SolutionIntegerValue(response, nodes_used[j]));
        out_nodes_used[j] = used;
    }
}

} // extern "C"
