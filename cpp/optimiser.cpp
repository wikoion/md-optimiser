// Core headers from the OR-Tools Constraint Programming (CP-SAT) solver
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_utils.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/util/sorted_interval_list.h"

#include <vector>
#include <cmath>
#include <string>
#include <iostream>

// Bring OR-Tools namespace into scope
using namespace operations_research;
namespace sat = operations_research::sat;

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
    // 2. Provide hint (optional)
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
    // 3. Assignment constraint
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
    // 4. Capacity constraints
    // ------------------------

    // For each deployment, compute how many nodes are needed and ensure total resource fits
    std::vector<sat::IntVar> nodes_used(num_mds);
    for (int j = 0; j < num_mds; ++j) {
        // Variable: number of nodes to provision for deployment j
        nodes_used[j] = model.NewIntVar(Domain(0, mds[j].max_scale_out));

        sat::LinearExpr cpu_sum;
        sat::LinearExpr mem_sum;

        // Sum of all pod CPU/memory placed on this deployment
        for (int i = 0; i < num_pods; ++i) {
            cpu_sum += x[i][j] * static_cast<int>(pods[i].cpu * 100);     // use 100x to convert to int
            mem_sum += x[i][j] * static_cast<int>(pods[i].memory * 100);
        }

        // Ensure pod demand fits within provisioned nodes * per-node capacity
        model.AddLessOrEqual(cpu_sum, nodes_used[j] * static_cast<int>(mds[j].cpu * 100));
        model.AddLessOrEqual(mem_sum, nodes_used[j] * static_cast<int>(mds[j].memory * 100));
    }

    // ------------------------
    // 5. Objective function
    // ------------------------

    // The goal is to minimize the total weighted cost of nodes used
    // Weights are derived from plugin_scores: higher scores = more desirable = lower weight
    sat::LinearExpr objective;
    for (int j = 0; j < num_mds; ++j) {
        int weight = static_cast<int>((1.0 - plugin_scores[j]) * 1000.0);  // inverse weighting
        objective += nodes_used[j] * weight;
    }

    model.Minimize(objective);  // Set objective

    // ------------------------
    // 6. Solve
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
    // 7. Extract solution
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
