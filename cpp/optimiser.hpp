#pragma once
#include <cstdint>

// Standard C++ protection for C-compatible headers
#ifdef __cplusplus
extern "C" {
#endif

struct MachineDeployment {
    const char* name;
    double cpu;
    double memory;
    int max_scale_out;
};

struct Pod {
    double cpu;
    double memory;
    const int* affinity_peers;
    const int* affinity_rules;
    int affinity_count;
    const int* soft_peers;
    const double* soft_affinities;
    int soft_affinity_count;
    int current_md_assignment;
};

struct SolverResult {
    bool success;
    double objective;
    int status_code;
    double solve_time_secs;
    double current_state_cost;
    bool already_optimal;
    bool used_beam_search;
    bool beam_search_fallback;
    int cpsat_attempts;
    int best_attempt;
    int unplaced_pods;
};

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
    const bool* use_beam_search_hint,
    const int* beam_search_hint_attempt,
    const bool* fallback_to_beam_search,
    const bool* beam_search_only
);

#ifdef __cplusplus
}
#endif
