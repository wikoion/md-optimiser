package optimiser_test

import (
	"testing"

	optimiser "github.com/wikoion/md-optimiser"
)

// This test reproduces the failing scenario from the cluster logs
func TestReproduceClusterFailure(t *testing.T) {
	// MDs from the logs with DaemonSet overhead applied
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "c7i.large", cpu: 0.708, memory: 1.8131177425384521, maxScaleOut: 9},
		&mockMD{name: "r7i.large", cpu: 0.708, memory: 12.813117742538452, maxScaleOut: 8},
		&mockMD{name: "r7i.xlarge", cpu: 2.708, memory: 26.813117742538452, maxScaleOut: 10},
	}

	// Pods from the logs
	pods := []optimiser.Pod{
		&mockPod{cpu: 0.1, memory: 0.48828125},            // my-deployment
		&mockPod{cpu: 1.0, memory: 2.0},                   // iox-shared-catalog-0
		&mockPod{cpu: 1.2, memory: 3.0},                   // iox-shared-ingester-2
		&mockPod{cpu: 1.47, memory: 5.146484375},          // iox-shared-querier
	}

	numMDs := len(mds)
	numPods := len(pods)

	// Plugin scores (normalized, all equal for simplicity)
	scores := make([]float64, numMDs)
	for i := range scores {
		scores[i] = 1.0
	}

	// Allowed matrix - pod 0 can only go on first 3 MDs (from logs: allowed_mds: 3)
	// Other pods can go on any MD that fits them
	allowed := make([]int, numPods*numMDs)
	for i := 0; i < numPods; i++ {
		for j := 0; j < numMDs; j++ {
			if i == 0 && j < 3 {
				// Pod 0 allowed on first 3 MDs
				allowed[i*numMDs+j] = 1
			} else if i == 3 && j >= 1 {
				// Pod 3 (querier) allowed on MDs that can fit it (not c7i.large with only 0.708 CPU)
				allowed[i*numMDs+j] = 1
			} else if i > 0 {
				// Other pods allowed on all MDs
				allowed[i*numMDs+j] = 1
			}
		}
	}

	initial := make([][][]int, 0)
	runtime := 15

	trueVal := true
	defaultAttempt := 1

	config := &optimiser.OptimizationConfig{
		MaxRuntimeSeconds:     &runtime,
		UseBeamSearchHint:     &trueVal,
		BeamSearchHintAttempt: &defaultAttempt,
		FallbackToBeamSearch:  &trueVal,
	}

	t.Logf("Running optimizer with %d MDs and %d pods", numMDs, numPods)
	for i, md := range mds {
		t.Logf("MD %d: %s cpu=%.3f mem=%.3f max=%d", i, md.GetName(), md.GetCPU(), md.GetMemory(), md.GetMaxScaleOut())
	}
	for i, pod := range pods {
		t.Logf("Pod %d: cpu=%.3f mem=%.3f", i, pod.GetCPU(), pod.GetMemory())
	}

	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)

	t.Logf("Result: succeeded=%v status=%d objective=%.2f solve_time=%.3fs",
		result.Succeeded, result.SolverStatus, result.Objective, result.SolveTimeSecs)
	t.Logf("  used_beam_search=%v beam_fallback=%v cpsat_attempts=%d",
		result.UsedBeamSearch, result.BeamSearchFallback, result.CPSATAttempts)
	t.Logf("  message: %s", result.Message)

	if !result.Succeeded {
		t.Fatalf("Optimizer failed to find solution: %s", result.Message)
	}

	// Verify all pods were assigned
	if len(result.PodAssignments) != numPods {
		t.Fatalf("Expected %d assignments, got %d", numPods, len(result.PodAssignments))
	}
}
