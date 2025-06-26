package optimiser_test

import (
	"regexp"
	"testing"

	optimiser "github.com/wikoion/md-optimiser"
)

type mockMD struct {
	name        string
	cpu         float64
	memory      float64
	maxScaleOut int
}

func (m *mockMD) GetName() string     { return m.name }
func (m *mockMD) GetCPU() float64     { return m.cpu }
func (m *mockMD) GetMemory() float64  { return m.memory }
func (m *mockMD) GetMaxScaleOut() int { return m.maxScaleOut }

type mockPod struct {
	cpu, memory      float64
	labels           map[string]string
	affinityPeers    []int
	affinityRules    []int
	colocationPref   int
	colocationWeight float64
}

func (p *mockPod) GetCPU() float64              { return p.cpu }
func (p *mockPod) GetMemory() float64           { return p.memory }
func (p *mockPod) GetLabel(key string) string   { return p.labels[key] }
func (p *mockPod) GetAffinityPeers() []int      { return p.affinityPeers }
func (p *mockPod) GetAffinityRules() []int      { return p.affinityRules }
func (p *mockPod) GetColocationPreference() int { return p.colocationPref }
func (p *mockPod) GetColocationWeight() float64 { return p.colocationWeight }

func TestOptimisePlacementRaw_NoAffinity(t *testing.T) {
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-a", cpu: 16, memory: 64, maxScaleOut: 2},
		&mockMD{name: "md-b", cpu: 8, memory: 32, maxScaleOut: 2},
		&mockMD{name: "md-c", cpu: 4, memory: 16, maxScaleOut: 3},
	}

	pods := []optimiser.Pod{
		&mockPod{cpu: 2, memory: 4},
		&mockPod{cpu: 4, memory: 8},
		&mockPod{cpu: 1, memory: 2},
		&mockPod{cpu: 6, memory: 12},
		&mockPod{cpu: 3, memory: 6},
	}

	numPods := len(pods)
	numMDs := len(mds)

	// All pods allowed on all MDs
	allowed := make([]int, 0, numPods*numMDs)
	for range pods {
		for range mds {
			allowed = append(allowed, 1)
		}
	}

	// Simulated plugin scores: favor md-c slightly
	scores := []float64{0.3, 0.5, 0.8}

	// No affinity at all: no special fields
	initial := make([]int, numPods)

	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial)

	if !result.Succeeded {
		t.Fatalf("Expected success, got failure: %s", result.Message)
	}
	if result.SolverStatus != 2 && result.SolverStatus != 4 {
		t.Errorf("Expected feasible or optimal status, got %d", result.SolverStatus)
	}

	t.Log("Assignments:")
	for i, mdIdx := range result.Assignments {
		t.Logf("  Pod %d → %s", i, mds[mdIdx].GetName())
	}
	t.Logf("Objective: %.2f", result.Objective)
}

func TestOptimisePlacementRaw_SmallFeasible(t *testing.T) {
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-large", cpu: 16, memory: 64, maxScaleOut: 2}, // Total: 32 CPU, 128 GB
	}

	pods := []optimiser.Pod{
		&mockPod{cpu: 4, memory: 8}, // Needs ~1/8th of 1 node
		&mockPod{cpu: 6, memory: 16},
		&mockPod{cpu: 2, memory: 4},
	}

	// Allowed: all 1s
	allowed := []int{
		1,
		1,
		1,
	}

	scores := []float64{0.0}
	initial := []int{0, 0, 0}

	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial)

	if !result.Succeeded {
		t.Fatalf("Expected success, got failure: %s", result.Message)
	}
	if result.SolverStatus != 2 && result.SolverStatus != 4 {
		t.Errorf("Expected feasible or optimal status, got %d", result.SolverStatus)
	}
}

func TestOptimisePlacementRaw_ModerateWithAffinity(t *testing.T) {
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-a", cpu: 16, memory: 64, maxScaleOut: 2}, // 32 CPU, 128 GiB
		&mockMD{name: "md-b", cpu: 8, memory: 32, maxScaleOut: 2},  // 16 CPU, 64 GiB
	}

	pods := []optimiser.Pod{
		&mockPod{cpu: 4, memory: 8}, // pod 0
		&mockPod{cpu: 2, memory: 4, affinityPeers: []int{0}, affinityRules: []int{1}}, // pod 1: must colocate w/ 0
		&mockPod{cpu: 6, memory: 16}, // pod 2
		&mockPod{cpu: 1, memory: 2},  // pod 3
	}

	numPods := len(pods)
	numMDs := len(mds)

	// Allow all pods on all MDs
	allowed := make([]int, 0, numPods*numMDs)
	for range pods {
		for range mds {
			allowed = append(allowed, 1)
		}
	}

	// Slight preference for md-a (simulate a plugin)
	scores := []float64{0.9, 0.1}

	initial := make([]int, numPods)

	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial)

	if !result.Succeeded {
		t.Fatalf("Expected success, got failure: %s", result.Message)
	}
	if result.SolverStatus != 2 && result.SolverStatus != 4 {
		t.Errorf("Expected feasible or optimal status, got %d", result.SolverStatus)
	}
	if len(result.Assignments) != numPods {
		t.Errorf("Expected %d assignments, got %d", numPods, len(result.Assignments))
	}
	if len(result.NodesUsed) != numMDs {
		t.Errorf("Expected %d nodesUsed entries, got %d", numMDs, len(result.NodesUsed))
	}

	t.Log("Assignments:")
	for i, mdIdx := range result.Assignments {
		t.Logf("  Pod %d → %s", i, mds[mdIdx].GetName())
	}

	if result.Assignments[0] != result.Assignments[1] {
		t.Errorf("Affinity failed: Pod 0 and 1 not colocated (got %d, %d)", result.Assignments[0], result.Assignments[1])
	}
}

func TestOptimisePlacementRaw_IncompatiblePodFails(t *testing.T) {
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-tiny", cpu: 1, memory: 1, maxScaleOut: 1},
	}

	pods := []optimiser.Pod{
		&mockPod{cpu: 4, memory: 8},
	}

	scores := []float64{0.0}
	allowed := []int{0} // pod not allowed on MD
	initial := []int{0}

	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial)

	if result.Succeeded {
		t.Fatalf("Expected failure due to incompatibility, got success")
	}
	if result.SolverStatus == 2 {
		t.Errorf("Expected failure status code, got 2 (feasible)")
	}
	if matched, _ := regexp.MatchString("failed", result.Message); !matched {
		t.Errorf("Expected failure message, got: %s", result.Message)
	}
}
