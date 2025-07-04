package optimiser_test

import (
	"fmt"
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
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
	cpu, memory        float64
	labels             map[string]string
	affinityPeers      []int
	affinityRules      []int
	softAffinityPeers  []int
	softAffinityValues []float64
}

func (p *mockPod) GetCPU() float64                  { return p.cpu }
func (p *mockPod) GetMemory() float64               { return p.memory }
func (p *mockPod) GetLabel(key string) string       { return p.labels[key] }
func (p *mockPod) GetAffinityPeers() []int          { return p.affinityPeers }
func (p *mockPod) GetAffinityRules() []int          { return p.affinityRules }
func (p *mockPod) GetSoftAffinityPeers() []int      { return p.softAffinityPeers }
func (p *mockPod) GetSoftAffinityValues() []float64 { return p.softAffinityValues }

func TestOptimisePlacementRaw_PrefersLargestMD(t *testing.T) {
	mds := []optimiser.MachineDeployment{}
	scores := []float64{}

	// Create 6 MDs with increasing resources and plugin scores
	for i := 1; i <= 6; i++ {
		mds = append(mds, &mockMD{
			name:        fmt.Sprintf("md-%d", i),
			cpu:         float64(i), // 1.0 vCPU to 6.0 vCPU
			memory:      float64(i), // 1 to 6 GiB
			maxScaleOut: 10,
		})
		scores = append(scores, float64(i)/6.0) // Normalized score
	}

	// Generate 10 pods with increasing size
	pods := []optimiser.Pod{}
	for i := 1; i <= 10; i++ {
		pods = append(pods, &mockPod{
			cpu:    float64(i*250) / 1000.0, // 0.25 to 2.5 vCPU
			memory: float64(i*300) / 1024.0, // 0.29 to 2.93 GiB
		})
	}

	numPods := len(pods)
	numMDs := len(mds)

	allowed := make([]int, 0, numPods*numMDs)
	for range pods {
		for range mds {
			allowed = append(allowed, 1)
		}
	}
	initial := make([]int, numPods)

	runtime := 15
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, &runtime)
	if !result.Succeeded {
		t.Fatalf("Expected success: %s", result.Message)
	}

	assert.Len(t, result.Assignments, numPods)
	assert.Len(t, result.NodesUsed, numMDs)

	// Expect most pods to land on md-6
	md6Index := 5
	assert.GreaterOrEqual(t, result.NodesUsed[md6Index], 3,
		"Expected majority of pods on md-6 due to highest score")

	t.Logf("Objective: %.2f", result.Objective)
	t.Logf("Nodes used: %+v", result.NodesUsed)
	t.Logf("Assignments: %+v", result.Assignments)
}

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

	allowed := make([]int, 0, numPods*numMDs)
	for range pods {
		for range mds {
			allowed = append(allowed, 1)
		}
	}
	scores := []float64{0.3, 0.5, 0.8}
	initial := make([]int, numPods)

	runtime := 15
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, &runtime)

	if !result.Succeeded {
		t.Fatalf("Expected success, got failure: %s", result.Message)
	}
	if result.SolverStatus != 2 && result.SolverStatus != 4 {
		t.Errorf("Expected feasible or optimal status, got %d", result.SolverStatus)
	}
}

func TestOptimisePlacementRaw_SmallFeasible(t *testing.T) {
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-large", cpu: 16, memory: 64, maxScaleOut: 2},
	}

	pods := []optimiser.Pod{
		&mockPod{cpu: 4, memory: 8},
		&mockPod{cpu: 6, memory: 16},
		&mockPod{cpu: 2, memory: 4},
	}

	allowed := []int{1, 1, 1}
	scores := []float64{0.0}
	initial := []int{0, 0, 0}

	runtime := 15
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, &runtime)

	if !result.Succeeded {
		t.Fatalf("Expected success, got failure: %s", result.Message)
	}
	if result.SolverStatus != 2 && result.SolverStatus != 4 {
		t.Errorf("Expected feasible or optimal status, got %d", result.SolverStatus)
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
	allowed := []int{0}
	initial := []int{0}

	runtime := 15
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, &runtime)

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
