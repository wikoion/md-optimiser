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

func (p *mockPod) GetCPU() float64                   { return p.cpu }
func (p *mockPod) GetMemory() float64                { return p.memory }
func (p *mockPod) GetLabel(key string) string        { return p.labels[key] }
func (p *mockPod) GetAffinityPeers() []int           { return p.affinityPeers }
func (p *mockPod) GetAffinityRules() []int           { return p.affinityRules }
func (p *mockPod) GetSoftAffinityPeers() []int       { return p.softAffinityPeers }
func (p *mockPod) GetSoftAffinityWeights() []float64 { return p.softAffinityValues }

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
	initial := make([][][]int, 0)

	runtime := 15
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, &runtime)
	if !result.Succeeded {
		t.Fatalf("Expected success: %s", result.Message)
	}

	assert.Len(t, result.PodAssignments, numPods)
	assert.Len(t, result.SlotsUsed, numMDs)

	// Expect most pods to land on md-6
	md6Index := 5
	usedCount := 0
	for _, used := range result.SlotsUsed[md6Index] {
		if used {
			usedCount++
		}
	}
	assert.GreaterOrEqual(t, usedCount, 3)

	t.Logf("Objective: %.2f", result.Objective)
	t.Logf("Slots used:")
	for j, mdSlots := range result.SlotsUsed {
		summary := ""
		for k, used := range mdSlots {
			if used {
				summary += fmt.Sprintf(" %d", k)
			}
		}
		t.Logf("  MD %d used slots:%s", j, summary)
	}
	t.Logf("Assignments:")
	for i, a := range result.PodAssignments {
		t.Logf("  Pod %d → MD %d Slot %d", i, a.MD, a.Slot)
	}

	for i, assign := range result.PodAssignments {
		assert.GreaterOrEqual(t, assign.MD, 0, fmt.Sprintf("Pod %d has invalid MD", i))
		assert.GreaterOrEqual(t, assign.Slot, 0, fmt.Sprintf("Pod %d has invalid slot", i))
	}
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
	initial := make([][][]int, 0)

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
	initial := make([][][]int, 0)

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
	initial := make([][][]int, 0)

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

func TestOptimisePlacementRaw_AntiAffinityThreePodsThreeSlots(t *testing.T) {
	// 1 MD with 3 slots, 3 pods with mutual anti-affinity
	// This should be feasible - each pod gets its own slot
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-large", cpu: 16, memory: 64, maxScaleOut: 3},
	}

	pods := []optimiser.Pod{
		&mockPod{
			cpu:           2,
			memory:        4,
			affinityPeers: []int{1, 2},   // anti-affinity with pods 1 and 2
			affinityRules: []int{-1, -1}, // -1 means anti-affinity
		},
		&mockPod{
			cpu:           2,
			memory:        4,
			affinityPeers: []int{0, 2},   // anti-affinity with pods 0 and 2
			affinityRules: []int{-1, -1}, // -1 means anti-affinity
		},
		&mockPod{
			cpu:           2,
			memory:        4,
			affinityPeers: []int{0, 1},   // anti-affinity with pods 0 and 1
			affinityRules: []int{-1, -1}, // -1 means anti-affinity
		},
	}

	numPods := len(pods)
	numMDs := len(mds)

	// All pods can be placed on the single MD
	allowed := make([]int, numPods*numMDs)
	for i := range allowed {
		allowed[i] = 1
	}
	scores := []float64{0.5}
	initial := make([][][]int, 0)

	runtime := 15
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, &runtime)

	if !result.Succeeded {
		t.Fatalf("Expected success with 3 pods, 3 slots, anti-affinity. Got failure: %s", result.Message)
	}

	// Verify each pod is on a different slot
	slots := make(map[int]bool)
	for i, assignment := range result.PodAssignments {
		if assignment.MD != 0 {
			t.Errorf("Pod %d assigned to MD %d, expected MD 0", i, assignment.MD)
		}
		if slots[assignment.Slot] {
			t.Errorf("Pod %d assigned to slot %d, which is already occupied", i, assignment.Slot)
		}
		slots[assignment.Slot] = true
	}

	// Should use exactly 3 slots
	usedSlots := 0
	for _, used := range result.SlotsUsed[0] {
		if used {
			usedSlots++
		}
	}
	assert.Equal(t, 3, usedSlots, "Expected exactly 3 slots to be used")

	t.Logf("Anti-affinity test successful:")
	for i, assignment := range result.PodAssignments {
		t.Logf("  Pod %d → MD %d Slot %d", i, assignment.MD, assignment.Slot)
	}
}
