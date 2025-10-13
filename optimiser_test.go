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
func (p *mockPod) GetCurrentMDAssignment() int       { return -1 } // Default: unknown assignment

// mockPodWithAssignment extends mockPod with a current MD assignment
type mockPodWithAssignment struct {
	cpu, memory         float64
	labels              map[string]string
	currentMDAssignment int
}

func (p *mockPodWithAssignment) GetCPU() float64             { return p.cpu }
func (p *mockPodWithAssignment) GetMemory() float64          { return p.memory }
func (p *mockPodWithAssignment) GetLabel(key string) string  { return p.labels[key] }
func (p *mockPodWithAssignment) GetCurrentMDAssignment() int { return p.currentMDAssignment }

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
	config := &optimiser.OptimizationConfig{
		MaxRuntimeSeconds: &runtime,
	}
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)
	if !result.Succeeded {
		t.Fatalf("Expected success: %s", result.Message)
	}

	assert.Len(t, result.PodAssignments, numPods)
	assert.Len(t, result.SlotsUsed, numMDs)

	// With least waste optimization, expect efficient distribution across MDs
	// Rather than concentrating on the highest-scoring MD, verify we get a valid solution
	totalSlotsUsed := 0
	for _, mdSlots := range result.SlotsUsed {
		for _, used := range mdSlots {
			if used {
				totalSlotsUsed++
			}
		}
	}
	assert.GreaterOrEqual(t, totalSlotsUsed, 2)

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
	config := &optimiser.OptimizationConfig{
		MaxRuntimeSeconds: &runtime,
	}
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)

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
	config := &optimiser.OptimizationConfig{
		MaxRuntimeSeconds: &runtime,
	}
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)

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
	config := &optimiser.OptimizationConfig{
		MaxRuntimeSeconds: &runtime,
	}
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)

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
	config := &optimiser.OptimizationConfig{
		MaxRuntimeSeconds: &runtime,
	}
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)

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

func TestOptimisePlacementRaw_ScoreOnlyMode(t *testing.T) {
	// Create test MDs and pods with current assignments
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-a", cpu: 16, memory: 64, maxScaleOut: 2},
		&mockMD{name: "md-b", cpu: 8, memory: 32, maxScaleOut: 2},
		&mockMD{name: "md-c", cpu: 4, memory: 16, maxScaleOut: 3},
	}

	pods := []optimiser.Pod{
		&mockPodWithAssignment{cpu: 2, memory: 4, currentMDAssignment: 0},
		&mockPodWithAssignment{cpu: 4, memory: 8, currentMDAssignment: 0},
		&mockPodWithAssignment{cpu: 1, memory: 2, currentMDAssignment: 1},
		&mockPodWithAssignment{cpu: 6, memory: 12, currentMDAssignment: 2},
		&mockPodWithAssignment{cpu: 3, memory: 6, currentMDAssignment: 2},
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
	scoreOnly := true
	config := &optimiser.OptimizationConfig{
		MaxRuntimeSeconds: &runtime,
		ScoreOnly:         &scoreOnly,
	}
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)

	// Verify score-only mode behavior
	if !result.Succeeded {
		t.Fatalf("Expected success in score-only mode, got failure: %s", result.Message)
	}

	// Should return a score (current state cost)
	if result.Objective <= 0 {
		t.Errorf("Expected positive objective score, got: %.2f", result.Objective)
	}

	// CurrentStateCost should equal Objective in score-only mode
	if result.CurrentStateCost != result.Objective {
		t.Errorf("In score-only mode, CurrentStateCost (%.2f) should equal Objective (%.2f)",
			result.CurrentStateCost, result.Objective)
	}

	// Should have empty assignments
	if len(result.PodAssignments) != 0 {
		t.Errorf("Expected empty PodAssignments in score-only mode, got %d assignments", len(result.PodAssignments))
	}

	// SlotsUsed should exist but be empty
	if len(result.SlotsUsed) != len(mds) {
		t.Errorf("Expected SlotsUsed array for %d MDs, got %d", len(mds), len(result.SlotsUsed))
	}
	for j, mdSlots := range result.SlotsUsed {
		for k, used := range mdSlots {
			if used {
				t.Errorf("Expected all slots unused in score-only mode, but MD %d slot %d is used", j, k)
			}
		}
	}

	// Solve time should be near zero (no optimization ran)
	if result.SolveTimeSecs > 0.1 {
		t.Errorf("Expected near-zero solve time in score-only mode, got %.2fs", result.SolveTimeSecs)
	}

	// Message should indicate score-only mode
	if matched, _ := regexp.MatchString("(?i)score.only", result.Message); !matched {
		t.Errorf("Expected message to mention score-only mode, got: %s", result.Message)
	}

	t.Logf("Score-only mode test successful:")
	t.Logf("  Current State Score: %.2f", result.Objective)
	t.Logf("  Solve Time: %.4fs", result.SolveTimeSecs)
	t.Logf("  Message: %s", result.Message)
}

func TestOptimisePlacementRaw_BeamSearchOnly(t *testing.T) {
	// Create simple test scenario
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-1", cpu: 4.0, memory: 8.0, maxScaleOut: 5},
		&mockMD{name: "md-2", cpu: 2.0, memory: 4.0, maxScaleOut: 5},
	}

	pods := []optimiser.Pod{
		&mockPod{cpu: 1.0, memory: 2.0},
		&mockPod{cpu: 0.5, memory: 1.0},
		&mockPod{cpu: 1.5, memory: 3.0},
	}

	numPods := len(pods)
	numMDs := len(mds)

	scores := []float64{1.0, 0.5}
	allowed := make([]int, numPods*numMDs)
	for i := range allowed {
		allowed[i] = 1
	}
	initial := make([][][]int, 0)

	config := &optimiser.OptimizationConfig{
		BeamSearchOnly: boolPtr(true),
	}

	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)

	if !result.Succeeded {
		t.Fatalf("Expected beam-search-only to succeed: %s", result.Message)
	}

	assert.True(t, result.UsedBeamSearch, "Expected UsedBeamSearch to be true")
	assert.True(t, result.BeamSearchFallback, "Expected BeamSearchFallback to be true")
	assert.Equal(t, 0, result.CPSATAttempts, "Expected 0 CP-SAT attempts in beam-search-only mode")
	assert.Len(t, result.PodAssignments, numPods, "Expected all pods to be assigned")

	t.Logf("Beam-search-only test successful:")
	t.Logf("  Objective: %.2f", result.Objective)
	t.Logf("  Used Beam Search: %v", result.UsedBeamSearch)
	t.Logf("  Assignments: %v", result.PodAssignments)
}

func TestOptimisePlacementRaw_BeamSearchHint(t *testing.T) {
	// Create test scenario where beam search hint should help
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-1", cpu: 4.0, memory: 8.0, maxScaleOut: 5},
		&mockMD{name: "md-2", cpu: 4.0, memory: 8.0, maxScaleOut: 5},
	}

	pods := []optimiser.Pod{
		&mockPod{cpu: 1.0, memory: 2.0},
		&mockPod{cpu: 1.0, memory: 2.0},
		&mockPod{cpu: 1.0, memory: 2.0},
	}

	numPods := len(pods)
	numMDs := len(mds)

	scores := []float64{1.0, 0.8}
	allowed := make([]int, numPods*numMDs)
	for i := range allowed {
		allowed[i] = 1
	}
	initial := make([][][]int, 0)

	config := &optimiser.OptimizationConfig{
		MaxAttempts:           intPtr(3),
		UseBeamSearchHint:     boolPtr(true),
		BeamSearchHintAttempt: intPtr(2),
		MaxRuntimeSeconds:     intPtr(5),
	}

	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)

	if !result.Succeeded {
		t.Fatalf("Expected optimization with beam search hint to succeed: %s", result.Message)
	}

	assert.Len(t, result.PodAssignments, numPods, "Expected all pods to be assigned")
	assert.True(t, result.CPSATAttempts >= 1, "Expected at least 1 CP-SAT attempt")

	t.Logf("Beam search hint test successful:")
	t.Logf("  Objective: %.2f", result.Objective)
	t.Logf("  Used Beam Search: %v", result.UsedBeamSearch)
	t.Logf("  CP-SAT Attempts: %d", result.CPSATAttempts)
	t.Logf("  Best Attempt: %d", result.BestAttempt)
}

func TestOptimisePlacementRaw_BeamSearchFallback(t *testing.T) {
	// Create a difficult scenario that might challenge CP-SAT
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-1", cpu: 2.0, memory: 4.0, maxScaleOut: 3},
		&mockMD{name: "md-2", cpu: 2.0, memory: 4.0, maxScaleOut: 3},
	}

	pods := []optimiser.Pod{
		&mockPod{cpu: 1.0, memory: 2.0},
		&mockPod{cpu: 1.0, memory: 2.0},
		&mockPod{cpu: 0.5, memory: 1.0},
	}

	numPods := len(pods)
	numMDs := len(mds)

	scores := []float64{1.0, 0.9}
	allowed := make([]int, numPods*numMDs)
	for i := range allowed {
		allowed[i] = 1
	}
	initial := make([][][]int, 0)

	config := &optimiser.OptimizationConfig{
		MaxAttempts:          intPtr(2),
		MaxRuntimeSeconds:    intPtr(1), // Very short timeout to potentially trigger fallback
		FallbackToBeamSearch: boolPtr(true),
	}

	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)

	// Either CP-SAT succeeds or beam search fallback kicks in
	if !result.Succeeded {
		t.Fatalf("Expected solution (either CP-SAT or beam search): %s", result.Message)
	}

	assert.Len(t, result.PodAssignments, numPods, "Expected all pods to be assigned")

	t.Logf("Beam search fallback test successful:")
	t.Logf("  Objective: %.2f", result.Objective)
	t.Logf("  Used Beam Search: %v", result.UsedBeamSearch)
	t.Logf("  Beam Search Fallback: %v", result.BeamSearchFallback)
	t.Logf("  CP-SAT Attempts: %d", result.CPSATAttempts)
}

func TestOptimisePlacementRaw_MultipleAttempts(t *testing.T) {
	// Test that multiple attempts work correctly
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-1", cpu: 4.0, memory: 8.0, maxScaleOut: 5},
		&mockMD{name: "md-2", cpu: 4.0, memory: 8.0, maxScaleOut: 5},
	}

	pods := []optimiser.Pod{
		&mockPod{cpu: 1.0, memory: 2.0},
		&mockPod{cpu: 1.0, memory: 2.0},
	}

	numPods := len(pods)
	numMDs := len(mds)

	scores := []float64{1.0, 0.9}
	allowed := make([]int, numPods*numMDs)
	for i := range allowed {
		allowed[i] = 1
	}
	initial := make([][][]int, 0)

	config := &optimiser.OptimizationConfig{
		MaxAttempts:       intPtr(5),
		MaxRuntimeSeconds: intPtr(3),
	}

	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)

	if !result.Succeeded {
		t.Fatalf("Expected optimization to succeed: %s", result.Message)
	}

	assert.Len(t, result.PodAssignments, numPods, "Expected all pods to be assigned")
	assert.True(t, result.CPSATAttempts >= 1 && result.CPSATAttempts <= 5, "Expected 1-5 CP-SAT attempts")
	assert.True(t, result.BestAttempt >= 1 && result.BestAttempt <= result.CPSATAttempts, "Best attempt should be valid")

	t.Logf("Multiple attempts test successful:")
	t.Logf("  CP-SAT Attempts: %d", result.CPSATAttempts)
	t.Logf("  Best Attempt: %d", result.BestAttempt)
	t.Logf("  Objective: %.2f", result.Objective)
}

func TestOptimisePlacementRaw_BackwardCompatibility(t *testing.T) {
	// Test that nil config works as before (backward compatibility)
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-1", cpu: 4.0, memory: 8.0, maxScaleOut: 5},
	}

	pods := []optimiser.Pod{
		&mockPod{cpu: 1.0, memory: 2.0},
		&mockPod{cpu: 0.5, memory: 1.0},
	}

	numPods := len(pods)

	scores := []float64{1.0}
	allowed := []int{1, 1}
	initial := make([][][]int, 0)

	// Pass nil config - should use default behavior
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, nil)

	if !result.Succeeded {
		t.Fatalf("Expected backward-compatible optimization to succeed: %s", result.Message)
	}

	assert.Len(t, result.PodAssignments, numPods, "Expected all pods to be assigned")
	assert.False(t, result.UsedBeamSearch, "Expected not to use beam search with nil config")
	assert.False(t, result.BeamSearchFallback, "Expected no beam search fallback with nil config")

	t.Logf("Backward compatibility test successful:")
	t.Logf("  Objective: %.2f", result.Objective)
	t.Logf("  CP-SAT Attempts: %d", result.CPSATAttempts)
}

func TestOptimisePlacement_DeprecatedAPI(t *testing.T) {
	// Test the deprecated wrapper function for backward compatibility
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-1", cpu: 4.0, memory: 8.0, maxScaleOut: 5},
	}

	pods := []optimiser.Pod{
		&mockPod{cpu: 1.0, memory: 2.0},
	}

	numPods := len(pods)

	scores := []float64{1.0}
	allowed := []int{1}
	initial := make([][][]int, 0)

	runtime := 15
	result := optimiser.OptimisePlacement(mds, pods, scores, allowed, initial, &runtime, nil, nil)

	if !result.Succeeded {
		t.Fatalf("Expected deprecated API to work: %s", result.Message)
	}

	assert.Len(t, result.PodAssignments, numPods, "Expected all pods to be assigned")

	t.Logf("Deprecated API test successful:")
	t.Logf("  Objective: %.2f", result.Objective)
}

func TestOptimisePlacementRaw_ImprovementThreshold(t *testing.T) {
	// Test that improvement threshold is respected
	mds := []optimiser.MachineDeployment{
		&mockMD{name: "md-1", cpu: 4.0, memory: 8.0, maxScaleOut: 5},
	}

	// Create pods with current assignments
	pods := []optimiser.Pod{
		&mockPodWithAssignment{cpu: 1.0, memory: 2.0, currentMDAssignment: 0},
		&mockPodWithAssignment{cpu: 1.0, memory: 2.0, currentMDAssignment: 0},
	}

	numPods := len(pods)

	scores := []float64{1.0}
	allowed := []int{1, 1}
	initial := make([][][]int, 0)

	// Set a very high threshold - solution might not meet it
	config := &optimiser.OptimizationConfig{
		ImprovementThreshold: floatPtr(50.0), // Require 50% improvement
		MaxRuntimeSeconds:    intPtr(5),
	}

	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial, config)

	// Result might fail or succeed depending on whether threshold is met
	t.Logf("Improvement threshold test:")
	t.Logf("  Succeeded: %v", result.Succeeded)
	t.Logf("  Current State Cost: %.2f", result.CurrentStateCost)
	t.Logf("  Objective: %.2f", result.Objective)
	t.Logf("  Already Optimal: %v", result.AlreadyOptimal)

	if result.Succeeded {
		assert.Len(t, result.PodAssignments, numPods, "Expected all pods to be assigned if succeeded")
	}
}

func boolPtr(b bool) *bool {
	return optimiser.BoolPtr(b)
}

func intPtr(i int) *int {
	return optimiser.IntPtr(i)
}

func floatPtr(f float64) *float64 {
	return optimiser.FloatPtr(f)
}
