package main

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"time"

	optimiser "github.com/wikoion/md-optimiser"
)

type MD struct {
	Name        string
	CPU         float64
	Memory      float64
	MaxScaleOut int
}

func (md *MD) GetName() string {
	return md.Name
}

func (md *MD) GetCPU() float64 {
	return md.CPU
}

func (md *MD) GetMemory() float64 {
	return md.Memory
}

func (md *MD) GetMaxScaleOut() int {
	return md.MaxScaleOut
}

func (md *MD) GetOriginalReplicas() (int64, bool) {
	// For the example, we don't track current replicas, so return false
	// In a real implementation, this would return the actual number of nodes currently running
	return 0, false
}

// Pod represents a container workload with resource demands and labels.
// Labels may be used to apply placement constraints like hardware preferences.
type Pod struct {
	CPU, Memory         float64
	Labels              map[string]string
	AffinityPeers       []int
	AffinityRules       []int
	SoftAffinityPeers   []int
	SoftAffinityWeights []float64
}

func (p *Pod) GetAffinityPeers() []int           { return p.AffinityPeers }
func (p *Pod) GetAffinityRules() []int           { return p.AffinityRules }
func (p *Pod) GetSoftAffinityPeers() []int       { return p.SoftAffinityPeers }
func (p *Pod) GetSoftAffinityWeights() []float64 { return p.SoftAffinityWeights }

func (p *Pod) GetCPU() float64 {
	return p.CPU
}

func (p *Pod) GetMemory() float64 {
	return p.Memory
}

func (p *Pod) GetLabel(label string) string {
	l, ok := p.Labels[label]
	if ok {
		return l
	}
	return ""
}

func (p *Pod) GetCurrentMDAssignment() int {
	return -1 // Default: unknown assignment for examples
}

// ScoringPlugin defines an interface for pluggable MD scoring logic.
// Plugins return a score per MD and are weighted to express relative priority.
type ScoringPlugin interface {
	Name() string
	Weight() float64
	Score(md optimiser.MachineDeployment, stats PodStats) float64
}

// PodStats summarizes the demand characteristics of a workload set.
// Plugins consume this instead of individual pods for efficiency.
type PodStats struct {
	TotalCPU, TotalMem float64
	AvgCPU, AvgMem     float64
	Count              int
}

// calculatePodStats computes the total and average resource demand for a set of pods.
// This is used to inform scoring plugins.
func calculatePodStats(pods []optimiser.Pod) PodStats {
	var totalCPU, totalMem float64
	for _, p := range pods {
		totalCPU += p.GetCPU()
		totalMem += p.GetMemory()
	}
	count := len(pods)
	if count == 0 {
		return PodStats{}
	}
	return PodStats{
		TotalCPU: totalCPU,
		TotalMem: totalMem,
		AvgCPU:   totalCPU / float64(count),
		AvgMem:   totalMem / float64(count),
		Count:    count,
	}
}

// FewestNodesPlugin scores MDs that require fewer nodes to satisfy total demand.
// It promotes dense packing but may increase fragmentation/waste.
type FewestNodesPlugin struct{ weight float64 }

func (p *FewestNodesPlugin) Name() string    { return "FewestNodes" }
func (p *FewestNodesPlugin) Weight() float64 { return p.weight }
func (p *FewestNodesPlugin) Score(md optimiser.MachineDeployment, stats PodStats) float64 {
	if stats.Count == 0 {
		return 0
	}
	nodesNeeded := math.Max(stats.TotalCPU/md.GetCPU(), stats.TotalMem/md.GetMemory())
	return 1.0 / nodesNeeded
}

// RegexMatchPlugin scores MDs that match a name pattern.
// This can be used to express preferences (e.g., certain instance families).
// Should be used sparingly to avoid overfitting scoring.
type RegexMatchPlugin struct {
	weight  float64
	pattern *regexp.Regexp
}

func (p *RegexMatchPlugin) Name() string    { return "RegexMatch" }
func (p *RegexMatchPlugin) Weight() float64 { return p.weight }
func (p *RegexMatchPlugin) Score(md optimiser.MachineDeployment, _ PodStats) float64 {
	if p.pattern.MatchString(md.GetName()) {
		return 1.0
	}
	return 0.0
}

// generateMDs returns a representative set of test MachineDeployments.
// Shapes vary to create tradeoffs in waste, density, and label compatibility.
func generateMDs() []optimiser.MachineDeployment {
	var mds []optimiser.MachineDeployment
	shapes := []struct {
		cpu   float64
		mem   float64
		scale int
	}{
		{16, 64, 3}, {24, 96, 4}, {32, 128, 5},
		{48, 96, 5}, {24, 192, 5}, {64, 256, 7},
		{16, 32, 3}, {8, 128, 3},
	}
	baseTypes := []string{"c7i", "general", "burst", "balanced", "mixed", "lowcost", "m6id"}
	for i := 0; i < 50; i++ {
		shape := shapes[i%len(shapes)]
		typeName := baseTypes[i%len(baseTypes)]
		name := fmt.Sprintf("md-%02d-%s", i, typeName)
		mds = append(mds, &MD{
			Name:        name,
			CPU:         shape.cpu,
			Memory:      shape.mem,
			MaxScaleOut: shape.scale,
		})
	}
	return mds
}

// generatePods returns a synthetic workload with a mix of pod shapes and optional constraints.
func generatePods() []optimiser.Pod {
	var pods []optimiser.Pod

	// Pod 0 hard colocates with Pod 1
	pods = append(pods, &Pod{
		CPU: 2, Memory: 4,
		AffinityPeers:       []int{1},
		AffinityRules:       []int{1}, // hard colocate
		SoftAffinityPeers:   []int{1},
		SoftAffinityWeights: []float64{0.8}, // prefer colocate
	})

	// Pod 1 prefers spreading from Pod 0
	pods = append(pods, &Pod{
		CPU: 2, Memory: 4,
		AffinityPeers:       []int{},
		AffinityRules:       []int{},
		SoftAffinityPeers:   []int{0},
		SoftAffinityWeights: []float64{-0.5}, // prefer spread
	})

	// Additional generic pods
	shapes := []struct {
		cpu float64
		mem float64
	}{{2, 4}, {1, 8}, {4, 2}, {2, 8}, {3, 6}, {6, 12}, {8, 16}, {1, 16}}
	for i := 0; i < 100; i++ {
		shape := shapes[i%len(shapes)]
		pods = append(pods, &Pod{CPU: shape.cpu, Memory: shape.mem})
	}
	return pods
}

// computeScores applies all scoring plugins to each MD and returns a normalized score array.
// Plugins are assumed to be pure functions and should not have side effects.
func computeScores(mds []optimiser.MachineDeployment, stats PodStats, plugins []ScoringPlugin) []float64 {
	s := make([]float64, len(mds))
	for i, md := range mds {
		for _, p := range plugins {
			s[i] += p.Weight() * p.Score(md, stats)
		}
	}
	return normalizeScores(s)
}

// normalizeScores scales the score values into [0,1] range and sharpens contrast using a power curve.
// This makes high-signal MDs more distinguishable in greedy strategies or heuristic sorting.
func normalizeScores(scores []float64) []float64 {
	min, max := scores[0], scores[0]
	for _, v := range scores {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	if math.Abs(max-min) < 1e-6 {
		return make([]float64, len(scores)) // Avoid divide-by-zero when all scores equal
	}
	out := make([]float64, len(scores))
	for i, v := range scores {
		norm := (v - min) / (max - min)
		out[i] = norm * norm * norm // Boost high scores further
	}
	return out
}

// computeAllowedMatrix builds a binary constraint matrix for pod-to-MD compatibility.
// A pod is allowed to run on an MD if its resource requests fit and label rules pass.
func computeAllowedMatrix(pods []optimiser.Pod, mds []optimiser.MachineDeployment) ([]int, [][]int) {
	numPods := len(pods)
	numMDs := len(mds)
	flat := make([]int, numPods*numMDs) // Linearized matrix for solver
	allowed := make([][]int, numPods)   // Per-pod allowed MD indexes

	for i, pod := range pods {
		for j, md := range mds {
			ok := pod.GetCPU() <= md.GetCPU() && pod.GetMemory() <= md.GetMemory()
			if pod.GetLabel("workload-type") == "nvme" {
				ok = ok && strings.Contains(md.GetName(), "m6id")
			}
			idx := i*numMDs + j
			if ok {
				flat[idx] = 1
				allowed[i] = append(allowed[i], j)
			}
		}
	}
	return flat, allowed
}

// computeInitialAssignments generates a greedy seed assignment based on plugin scores.
// Used to warm-start the SAT solver and speed up convergence.
// calculateWaste calculates CPU and memory waste for a given result
func calculateWaste(
	result optimiser.Result,
	mds []optimiser.MachineDeployment,
	pods []optimiser.Pod,
) (float64, float64, int) {
	if !result.Succeeded {
		return 0, 0, 0
	}

	// Calculate total provisioned and used resources per MD
	type MDResources struct {
		provisionedCPU float64
		provisionedMem float64
		usedCPU        float64
		usedMem        float64
	}

	mdResources := make(map[int]*MDResources)
	totalNodes := 0

	// Count nodes and provisioned resources
	for mdIdx, slotsUsed := range result.SlotsUsed {
		nodesUsed := 0
		for _, used := range slotsUsed {
			if used {
				nodesUsed++
			}
		}
		if nodesUsed > 0 {
			totalNodes += nodesUsed
			mdResources[mdIdx] = &MDResources{
				provisionedCPU: float64(nodesUsed) * mds[mdIdx].GetCPU(),
				provisionedMem: float64(nodesUsed) * mds[mdIdx].GetMemory(),
			}
		}
	}

	// Calculate used resources
	for podIdx, assignment := range result.PodAssignments {
		if assignment.MD >= 0 {
			if res, ok := mdResources[assignment.MD]; ok {
				res.usedCPU += pods[podIdx].GetCPU()
				res.usedMem += pods[podIdx].GetMemory()
			}
		}
	}

	// Calculate total waste
	var totalWasteCPU, totalWasteMem float64
	for _, res := range mdResources {
		totalWasteCPU += res.provisionedCPU - res.usedCPU
		totalWasteMem += res.provisionedMem - res.usedMem
	}

	return totalWasteCPU, totalWasteMem, totalNodes
}

func computeInitialAssignments(
	pods []optimiser.Pod,
	mds []optimiser.MachineDeployment,
	scores []float64,
) [][][]int {
	type podIdx struct {
		i    int
		c, m float64
	}
	type mdScore struct {
		i int
		s float64
	}

	numPods := len(pods)
	initial := make([][][]int, 0) // Return empty 3D matrix for no hints
	mdCPU := make([]float64, len(mds))
	mdMem := make([]float64, len(mds))

	// Sort pods by descending total demand (to place heavy pods first)
	podOrder := make([]podIdx, numPods)
	for i, p := range pods {
		podOrder[i] = podIdx{i, p.GetCPU(), p.GetMemory()}
	}
	sort.Slice(podOrder, func(i, j int) bool {
		return podOrder[i].c+podOrder[i].m > podOrder[j].c+podOrder[j].m
	})

	// Sort MDs by descending score
	mdOrder := make([]mdScore, len(mds))
	for i, s := range scores {
		mdOrder[i] = mdScore{i, s}
	}
	sort.Slice(mdOrder, func(i, j int) bool {
		return mdOrder[i].s > mdOrder[j].s
	})

	// For this example, we'll return an empty 3D matrix (no hints)
	// In practice, you could build a 3D matrix here based on the greedy assignment
	_ = podOrder
	_ = mdOrder
	_ = mdCPU
	_ = mdMem
	return initial
}

// evaluateCurrentState runs score-only mode to evaluate the current state
func evaluateCurrentState(
	mds []optimiser.MachineDeployment,
	pods []optimiser.Pod,
	scores []float64,
	allowedMatrix []int,
	initial [][][]int,
) {
	fmt.Println("\n=== Score-Only Mode: Evaluating Current State ===")
	scoreStart := time.Now()
	scoreConfig := &optimiser.OptimizationConfig{
		ScoreOnly: optimiser.BoolPtr(true),
	}
	scoreResult := optimiser.OptimisePlacementRaw(mds, pods, scores, allowedMatrix, initial, scoreConfig)
	scoreDuration := time.Since(scoreStart)

	if scoreResult.Succeeded {
		fmt.Printf("Current State Score: %.2f\n", scoreResult.Objective)
		fmt.Printf("Evaluation Time: %s\n", scoreDuration)
	} else {
		fmt.Printf("Score-only mode failed: %s\n", scoreResult.Message)
	}
}

// strategyResults holds results from all optimization strategies
type strategyResults struct {
	beamSearchOnly         optimiser.Result
	beamSearchOnlyDuration time.Duration
	beam                   optimiser.Result
	beamDuration           time.Duration
	cpsat                  optimiser.Result
	cpsatDuration          time.Duration
	hybrid                 optimiser.Result
	hybridDuration         time.Duration
}

// runAllStrategies executes all optimization strategies and returns their results
func runAllStrategies(
	mds []optimiser.MachineDeployment,
	pods []optimiser.Pod,
	scores []float64,
	allowedMatrix []int,
) strategyResults {
	var results strategyResults

	// Strategy 1: Beam-Search-Only
	fmt.Println("\n[1] Beam-Search-Only Mode (Fast Heuristic)")
	fmt.Println(strings.Repeat("-", 80))
	start := time.Now()
	config := &optimiser.OptimizationConfig{
		BeamSearchOnly: optimiser.BoolPtr(true),
	}
	results.beamSearchOnly = optimiser.OptimisePlacementRaw(mds, pods, scores, allowedMatrix, nil, config)
	results.beamSearchOnlyDuration = time.Since(start)
	printBeamSearchOnlyResult(results.beamSearchOnly, results.beamSearchOnlyDuration, mds, pods)

	// Strategy 1b: Beam Search Enhanced
	fmt.Println("\n[1b] Beam Search Mode (Enhanced + Limited Backtracking)")
	fmt.Println(strings.Repeat("-", 80))
	start = time.Now()
	beamConfig := &optimiser.BeamSearchConfig{
		BeamWidth:    optimiser.IntPtr(3),
		MDCandidates: optimiser.IntPtr(3),
	}
	results.beam = optimiser.OptimisePlacementBeamSearch(mds, pods, scores, allowedMatrix, beamConfig)
	results.beamDuration = time.Since(start)
	printBeamResult(results.beam, results.beamDuration, results.beamSearchOnly, results.beamSearchOnlyDuration, mds, pods)

	// Strategy 2: CP-SAT Only
	fmt.Println("\n[2] CP-SAT Only Mode (Optimal Search)")
	fmt.Println(strings.Repeat("-", 80))
	start = time.Now()
	cpsatConfig := &optimiser.OptimizationConfig{
		MaxRuntimeSeconds: optimiser.IntPtr(30),
		MaxAttempts:       optimiser.IntPtr(1),
	}
	results.cpsat = optimiser.OptimisePlacementRaw(mds, pods, scores, allowedMatrix, nil, cpsatConfig)
	results.cpsatDuration = time.Since(start)
	printCPSATResult(results.cpsat, results.cpsatDuration, mds, pods)

	// Strategy 3: Hybrid
	fmt.Println("\n[3] CP-SAT with Greedy Warm-Start (Hybrid)")
	fmt.Println(strings.Repeat("-", 80))
	start = time.Now()
	hybridConfig := &optimiser.OptimizationConfig{
		MaxRuntimeSeconds:     optimiser.IntPtr(30),
		MaxAttempts:           optimiser.IntPtr(3),
		UseBeamSearchHint:     optimiser.BoolPtr(true),
		BeamSearchHintAttempt: optimiser.IntPtr(2),
		FallbackToBeamSearch:  optimiser.BoolPtr(true),
	}
	results.hybrid = optimiser.OptimisePlacementRaw(mds, pods, scores, allowedMatrix, nil, hybridConfig)
	results.hybridDuration = time.Since(start)
	printHybridResult(results.hybrid, results.hybridDuration, mds, pods)

	return results
}

func printBeamSearchOnlyResult(
	result optimiser.Result,
	duration time.Duration,
	mds []optimiser.MachineDeployment,
	pods []optimiser.Pod,
) {
	if result.Succeeded {
		wasteCPU, wasteMem, nodes := calculateWaste(result, mds, pods)
		fmt.Printf("✓ Success\n")
		fmt.Printf("  Objective:      %.2f\n", result.Objective)
		fmt.Printf("  Time:           %s\n", duration)
		fmt.Printf("  Used Beam Search: %v\n", result.UsedBeamSearch)
		fmt.Printf("  Unplaced Pods:  %d\n", result.UnplacedPods)
		fmt.Printf("  Nodes Used:     %d\n", nodes)
		fmt.Printf("  Waste:          %.2f CPU, %.2f GiB Memory\n", wasteCPU, wasteMem)
	} else {
		fmt.Printf("✗ Failed: %s\n", result.Message)
		fmt.Printf("  Unplaced Pods:  %d\n", result.UnplacedPods)
	}
}

func printBeamResult(
	result optimiser.Result,
	duration time.Duration,
	baselineResult optimiser.Result,
	baselineDuration time.Duration,
	mds []optimiser.MachineDeployment,
	pods []optimiser.Pod,
) {
	if result.Succeeded {
		wasteCPU, wasteMem, nodes := calculateWaste(result, mds, pods)
		fmt.Printf("✓ Success\n")
		fmt.Printf("  Objective:      %.2f\n", result.Objective)
		fmt.Printf("  Time:           %s\n", duration)
		fmt.Printf("  Beam Width:     3\n")
		fmt.Printf("  MD Candidates:  3\n")
		fmt.Printf("  Nodes Used:     %d\n", nodes)
		fmt.Printf("  Waste:          %.2f CPU, %.2f GiB Memory\n", wasteCPU, wasteMem)

		if baselineResult.Succeeded {
			improvement := ((baselineResult.Objective - result.Objective) / baselineResult.Objective) * 100.0
			speedRatio := float64(duration.Microseconds()) / float64(baselineDuration.Microseconds())
			fmt.Printf("  vs Beam-Only:   %.2fx slower, %.1f%% better objective\n", speedRatio, improvement)
		}
	} else {
		fmt.Printf("✗ Failed: %s\n", result.Message)
		fmt.Printf("  Unplaced Pods:  %d\n", result.UnplacedPods)
	}
}

func printCPSATResult(
	result optimiser.Result,
	duration time.Duration,
	mds []optimiser.MachineDeployment,
	pods []optimiser.Pod,
) {
	if result.Succeeded {
		wasteCPU, wasteMem, nodes := calculateWaste(result, mds, pods)
		fmt.Printf("✓ Success\n")
		fmt.Printf("  Objective:      %.2f\n", result.Objective)
		fmt.Printf("  Time:           %s\n", duration)
		fmt.Printf("  Solve Time:     %.2fs\n", result.SolveTimeSecs)
		fmt.Printf("  Status Code:    %d\n", result.SolverStatus)
		fmt.Printf("  CP-SAT Attempts: %d\n", result.CPSATAttempts)
		fmt.Printf("  Nodes Used:     %d\n", nodes)
		fmt.Printf("  Waste:          %.2f CPU, %.2f GiB Memory\n", wasteCPU, wasteMem)
	} else {
		fmt.Printf("✗ Failed: %s\n", result.Message)
	}
}

func printHybridResult(
	result optimiser.Result,
	duration time.Duration,
	mds []optimiser.MachineDeployment,
	pods []optimiser.Pod,
) {
	if result.Succeeded {
		wasteCPU, wasteMem, nodes := calculateWaste(result, mds, pods)
		fmt.Printf("✓ Success\n")
		fmt.Printf("  Objective:      %.2f\n", result.Objective)
		fmt.Printf("  Time:           %s\n", duration)
		fmt.Printf("  Solve Time:     %.2fs\n", result.SolveTimeSecs)
		fmt.Printf("  Used Beam Search: %v\n", result.UsedBeamSearch)
		fmt.Printf("  Beam Fallback:  %v\n", result.BeamSearchFallback)
		fmt.Printf("  CP-SAT Attempts: %d\n", result.CPSATAttempts)
		fmt.Printf("  Best Attempt:   %d\n", result.BestAttempt)
		fmt.Printf("  Nodes Used:     %d\n", nodes)
		fmt.Printf("  Waste:          %.2f CPU, %.2f GiB Memory\n", wasteCPU, wasteMem)
	} else {
		fmt.Printf("✗ Failed: %s\n", result.Message)
	}
}

// printSummary prints a summary comparison table of all strategies
func printSummary(results strategyResults) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("=== SUMMARY ===")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("\nStrategy Comparison:")
	fmt.Printf("%-25s | %-12s | %-15s | %-10s\n", "Strategy", "Objective", "Time", "Success")
	fmt.Println(strings.Repeat("-", 80))

	printSummaryRow("Beam-Search-Only", results.beamSearchOnly, results.beamSearchOnlyDuration)
	printSummaryRow("Beam Search Enhanced", results.beam, results.beamDuration)
	printSummaryRow("CP-SAT Only", results.cpsat, results.cpsatDuration)
	printSummaryRow("Hybrid (Greedy + CP-SAT)", results.hybrid, results.hybridDuration)
}

func printSummaryRow(name string, result optimiser.Result, duration time.Duration) {
	if result.Succeeded {
		fmt.Printf("%-25s | %12.2f | %15s | %-10s\n", name, result.Objective, duration, "✓")
	} else {
		fmt.Printf("%-25s | %12s | %15s | %-10s\n", name, "N/A", duration, "✗")
	}
}

// selectBestStrategy returns the best result and its name based on lowest objective
func selectBestStrategy(results strategyResults) (optimiser.Result, string) {
	candidates := []struct {
		result optimiser.Result
		name   string
	}{
		{results.beamSearchOnly, "Beam-Search-Only"},
		{results.beam, "Beam Search Enhanced"},
		{results.cpsat, "CP-SAT Only"},
		{results.hybrid, "Hybrid"},
	}

	bestObjective := float64(1e100)
	var bestResult optimiser.Result
	var bestName string

	for _, c := range candidates {
		if c.result.Succeeded && c.result.Objective < bestObjective {
			bestResult = c.result
			bestName = c.name
			bestObjective = c.result.Objective
		}
	}

	return bestResult, bestName
}

// printDetailedResults prints detailed information about the best strategy
func printDetailedResults(results strategyResults, mds []optimiser.MachineDeployment, pods []optimiser.Pod) {
	result, strategyName := selectBestStrategy(results)
	if strategyName == "" {
		fmt.Println("\n✗ No strategy produced a valid solution.")
		return
	}

	fmt.Printf("\n✓ Best strategy: %s (Objective: %.2f)\n", strategyName, result.Objective)
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Printf("=== DETAILED RESULTS: %s ===\n", strategyName)
	fmt.Println(strings.Repeat("=", 80))

	totalNodes := 0
	for _, slots := range result.SlotsUsed {
		for _, used := range slots {
			if used {
				totalNodes++
			}
		}
	}

	fmt.Printf("\nTotal Nodes: %d, Nodes used:\n", totalNodes)
	for i, slots := range result.SlotsUsed {
		replicas := 0
		for _, used := range slots {
			if used {
				replicas++
			}
		}
		if replicas > 0 {
			fmt.Printf("%s → %d nodes\n", mds[i].GetName(), replicas)
		}
	}

	var totalWasteCPU, totalWasteMem float64
	mdCPUUsed := make([]float64, len(mds))
	mdMemUsed := make([]float64, len(mds))

	for i, assignment := range result.PodAssignments {
		p := pods[i]
		mdCPUUsed[assignment.MD] += p.GetCPU()
		mdMemUsed[assignment.MD] += p.GetMemory()
	}

	for i, slots := range result.SlotsUsed {
		replicas := 0
		for _, used := range slots {
			if used {
				replicas++
			}
		}
		if replicas == 0 {
			continue
		}
		provisionedCPU := float64(replicas) * mds[i].GetCPU()
		provisionedMem := float64(replicas) * mds[i].GetMemory()
		totalWasteCPU += provisionedCPU - mdCPUUsed[i]
		totalWasteMem += provisionedMem - mdMemUsed[i]
	}

	fmt.Println("\nPod Assignments:")
	for i, assignment := range result.PodAssignments {
		p := pods[i]
		md := mds[assignment.MD]
		fmt.Printf("Pod %02d (CPU: %.1f, Mem: %.1f) → %s (slot %d)\n",
			i, p.GetCPU(), p.GetMemory(), md.GetName(), assignment.Slot)
	}

	fmt.Printf("\nTotal Waste:\n  CPU:    %.2f cores\n  Memory: %.2f GiB\n", totalWasteCPU, totalWasteMem)
}

// main executes the full optimisation flow with example data.
// This includes plugin-based scoring, constraint generation, warm start, and final solver execution.
func main() {
	mds := generateMDs()
	pods := generatePods()

	plugins := []ScoringPlugin{
		&FewestNodesPlugin{weight: 0.2},
		&RegexMatchPlugin{weight: 0.1, pattern: regexp.MustCompile(`.*-c7i$`)},
	}

	stats := calculatePodStats(pods)
	scores := computeScores(mds, stats, plugins)

	allowedMatrix, _ := computeAllowedMatrix(pods, mds)
	initial := computeInitialAssignments(pods, mds, scores)

	evaluateCurrentState(mds, pods, scores, allowedMatrix, initial)

	// Compare optimization strategies
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("=== COMPARISON: Greedy vs CP-SAT vs Greedy Warm-Start ===")
	fmt.Println(strings.Repeat("=", 80))

	results := runAllStrategies(mds, pods, scores, allowedMatrix)
	printSummary(results)
	printDetailedResults(results, mds, pods)
}
