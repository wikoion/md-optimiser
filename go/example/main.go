// ========================================================================================
// MD Solver
// ========================================================================================
// This Go programme acts as a frontend to a custom OR-Tools-based C++ solver for scheduling
// pods (workloads) onto MachineDeployments (instance types). It demonstrates:
//
//   - Pluggable scoring via a flexible plugin interface
//   - Deterministic workload and instance pool generation for benchmarking
//   - Conversion of domain objects (Pods, MachineDeployments) to flat C-friendly structs
//   - Scoring normalisation to influence the solver's weighted objective
//   - Optional initial assignment hints to accelerate solving
//   - Solver post-processing including detailed waste calculation
//
// ========================================================================================
// Usage Notes:
//
// • Plugins:
//   - Plugins compute per-MD scores; higher is better.
//   - These are linearly combined and normalised to affect C++ penalty weights.
//   - If all plugin scores are nearly equal, their influence on optimisation will be minimal.
//   - Example plugins included: FewestNodes, LeastWaste, RegexMatch.
//
// • Hints:
//   - A greedy first-fit bin-packing algorithm is used to create an initial assignment.
//   - This hint is optional but improves convergence and solution quality.
//
// • Constraints:
//   - Resource feasibility and label-based filtering are enforced in `computeAllowedMatrix()`.
//   - e.g., NVMe pods are restricted to m6id deployments only.
//
// • Waste Metrics:
//   - After solving, total unutilised CPU and memory across all MDs are reported.
//   - Helps evaluate trade-offs between cost, resource efficiency, and plugin behaviour.
// ========================================================================================

package main

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"time"

	"md_solver"
)

// Pod represents a container workload with resource demands and labels.
// Labels are used for placement constraints (e.g., requiring a certain hardware class).
type Pod struct {
	CPU    float64           // CPU cores requested
	Memory float64           // Memory in GiB requested
	Labels map[string]string // e.g. {"workload-type": "nvme"}
}

// ScoringPlugin allows custom pluggable scoring of MDs for a given workload profile.
// Higher scores indicate more desirable MDs; weights indicate relative importance.
type ScoringPlugin interface {
	Name() string
	Weight() float64
	Score(md md_solver.MachineDeployment, stats PodStats) float64
}

// PodStats provides summary statistics across all pods to avoid repeated aggregation.
// Plugins use this instead of raw pods for performance.
type PodStats struct {
	TotalCPU, TotalMem float64
	AvgCPU, AvgMem     float64
	Count              int
}

// calculatePodStats returns totals and averages from a slice of pods.
// If no pods exist, zeroes are returned and downstream scoring plugins will fallback.
func calculatePodStats(pods []Pod) PodStats {
	var totalCPU, totalMem float64
	for _, p := range pods {
		totalCPU += p.CPU
		totalMem += p.Memory
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

// FewestNodesPlugin scores MDs based on how many average pods fit per node.
// A high score means fewer nodes are likely needed, favoring dense packing.
// Caveat: May lead to more wasted resources if pod shapes mismatch MD shapes.
type FewestNodesPlugin struct{ weight float64 }

func (p *FewestNodesPlugin) Name() string    { return "FewestNodes" }
func (p *FewestNodesPlugin) Weight() float64 { return p.weight }

// Estimate based on real total demand
func (p *FewestNodesPlugin) Score(md md_solver.MachineDeployment, stats PodStats) float64 {
	if stats.Count == 0 {
		return 0
	}
	nodesNeeded := math.Max(stats.TotalCPU/md.CPU, stats.TotalMem/md.Memory)
	return 1.0 / nodesNeeded // Lower nodes needed = higher score
}

// LeastWastePlugin favors MDs that closely match aggregate resource demand.
// Waste is calculated as unused provisioned resources. Higher score = less waste.
// This can conflict with FewestNodes if minimizing waste means spreading over more nodes.
type LeastWastePlugin struct{ weight float64 }

func (p *LeastWastePlugin) Name() string    { return "LeastWaste" }
func (p *LeastWastePlugin) Weight() float64 { return p.weight }
func (p *LeastWastePlugin) Score(md md_solver.MachineDeployment, stats PodStats) float64 {
	if stats.Count == 0 {
		return 0
	}
	nodesNeeded := math.Max(stats.TotalCPU/md.CPU, stats.TotalMem/md.Memory)
	totalProvisionedCPU := math.Ceil(nodesNeeded) * md.CPU
	totalProvisionedMem := math.Ceil(nodesNeeded) * md.Memory

	wasteCPU := totalProvisionedCPU - stats.TotalCPU
	wasteMem := totalProvisionedMem - stats.TotalMem
	wasteRatio := (wasteCPU + wasteMem) / (totalProvisionedCPU + totalProvisionedMem)

	return 1.0 - wasteRatio
}

// RegexMatchPlugin scores MDs based on regex name match. Used for expressing preferences
// for instance families (e.g., favoring c7i nodes). Should be used cautiously in general-purpose packing.
type RegexMatchPlugin struct {
	weight  float64
	pattern *regexp.Regexp
}

func (p *RegexMatchPlugin) Name() string    { return "RegexMatch" }
func (p *RegexMatchPlugin) Weight() float64 { return p.weight }
func (p *RegexMatchPlugin) Score(md md_solver.MachineDeployment, _ PodStats) float64 {
	if p.pattern.MatchString(md.Name) {
		return 1.0
	}
	return 0.0
}

func generateMDs() []md_solver.MachineDeployment {
	// Generates a fixed set of 50 MDs with diverse shapes to expose solver tradeoffs.
	// Includes both CPU-heavy, memory-heavy, and balanced shapes.
	var mds []md_solver.MachineDeployment
	shapes := []struct {
		cpu   float64
		mem   float64
		scale int
	}{
		{16, 64, 3},  // balanced small
		{24, 96, 4},  // balanced medium
		{32, 128, 5}, // balanced large
		{48, 96, 5},  // CPU-heavy
		{24, 192, 5}, // memory-heavy
		{64, 256, 7}, // high-end
		{16, 32, 3},  // very CPU-heavy
		{8, 128, 3},  // very memory-heavy
	}
	baseTypes := []string{"c7i", "general", "burst", "balanced", "mixed", "lowcost", "m6id"}
	for i := 0; i < 50; i++ {
		shape := shapes[i%len(shapes)]
		typeName := baseTypes[i%len(baseTypes)]
		name := fmt.Sprintf("md-%02d-%s", i, typeName)
		mds = append(mds, md_solver.MachineDeployment{
			Name:        name,
			CPU:         shape.cpu,
			Memory:      shape.mem,
			MaxScaleOut: shape.scale,
		})
	}
	return mds
}

func generatePods() []Pod {
	// Generates a reproducible set of pods with a variety of shapes to increase packing ambiguity.
	// This promotes tradeoffs between waste and density when optimising.
	pods := []Pod{{CPU: 5, Memory: 11, Labels: map[string]string{"workload-type": "nvme"}}}
	shapes := []struct {
		cpu float64
		mem float64
	}{
		{2, 4},  // light
		{1, 8},  // mem-heavy
		{4, 2},  // cpu-heavy
		{2, 8},  // balanced
		{3, 6},  // medium
		{6, 12}, // medium-large
		{8, 16}, // heavy
		{1, 16}, // very mem-heavy
	}
	for i := 0; i < 200; i++ {
		shape := shapes[i%len(shapes)]
		pods = append(pods, Pod{CPU: shape.cpu, Memory: shape.mem})
	}
	return pods
}

func computeScores(mds []md_solver.MachineDeployment, stats PodStats, plugins []ScoringPlugin) []float64 {
	s := make([]float64, len(mds))
	for i, md := range mds {
		for _, p := range plugins {
			s[i] += p.Weight() * p.Score(md, stats)
		}
	}
	return normalizeScores(s)
}

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
		return make([]float64, len(scores)) // Avoid flat normalization
	}
	out := make([]float64, len(scores))
	for i, v := range scores {
		norm := (v - min) / (max - min)
		out[i] = math.Pow(norm, 3) // Cubic boost — spreads the tail
	}
	return out
}

func computeAllowedMatrix(pods []Pod, mds []md_solver.MachineDeployment) ([]int, [][]int) {
	numPods := len(pods)
	numMDs := len(mds)
	flat := make([]int, numPods*numMDs)
	allowed := make([][]int, numPods)
	for i, pod := range pods {
		for j, md := range mds {
			ok := pod.CPU <= md.CPU && pod.Memory <= md.Memory
			if pod.Labels["workload-type"] == "nvme" {
				ok = strings.Contains(md.Name, "m6id") // Label constraint
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

func computeInitialAssignments(pods []Pod, mds []md_solver.MachineDeployment, allowed [][]int, scores []float64) []int {
	type podIdx struct {
		i    int
		c, m float64
	}
	type mdScore struct {
		i int
		s float64
	}

	numPods := len(pods)
	initial := make([]int, numPods)
	mdCPU := make([]float64, len(mds))
	mdMem := make([]float64, len(mds))

	// Sort pods by descending demand
	podOrder := make([]podIdx, numPods)
	for i, p := range pods {
		podOrder[i] = podIdx{i, p.CPU, p.Memory}
	}
	sort.Slice(podOrder, func(i, j int) bool {
		return podOrder[i].c+podOrder[i].m > podOrder[j].c+podOrder[j].m
	})

	// Sort MDs by descending plugin score
	mdOrder := make([]mdScore, len(mds))
	for i, s := range scores {
		mdOrder[i] = mdScore{i, s}
	}
	sort.Slice(mdOrder, func(i, j int) bool {
		return mdOrder[i].s > mdOrder[j].s
	})

	// Greedy placement respecting capacity limits
	for _, p := range podOrder {
		for _, md := range mdOrder {
			if !contains(allowed[p.i], md.i) {
				continue
			}
			if mdCPU[md.i]+p.c <= float64(mds[md.i].MaxScaleOut)*mds[md.i].CPU &&
				mdMem[md.i]+p.m <= float64(mds[md.i].MaxScaleOut)*mds[md.i].Memory {
				initial[p.i] = md.i
				mdCPU[md.i] += p.c
				mdMem[md.i] += p.m
				break
			}
		}
	}
	return initial
}

func convertPods(pods []Pod) []md_solver.Pod {
	out := make([]md_solver.Pod, len(pods))
	for i, p := range pods {
		out[i] = md_solver.Pod{CPU: p.CPU, Memory: p.Memory}
	}
	return out
}

func contains(slice []int, val int) bool {
	for _, x := range slice {
		if x == val {
			return true
		}
	}
	return false
}

func main() {
	mds := generateMDs()
	pods := generatePods()

	plugins := []ScoringPlugin{
		&FewestNodesPlugin{weight: 0.3},
		&LeastWastePlugin{weight: 0.7},
		&RegexMatchPlugin{weight: 0.1, pattern: regexp.MustCompile(`.*-c7i$`)},
	}

	stats := calculatePodStats(pods)
	scores := computeScores(mds, stats, plugins)

	allowedMatrix, allowedMDs := computeAllowedMatrix(pods, mds)
	initial := computeInitialAssignments(pods, mds, allowedMDs, scores)

	start := time.Now()
	result := md_solver.OptimisePlacementRaw(mds, convertPods(pods), scores, allowedMatrix, initial)
	duration := time.Since(start)

	fmt.Printf("\nResult: %s\nStatus Code: %d\nObjective: %.2f\nSolve Time: %.2fs\nDuration (Go): %s\n",
		result.Message, result.SolverStatus, result.Objective, result.SolveTimeSecs, duration)

	if !result.Succeeded {
		fmt.Println("\nNo solution found by optimiser.")
		return
	}

	totalNodes := 0
	for _, n := range result.NodesUsed {
		if n > 0 {
			totalNodes += n
		}
	}

	fmt.Printf("\nTotal Nodes: %d, Nodes used:\n", totalNodes)
	for i, n := range result.NodesUsed {
		if n > 0 {
			fmt.Printf("%s → %d nodes\n", mds[i].Name, n)
		}
	}

	// Compute and print total resource waste (unused provisioned capacity)
	var totalWasteCPU, totalWasteMem float64
	mdCPUUsed := make([]float64, len(mds))
	mdMemUsed := make([]float64, len(mds))

	for i, podIdx := range result.Assignments {
		p := pods[i]
		mdCPUUsed[podIdx] += p.CPU
		mdMemUsed[podIdx] += p.Memory
	}

	for i, used := range result.NodesUsed {
		if used == 0 {
			continue
		}
		provisionedCPU := float64(used) * mds[i].CPU
		provisionedMem := float64(used) * mds[i].Memory

		wasteCPU := provisionedCPU - mdCPUUsed[i]
		wasteMem := provisionedMem - mdMemUsed[i]

		totalWasteCPU += wasteCPU
		totalWasteMem += wasteMem
	}

	fmt.Printf("\nTotal Waste:\n  CPU:    %.2f cores\n  Memory: %.2f GiB\n", totalWasteCPU, totalWasteMem)
}
