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

// LeastWastePlugin scores MDs by how closely they match aggregate resource demand.
// It penalizes overprovisioned combinations (i.e., high unused capacity).
type LeastWastePlugin struct{ weight float64 }

func (p *LeastWastePlugin) Name() string    { return "LeastWaste" }
func (p *LeastWastePlugin) Weight() float64 { return p.weight }
func (p *LeastWastePlugin) Score(md optimiser.MachineDeployment, stats PodStats) float64 {
	if stats.Count == 0 {
		return 0
	}
	nodesNeeded := math.Max(stats.TotalCPU/md.GetCPU(), stats.TotalMem/md.GetMemory())
	totalProvisionedCPU := math.Ceil(nodesNeeded) * md.GetCPU()
	totalProvisionedMem := math.Ceil(nodesNeeded) * md.GetMemory()

	wasteCPU := totalProvisionedCPU - stats.TotalCPU
	wasteMem := totalProvisionedMem - stats.TotalMem
	normWasteCPU := wasteCPU / totalProvisionedCPU
	normWasteMem := wasteMem / totalProvisionedMem
	wasteRatio := (normWasteCPU + normWasteMem) / 2

	return 1.0 - wasteRatio
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

// main executes the full optimisation flow with example data.
// This includes plugin-based scoring, constraint generation, warm start, and final solver execution.
func main() {
	mds := generateMDs()
	pods := generatePods()

	plugins := []ScoringPlugin{
		&FewestNodesPlugin{weight: 0.2},
		&LeastWastePlugin{weight: 1.0},
		&RegexMatchPlugin{weight: 0.1, pattern: regexp.MustCompile(`.*-c7i$`)},
	}

	stats := calculatePodStats(pods)
	scores := computeScores(mds, stats, plugins)

	allowedMatrix, _ := computeAllowedMatrix(pods, mds)
	initial := computeInitialAssignments(pods, mds, scores)

	start := time.Now()
	runtime := 15
	result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowedMatrix, initial, &runtime)
	duration := time.Since(start)

	fmt.Printf("\nResult: %s\nStatus Code: %d\nObjective: %.2f\nSolve Time: %.2fs\nDuration (Go): %s\n",
		result.Message, result.SolverStatus, result.Objective, result.SolveTimeSecs, duration)

	if !result.Succeeded {
		fmt.Println("\nNo solution found by optimiser.")
		return
	}

	totalNodes := 0
	for _, slots := range result.SlotsUsed {
		replicas := 0
		for _, used := range slots {
			if used {
				replicas++
			}
		}
		if replicas > 0 {
			totalNodes += replicas
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

	// Report total resource waste for post-analysis or plugin tuning
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

		wasteCPU := provisionedCPU - mdCPUUsed[i]
		wasteMem := provisionedMem - mdMemUsed[i]

		totalWasteCPU += wasteCPU
		totalWasteMem += wasteMem
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
