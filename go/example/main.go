// example/main.go
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

type Pod struct {
	CPU    float64
	Memory float64
	Labels map[string]string
}

type ScoringPlugin interface {
	Name() string
	Weight() float64
	Score(md md_solver.MachineDeployment, pods []Pod) float64
}

type FewestNodesPlugin struct{ weight float64 }

func (p *FewestNodesPlugin) Name() string    { return "FewestNodes" }
func (p *FewestNodesPlugin) Weight() float64 { return p.weight }
func (p *FewestNodesPlugin) Score(_ md_solver.MachineDeployment, _ []Pod) float64 {
	return 1.0
}

type LeastWastePlugin struct{ weight float64 }

func (p *LeastWastePlugin) Name() string    { return "LeastWaste" }
func (p *LeastWastePlugin) Weight() float64 { return p.weight }
func (p *LeastWastePlugin) Score(md md_solver.MachineDeployment, pods []Pod) float64 {
	var cpuSum, memSum float64
	for _, pod := range pods {
		cpuSum += pod.CPU
		memSum += pod.Memory
	}
	waste := math.Max(0, md.CPU-cpuSum) + math.Max(0, md.Memory-memSum)
	return waste / (md.CPU + md.Memory)
}

type RegexMatchPlugin struct {
	weight  float64
	pattern *regexp.Regexp
}

func (p *RegexMatchPlugin) Name() string    { return "RegexMatch" }
func (p *RegexMatchPlugin) Weight() float64 { return p.weight }
func (p *RegexMatchPlugin) Score(md md_solver.MachineDeployment, _ []Pod) float64 {
	if p.pattern.MatchString(md.Name) {
		return 1.0
	}
	return 0.0
}

func main() {
	mds := generateMDs()
	pods := generatePods()

	plugins := []ScoringPlugin{
		&FewestNodesPlugin{weight: 0.5},
		&LeastWastePlugin{weight: 0.5},
		&RegexMatchPlugin{weight: 0.9, pattern: regexp.MustCompile(`.*-c7i$`)},
	}

	scores := computeScores(mds, pods, plugins)
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

	fmt.Println("\nAssignments:")
	for i, j := range result.Assignments {
		fmt.Printf("Pod %d → %s\n", i, mds[j].Name)
	}

	fmt.Println("\nNodes used:")
	for i, n := range result.NodesUsed {
		if n > 0 {
			fmt.Printf("%s → %d nodes\n", mds[i].Name, n)
		}
	}
}

func generateMDs() []md_solver.MachineDeployment {
	var mds []md_solver.MachineDeployment
	baseTypes := []string{"c7i", "general", "burst", "balanced", "mixed", "lowcost", "m6id"}
	for i := 0; i < 50; i++ {
		typeName := baseTypes[i%len(baseTypes)]
		name := fmt.Sprintf("md-%02d-%s", i, typeName)
		mds = append(mds, md_solver.MachineDeployment{
			Name:        name,
			CPU:         16 + float64((i%5)*8),
			Memory:      64 + float64((i%4)*32),
			MaxScaleOut: 3 + (i % 5),
		})
	}
	return mds
}

func generatePods() []Pod {
	pods := []Pod{{CPU: 5, Memory: 11, Labels: map[string]string{"workload-type": "nvme"}}}
	for i := 0; i < 200; i++ {
		pods = append(pods, Pod{
			CPU:    float64(2 + (i % 4)),
			Memory: float64(2 + ((i * 3) % 10)),
		})
	}
	return pods
}

func computeScores(mds []md_solver.MachineDeployment, pods []Pod, plugins []ScoringPlugin) []float64 {
	s := make([]float64, len(mds))
	for i, md := range mds {
		for _, p := range plugins {
			s[i] += p.Weight() * p.Score(md, pods)
		}
	}
	return s
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
				ok = strings.Contains(md.Name, "m6id")
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

	podOrder := make([]podIdx, numPods)
	for i, p := range pods {
		podOrder[i] = podIdx{i, p.CPU, p.Memory}
	}
	sort.Slice(podOrder, func(i, j int) bool {
		return podOrder[i].c+podOrder[i].m > podOrder[j].c+podOrder[j].m
	})

	mdOrder := make([]mdScore, len(mds))
	for i, s := range scores {
		mdOrder[i] = mdScore{i, s}
	}
	sort.Slice(mdOrder, func(i, j int) bool {
		return mdOrder[i].s > mdOrder[j].s
	})

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
