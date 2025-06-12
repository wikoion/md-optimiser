//go:generate make -C ..

package main

/*
#cgo LDFLAGS: -L. -loptimiser
#include <stdlib.h>

typedef struct {
    const char* name;
    double cpu;
    double memory;
    int max_scale_out;
} MachineDeployment;

typedef struct {
    double cpu;
    double memory;
} Pod;

void OptimisePlacement(
    const MachineDeployment* mds, int num_mds,
    const Pod* pods, int num_pods,
    const double* plugin_scores,
    const int* allowed_matrix,
    const int* initial_assignment,
    int* out_assignments,
    int* out_nodes_used
);
*/
import "C"

import (
	"math"
	"regexp"
	"sort"
	"strings"
	"unsafe"
)

type MachineDeployment struct {
	Name        string
	CPU         float64
	Memory      float64
	MaxScaleOut int
}

type Pod struct {
	CPU    float64
	Memory float64
	Labels map[string]string
}

type ScoringPlugin interface {
	Name() string
	Weight() float64
	Score(md MachineDeployment, pods []Pod) float64
}

type RegexMatchPlugin struct {
	weight  float64
	pattern *regexp.Regexp
}

func (p *RegexMatchPlugin) Name() string    { return "RegexMatch" }
func (p *RegexMatchPlugin) Weight() float64 { return p.weight }
func (p *RegexMatchPlugin) Score(md MachineDeployment, _ []Pod) float64 {
	if p.pattern.MatchString(md.Name) {
		return 1.0
	}
	return 0.0
}

type FewestNodesPlugin struct{ weight float64 }

func (p *FewestNodesPlugin) Name() string                                   { return "FewestNodes" }
func (p *FewestNodesPlugin) Weight() float64                                { return p.weight }
func (p *FewestNodesPlugin) Score(md MachineDeployment, pods []Pod) float64 { return 1.0 }

type LeastWastePlugin struct{ weight float64 }

func (p *LeastWastePlugin) Name() string    { return "LeastWaste" }
func (p *LeastWastePlugin) Weight() float64 { return p.weight }
func (p *LeastWastePlugin) Score(md MachineDeployment, pods []Pod) float64 {
	var totalCPU, totalMem float64
	for _, pod := range pods {
		totalCPU += pod.CPU
		totalMem += pod.Memory
	}
	wasteCPU := math.Max(0, md.CPU-totalCPU)
	wasteMem := math.Max(0, md.Memory-totalMem)
	return (wasteCPU + wasteMem) / (md.CPU + md.Memory)
}

func getAllowedMDs(pod Pod, mds []MachineDeployment) []int {
	var allowed []int
	for i, md := range mds {
		if pod.CPU > md.CPU || pod.Memory > md.Memory {
			continue
		}
		if pod.Labels["workload-type"] == "nvme" {
			if strings.Contains(md.Name, "m6id") {
				allowed = append(allowed, i)
			}
			continue
		}
		allowed = append(allowed, i)
	}
	return allowed
}

func contains(slice []int, v int) bool {
	for _, x := range slice {
		if x == v {
			return true
		}
	}
	return false
}

func Optimise(mds []MachineDeployment, pods []Pod, plugins []ScoringPlugin) ([]int, []int) {
	numMDs := len(mds)
	numPods := len(pods)

	scores := make([]float64, numMDs)
	for i, md := range mds {
		for _, plugin := range plugins {
			scores[i] += plugin.Weight() * plugin.Score(md, pods)
		}
	}

	allowed := make([][]bool, len(pods))
	for i, pod := range pods {
		allowed[i] = make([]bool, len(mds))
		allowedMDs := getAllowedMDs(pod, mds)
		for _, j := range allowedMDs {
			allowed[i][j] = true
		}
	}

	flat := make([]C.int, len(pods)*len(mds))
	for i := range pods {
		for j := range mds {
			if allowed[i][j] {
				flat[i*len(mds)+j] = 1
			} else {
				flat[i*len(mds)+j] = 0
			}
		}
	}

	cMDs := make([]C.MachineDeployment, numMDs)
	cNames := make([]*C.char, numMDs)
	for i, md := range mds {
		cNames[i] = C.CString(md.Name)
		cMDs[i] = C.MachineDeployment{
			name:          cNames[i],
			cpu:           C.double(md.CPU),
			memory:        C.double(md.Memory),
			max_scale_out: C.int(md.MaxScaleOut),
		}
	}

	cPods := make([]C.Pod, numPods)
	for i, p := range pods {
		cPods[i] = C.Pod{cpu: C.double(p.CPU), memory: C.double(p.Memory)}
	}

	cAllowed := (*C.int)(C.malloc(C.size_t(len(flat)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cAllowed))
	goAllowed := (*[1 << 30]C.int)(unsafe.Pointer(cAllowed))[:len(flat):len(flat)]
	copy(goAllowed, flat)

	cScores := (*C.double)(C.malloc(C.size_t(numMDs) * C.size_t(unsafe.Sizeof(C.double(0)))))
	defer C.free(unsafe.Pointer(cScores))
	goScores := (*[1 << 30]C.double)(unsafe.Pointer(cScores))[:numMDs:numMDs]
	for i := range scores {
		goScores[i] = C.double(scores[i])
	}

	outAssign := (*C.int)(C.malloc(C.size_t(numPods) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(outAssign))
	outNodes := (*C.int)(C.malloc(C.size_t(numMDs) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(outNodes))

	type podWithIndex struct {
		index int
		cpu   float64
		mem   float64
	}
	var sortedPods []podWithIndex
	for i, p := range pods {
		sortedPods = append(sortedPods, podWithIndex{i, p.CPU, p.Memory})
	}
	sort.Slice(sortedPods, func(i, j int) bool {
		return (sortedPods[i].cpu + sortedPods[i].mem) > (sortedPods[j].cpu + sortedPods[j].mem)
	})

	type mdWithScore struct {
		index int
		score float64
	}
	var mdScores []mdWithScore
	for i, score := range scores {
		mdScores = append(mdScores, mdWithScore{i, score})
	}
	sort.Slice(mdScores, func(i, j int) bool {
		return mdScores[i].score > mdScores[j].score
	})

	initialAssignment := make([]int, len(pods))
	mdCPU := make([]float64, len(mds))
	mdMem := make([]float64, len(mds))

	for _, podInfo := range sortedPods {
		i := podInfo.index
		pod := pods[i]
		assigned := false
		allowedMDs := getAllowedMDs(pod, mds)
		for _, md := range mdScores {
			j := md.index
			if !contains(allowedMDs, j) {
				continue
			}
			if mdCPU[j]+pod.CPU <= float64(mds[j].MaxScaleOut)*mds[j].CPU &&
				mdMem[j]+pod.Memory <= float64(mds[j].MaxScaleOut)*mds[j].Memory {
				initialAssignment[i] = j
				mdCPU[j] += pod.CPU
				mdMem[j] += pod.Memory
				assigned = true
				break
			}
		}
		if !assigned && len(allowedMDs) > 0 {
			initialAssignment[i] = allowedMDs[0]
		}
	}

	cHints := (*C.int)(C.malloc(C.size_t(len(initialAssignment)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cHints))
	goHints := (*[1 << 30]C.int)(unsafe.Pointer(cHints))[:len(initialAssignment):len(initialAssignment)]
	for i, h := range initialAssignment {
		goHints[i] = C.int(h)
	}

	C.OptimisePlacement(
		(*C.MachineDeployment)(unsafe.Pointer(&cMDs[0])), C.int(numMDs),
		(*C.Pod)(unsafe.Pointer(&cPods[0])), C.int(numPods),
		cScores,
		cAllowed,
		cHints,
		outAssign,
		outNodes,
	)

	assignments := make([]int, numPods)
	nodesUsed := make([]int, numMDs)
	goAssign := (*[1 << 30]C.int)(unsafe.Pointer(outAssign))[:numPods:numPods]
	goNodes := (*[1 << 30]C.int)(unsafe.Pointer(outNodes))[:numMDs:numMDs]
	for i := 0; i < numPods; i++ {
		assignments[i] = int(goAssign[i])
	}
	for i := 0; i < numMDs; i++ {
		nodesUsed[i] = int(goNodes[i])
	}

	for _, cstr := range cNames {
		C.free(unsafe.Pointer(cstr))
	}

	return assignments, nodesUsed
}
