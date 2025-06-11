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
    int* out_assignments,
    int* out_nodes_used
);
*/
import "C"

import (
	"math"
	"regexp"
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
	} else {
		return 0.0
	}
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

func Optimise(mds []MachineDeployment, pods []Pod, plugins []ScoringPlugin) ([]int, []int) {
	numMDs := len(mds)
	numPods := len(pods)

	scores := make([]float64, numMDs)
	for i, md := range mds {
		for _, plugin := range plugins {
			scores[i] += plugin.Weight() * plugin.Score(md, pods)
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

	C.OptimisePlacement(
		(*C.MachineDeployment)(unsafe.Pointer(&cMDs[0])), C.int(numMDs),
		(*C.Pod)(unsafe.Pointer(&cPods[0])), C.int(numPods),
		cScores,
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
