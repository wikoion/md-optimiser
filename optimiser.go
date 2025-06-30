// optimiser.go
//go:generate make build

package optimiser

/*
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath,${SRCDIR}/lib -loptimiser_darwin_arm64
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib -Wl,-rpath,${SRCDIR}/lib -loptimiser
#include <dlfcn.h>
#include <stdlib.h>

// C struct definition mirroring the ones in optimiser.cpp
typedef struct {
    const char* name;    // Deployment name (e.g., "c5.large")
    double cpu;          // Per-node CPU
    double memory;       // Per-node memory
    int max_scale_out;   // Max nodes allowed
} MachineDeployment;

typedef struct {
    double cpu;          // Pod CPU request
    double memory;       // Pod memory request

    const int* affinity_peers;   // index list of pods this pod references
    const int* affinity_rules;   // 1-to-1 rules (1 colocate, -1 anti, 0 none)
    int affinity_count;

    const int* soft_peers;        // soft affinity peer indices
    const double* soft_affinities;// weights (positive = colocate, negative = spread)
    int soft_affinity_count;
} Pod;

typedef struct {
    _Bool success;          // Whether a valid solution was found
    double objective;       // Objective value of the solution
    int status_code;        // Solver status (e.g., 2 = feasible)
    double solve_time_secs; // Time solver spent in seconds
} SolverResult;

// Declaration of external C function we will call from Go
SolverResult OptimisePlacement(
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
	"fmt"
	"unsafe"
)

// ------------------------
// Public Go equivalents of the C structs
// ------------------------

type MachineDeployment interface {
	GetName() string
	GetCPU() float64
	GetMemory() float64
	GetMaxScaleOut() int
}

type Pod interface {
	GetCPU() float64
	GetMemory() float64
	GetLabel(string) string
}

type HasHardAffinity interface {
	GetAffinityPeers() []int
	GetAffinityRules() []int
}

type HasSoftAffinity interface {
	GetSoftAffinityPeers() []int
	GetSoftAffinityWeights() []float64
}

type Result struct {
	Assignments   []int
	NodesUsed     []int
	Succeeded     bool
	Objective     float64
	SolverStatus  int
	SolveTimeSecs float64
	Message       string
}

func OptimisePlacementRaw(
	mds []MachineDeployment,
	pods []Pod,
	pluginScores []float64,
	allowedMatrix []int,
	initialAssignment []int,
) Result {
	if err := extractAndLoadSharedLibrary(); err != nil {
		return Result{Message: fmt.Sprintf("Failed to load optimiser lib: %v", err)}
	}

	numMDs := len(mds)
	numPods := len(pods)

	cMDs := make([]C.MachineDeployment, numMDs)
	cNames := make([]*C.char, numMDs)
	for i, md := range mds {
		cNames[i] = C.CString(md.GetName())
		cMDs[i] = C.MachineDeployment{
			name:          cNames[i],
			cpu:           C.double(md.GetCPU()),
			memory:        C.double(md.GetMemory()),
			max_scale_out: C.int(md.GetMaxScaleOut()),
		}
	}
	defer func() {
		for _, cstr := range cNames {
			C.free(unsafe.Pointer(cstr))
		}
	}()

	cPods := make([]C.Pod, numPods)
	hardPtrs := []unsafe.Pointer{}
	softPtrs := []unsafe.Pointer{}
	defer func() {
		for _, ptr := range hardPtrs {
			C.free(ptr)
		}
		for _, ptr := range softPtrs {
			C.free(ptr)
		}
	}()

	for i, p := range pods {
		base := p

		var peers, rules []int
		if ha, ok := p.(HasHardAffinity); ok {
			peers = ha.GetAffinityPeers()
			rules = ha.GetAffinityRules()
			if len(peers) != len(rules) {
				panic(fmt.Sprintf("pod %d: affinity_peers and affinity_rules mismatch", i))
			}
		}

		var softPeers []int
		var softWeights []float64
		if sa, ok := p.(HasSoftAffinity); ok {
			softPeers = sa.GetSoftAffinityPeers()
			softWeights = sa.GetSoftAffinityWeights()
			if len(softPeers) != len(softWeights) {
				panic(fmt.Sprintf("pod %d: soft affinity peers and weights mismatch", i))
			}
		}

		var cPeers, cRules *C.int
		if len(peers) > 0 {
			cPeers = (*C.int)(C.malloc(C.size_t(len(peers)) * C.size_t(unsafe.Sizeof(C.int(0)))))
			cRules = (*C.int)(C.malloc(C.size_t(len(rules)) * C.size_t(unsafe.Sizeof(C.int(0)))))
			hardPtrs = append(hardPtrs, unsafe.Pointer(cPeers), unsafe.Pointer(cRules))
			goPeers := (*[1 << 30]C.int)(unsafe.Pointer(cPeers))[:len(peers):len(peers)]
			goRules := (*[1 << 30]C.int)(unsafe.Pointer(cRules))[:len(rules):len(rules)]
			for j := range peers {
				goPeers[j] = C.int(peers[j])
				goRules[j] = C.int(rules[j])
			}
		}

		var cSoftPeers *C.int
		var cSoftWeights *C.double
		if len(softPeers) > 0 {
			cSoftPeers = (*C.int)(C.malloc(C.size_t(len(softPeers)) * C.size_t(unsafe.Sizeof(C.int(0)))))
			cSoftWeights = (*C.double)(C.malloc(C.size_t(len(softWeights)) * C.size_t(unsafe.Sizeof(C.double(0)))))
			softPtrs = append(softPtrs, unsafe.Pointer(cSoftPeers), unsafe.Pointer(cSoftWeights))
			goSoftPeers := (*[1 << 30]C.int)(unsafe.Pointer(cSoftPeers))[:len(softPeers):len(softPeers)]
			goSoftWeights := (*[1 << 30]C.double)(unsafe.Pointer(cSoftWeights))[:len(softWeights):len(softWeights)]
			for j := range softPeers {
				goSoftPeers[j] = C.int(softPeers[j])
				goSoftWeights[j] = C.double(softWeights[j])
			}
		}

		cPods[i] = C.Pod{
			cpu:                 C.double(base.GetCPU()),
			memory:              C.double(base.GetMemory()),
			affinity_peers:      cPeers,
			affinity_rules:      cRules,
			affinity_count:      C.int(len(peers)),
			soft_peers:          cSoftPeers,
			soft_affinities:     cSoftWeights,
			soft_affinity_count: C.int(len(softPeers)),
		}
	}

	cScores := (*C.double)(C.malloc(C.size_t(len(pluginScores)) * C.size_t(unsafe.Sizeof(C.double(0)))))
	defer C.free(unsafe.Pointer(cScores))
	goScores := (*[1 << 30]C.double)(unsafe.Pointer(cScores))[:len(pluginScores):len(pluginScores)]
	for i, s := range pluginScores {
		goScores[i] = C.double(s)
	}

	cAllowed := (*C.int)(C.malloc(C.size_t(len(allowedMatrix)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cAllowed))
	goAllowed := (*[1 << 30]C.int)(unsafe.Pointer(cAllowed))[:len(allowedMatrix):len(allowedMatrix)]
	for i, val := range allowedMatrix {
		goAllowed[i] = C.int(val)
	}

	cHints := (*C.int)(C.malloc(C.size_t(len(initialAssignment)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cHints))
	goHints := (*[1 << 30]C.int)(unsafe.Pointer(cHints))[:len(initialAssignment):len(initialAssignment)]
	for i, h := range initialAssignment {
		goHints[i] = C.int(h)
	}

	outAssign := (*C.int)(C.malloc(C.size_t(numPods) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(outAssign))
	outNodes := (*C.int)(C.malloc(C.size_t(numMDs) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(outNodes))

	res := C.OptimisePlacement(
		(*C.MachineDeployment)(unsafe.Pointer(&cMDs[0])), C.int(numMDs),
		(*C.Pod)(unsafe.Pointer(&cPods[0])), C.int(numPods),
		cScores, cAllowed, cHints, outAssign, outNodes,
	)

	result := Result{
		Succeeded:     bool(res.success),
		Objective:     float64(res.objective),
		SolverStatus:  int(res.status_code),
		SolveTimeSecs: float64(res.solve_time_secs),
		Assignments:   make([]int, numPods),
		NodesUsed:     make([]int, numMDs),
		Message:       "",
	}

	if !result.Succeeded {
		result.Message = fmt.Sprintf("Solver failed with status code %d", result.SolverStatus)
		return result
	}

	goAssign := (*[1 << 30]C.int)(unsafe.Pointer(outAssign))[:numPods:numPods]
	goNodes := (*[1 << 30]C.int)(unsafe.Pointer(outNodes))[:numMDs:numMDs]
	for i := 0; i < numPods; i++ {
		result.Assignments[i] = int(goAssign[i])
	}
	for i := 0; i < numMDs; i++ {
		result.NodesUsed[i] = int(goNodes[i])
	}

	result.Message = "Optimisation successful"
	return result
}
