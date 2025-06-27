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

    int colocation_preference;   // soft preference: 1 colocate, -1 spread, 0 none
    double colocation_weight;    // how strongly to apply the soft preference
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

// Defines a machine deployment in Go
type MachineDeployment interface {
	GetName() string
	GetCPU() float64
	GetMemory() float64
	GetMaxScaleOut() int
}

// Represents a pod in Go
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
	GetColocationPreference() int
	GetColocationWeight() float64
}

// Result returned from the optimisation function
type Result struct {
	Assignments   []int   // For each pod, the index of its assigned deployment
	NodesUsed     []int   // For each deployment, number of nodes used
	Succeeded     bool    // True if solution was found
	Objective     float64 // Objective value from the solver
	SolverStatus  int     // Status code from the solver
	SolveTimeSecs float64 // Time the solver took in seconds
	Message       string  // Human-readable summary
}

// ------------------------
// Main wrapper function: converts Go structs into C-compatible values,
// invokes the C++ solver, and converts results back to Go
// ------------------------

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

	// ---------------------------------------
	// Convert Go MachineDeployments to C
	// ---------------------------------------
	cMDs := make([]C.MachineDeployment, numMDs)
	cNames := make([]*C.char, numMDs) // hold allocated C strings so we can free them later
	for i, md := range mds {
		cNames[i] = C.CString(md.GetName()) // convert Go string to *C.char
		cMDs[i] = C.MachineDeployment{
			name:          cNames[i],
			cpu:           C.double(md.GetCPU()),
			memory:        C.double(md.GetMemory()),
			max_scale_out: C.int(md.GetMaxScaleOut()),
		}
	}
	// Clean up all allocated strings when done
	defer func() {
		for _, cstr := range cNames {
			C.free(unsafe.Pointer(cstr))
		}
	}()

	// ---------------------------------------
	// Convert Go Pods to C
	// ---------------------------------------
	cPods := make([]C.Pod, numPods)

	// Track any allocated C arrays for freeing later
	affinityArrays := []unsafe.Pointer{}
	defer func() {
		for _, ptr := range affinityArrays {
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
				panic(fmt.Sprintf("pod %d: affinity_peers and affinity_rules length mismatch", i))
			}
		}

		var colocPref int
		var colocWeight float64
		if sa, ok := p.(HasSoftAffinity); ok {
			colocPref = sa.GetColocationPreference()
			colocWeight = sa.GetColocationWeight()
		}

		// Marshal affinity arrays (if present)
		var cPeers, cRules *C.int
		if len(peers) > 0 {
			cPeers = (*C.int)(C.malloc(C.size_t(len(peers)) * C.size_t(unsafe.Sizeof(C.int(0)))))
			cRules = (*C.int)(C.malloc(C.size_t(len(rules)) * C.size_t(unsafe.Sizeof(C.int(0)))))
			affinityArrays = append(affinityArrays, unsafe.Pointer(cPeers), unsafe.Pointer(cRules))

			goPeers := (*[1 << 30]C.int)(unsafe.Pointer(cPeers))[:len(peers):len(peers)]
			goRules := (*[1 << 30]C.int)(unsafe.Pointer(cRules))[:len(rules):len(rules)]
			for j := range peers {
				goPeers[j] = C.int(peers[j])
				goRules[j] = C.int(rules[j])
			}
		}

		cPods[i] = C.Pod{
			cpu:    C.double(base.GetCPU()),
			memory: C.double(base.GetMemory()),

			affinity_peers: cPeers,
			affinity_rules: cRules,
			affinity_count: C.int(len(peers)),

			colocation_preference: C.int(colocPref),
			colocation_weight:     C.double(colocWeight),
		}
	}

	// ---------------------------------------
	// Convert pluginScores to C array
	// ---------------------------------------
	cScores := (*C.double)(C.malloc(C.size_t(len(pluginScores)) * C.size_t(unsafe.Sizeof(C.double(0)))))
	defer C.free(unsafe.Pointer(cScores))
	goScores := (*[1 << 30]C.double)(unsafe.Pointer(cScores))[:len(pluginScores):len(pluginScores)]
	for i, s := range pluginScores {
		goScores[i] = C.double(s)
	}

	// ---------------------------------------
	// Convert allowedMatrix (pod x deployment permissions) to C array
	// ---------------------------------------
	cAllowed := (*C.int)(C.malloc(C.size_t(len(allowedMatrix)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cAllowed))
	goAllowed := (*[1 << 30]C.int)(unsafe.Pointer(cAllowed))[:len(allowedMatrix):len(allowedMatrix)]
	for i, val := range allowedMatrix {
		goAllowed[i] = C.int(val)
	}

	// ---------------------------------------
	// Convert initial assignment hint to C array
	// ---------------------------------------
	cHints := (*C.int)(C.malloc(C.size_t(len(initialAssignment)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(cHints))
	goHints := (*[1 << 30]C.int)(unsafe.Pointer(cHints))[:len(initialAssignment):len(initialAssignment)]
	for i, h := range initialAssignment {
		goHints[i] = C.int(h)
	}

	// ---------------------------------------
	// Allocate memory for solver outputs
	// ---------------------------------------
	outAssign := (*C.int)(C.malloc(C.size_t(numPods) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(outAssign))
	outNodes := (*C.int)(C.malloc(C.size_t(numMDs) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(outNodes))

	// ---------------------------------------
	// Call the actual C++ solver function
	// ---------------------------------------
	res := C.OptimisePlacement(
		(*C.MachineDeployment)(unsafe.Pointer(&cMDs[0])), C.int(numMDs),
		(*C.Pod)(unsafe.Pointer(&cPods[0])), C.int(numPods),
		cScores, cAllowed, cHints, outAssign, outNodes,
	)

	// ---------------------------------------
	// Unpack the solver result into Go
	// ---------------------------------------
	result := Result{
		Succeeded:     bool(res.success),
		Objective:     float64(res.objective),
		SolverStatus:  int(res.status_code),
		SolveTimeSecs: float64(res.solve_time_secs),
		Assignments:   make([]int, numPods),
		NodesUsed:     make([]int, numMDs),
		Message:       "",
	}

	// Solver gave up or failed
	if !result.Succeeded {
		result.Message = fmt.Sprintf("Solver failed with status code %d", result.SolverStatus)
		return result
	}

	// Extract C array results back into Go slices
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
