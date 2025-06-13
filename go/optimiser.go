// optimiser.go
//go:generate make -C ..

package md_solver

/*
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
type MachineDeployment struct {
	Name        string
	CPU         float64
	Memory      float64
	MaxScaleOut int
}

// Represents a pod in Go
type Pod struct {
	CPU    float64
	Memory float64
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
	numMDs := len(mds)
	numPods := len(pods)

	// ---------------------------------------
	// Convert Go MachineDeployments to C
	// ---------------------------------------
	cMDs := make([]C.MachineDeployment, numMDs)
	cNames := make([]*C.char, numMDs) // hold allocated C strings so we can free them later
	for i, md := range mds {
		cNames[i] = C.CString(md.Name) // convert Go string to *C.char
		cMDs[i] = C.MachineDeployment{
			name:          cNames[i],
			cpu:           C.double(md.CPU),
			memory:        C.double(md.Memory),
			max_scale_out: C.int(md.MaxScaleOut),
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
	for i, p := range pods {
		cPods[i] = C.Pod{cpu: C.double(p.CPU), memory: C.double(p.Memory)}
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
