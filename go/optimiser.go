// optimiser.go
//go:generate make -C ..

package md_solver

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

typedef struct {
    _Bool success;
    double objective;
    int status_code;
    double solve_time_secs;
} SolverResult;

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
	numMDs := len(mds)
	numPods := len(pods)

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
	defer func() {
		for _, cstr := range cNames {
			C.free(unsafe.Pointer(cstr))
		}
	}()

	cPods := make([]C.Pod, numPods)
	for i, p := range pods {
		cPods[i] = C.Pod{cpu: C.double(p.CPU), memory: C.double(p.Memory)}
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
