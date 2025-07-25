// optimiser.go
//go:generate make build

package optimiser

/*
#cgo darwin LDFLAGS: -ldl
#cgo linux LDFLAGS: -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef RTLD_DEFAULT
#define RTLD_DEFAULT ((void *) 0)
#endif

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

// C wrapper function that dynamically calls OptimisePlacement
SolverResult call_optimise_placement_with_handle(
    void* lib_handle,
    const MachineDeployment* mds, int num_mds,
    const Pod* pods, int num_pods,
    const double* plugin_scores,
    const int* allowed_matrix,
    const int* initial_assignment,
    int* out_slot_assignments,
    uint8_t* out_slots_used,
    const int* max_runtime_secs
) {
    // Get the function pointer using dlsym with the specific handle
    void* sym = dlsym(lib_handle, "OptimisePlacement");
    if (!sym) {
        // Try with underscore prefix (macOS)
        sym = dlsym(lib_handle, "_OptimisePlacement");
        if (!sym) {
            SolverResult err_result = {0, 0.0, -1, 0.0};
            return err_result;
        }
    }

    // Cast to the correct function pointer type
    typedef SolverResult (*OptimisePlacementFunc)(
        const MachineDeployment*, int,
        const Pod*, int,
        const double*,
        const int*,
        const int*,
        int*,
        uint8_t*,
        const int*
    );

    OptimisePlacementFunc func = (OptimisePlacementFunc)sym;

    // Call the function
    return func(mds, num_mds, pods, num_pods, plugin_scores,
                allowed_matrix, initial_assignment, out_slot_assignments,
                out_slots_used, max_runtime_secs);
}
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

type PodSlotAssignment struct {
	MD   int
	Slot int
}

type Result struct {
	PodAssignments []PodSlotAssignment
	SlotsUsed      [][]bool
	Succeeded      bool
	Objective      float64
	SolverStatus   int
	SolveTimeSecs  float64
	Message        string
}

func makeCIntSlice(data []int) *C.int {
	if len(data) == 0 {
		return nil
	}
	ptr := (*C.int)(C.malloc(C.size_t(len(data)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	goSlice := (*[1 << 30]C.int)(unsafe.Pointer(ptr))[:len(data):len(data)]
	for i, v := range data {
		goSlice[i] = C.int(v)
	}
	return ptr
}

func makeCDoubleSlice(data []float64) *C.double {
	if len(data) == 0 {
		return nil
	}
	ptr := (*C.double)(C.malloc(C.size_t(len(data)) * C.size_t(unsafe.Sizeof(C.double(0)))))
	goSlice := (*[1 << 30]C.double)(unsafe.Pointer(ptr))[:len(data):len(data)]
	for i, v := range data {
		goSlice[i] = C.double(v)
	}
	return ptr
}

func OptimisePlacementRaw(
	mds []MachineDeployment,
	pods []Pod,
	pluginScores []float64,
	allowedMatrix []int,
	initialAssignment []int,
	maxRuntimeSeconds *int,
) Result {
	if err := extractAndLoadSharedLibrary(); err != nil {
		return Result{Message: fmt.Sprintf("Failed to load optimiser lib: %v", err)}
	}

	numMDs := len(mds)
	slotsPerMD := make([]int, numMDs)
	for i, md := range mds {
		slotsPerMD[i] = md.GetMaxScaleOut()
	}

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
			cPeers = makeCIntSlice(peers)
			cRules = makeCIntSlice(rules)
			hardPtrs = append(hardPtrs, unsafe.Pointer(cPeers), unsafe.Pointer(cRules))
		}

		var cSoftPeers *C.int
		var cSoftWeights *C.double
		if len(softPeers) > 0 {
			cSoftPeers = makeCIntSlice(softPeers)
			cSoftWeights = makeCDoubleSlice(softWeights)
			softPtrs = append(softPtrs, unsafe.Pointer(cSoftPeers), unsafe.Pointer(cSoftWeights))
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

	// Defatul to 15s
	if maxRuntimeSeconds == nil {
		defaultRuntime := 15
		maxRuntimeSeconds = &defaultRuntime
	}

	cMaxRuntime := (*C.int)(C.malloc(C.size_t(unsafe.Sizeof(C.int(0)))))
	*cMaxRuntime = C.int(*maxRuntimeSeconds)
	defer C.free(unsafe.Pointer(cMaxRuntime))

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

	outAssign := (*C.int)(C.malloc(C.size_t(numPods*2) * C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(outAssign))

	outSlotsUsedCount := 0
	for _, n := range slotsPerMD {
		outSlotsUsedCount += n
	}
	outSlots := (*C.uint8_t)(C.malloc(C.size_t(outSlotsUsedCount)))
	defer C.free(unsafe.Pointer(outSlots))

	res := C.call_optimise_placement_with_handle(
		GetLibHandle(),
		(*C.MachineDeployment)(unsafe.Pointer(&cMDs[0])), C.int(numMDs),
		(*C.Pod)(unsafe.Pointer(&cPods[0])), C.int(numPods),
		cScores, cAllowed, cHints, outAssign, outSlots, cMaxRuntime,
	)

	rawAssign := (*[1 << 30]C.int)(unsafe.Pointer(outAssign))[: numPods*2 : numPods*2]
	assignments := make([]PodSlotAssignment, numPods)
	for i := range numPods {
		assignments[i] = PodSlotAssignment{
			MD:   int(rawAssign[i*2]),
			Slot: int(rawAssign[i*2+1]),
		}
	}

	rawSlots := (*[1 << 30]C.uint8_t)(unsafe.Pointer(outSlots))[:outSlotsUsedCount:outSlotsUsedCount]
	slotsUsed := make([][]bool, numMDs)
	offset := 0
	for j := range numMDs {
		slotsUsed[j] = make([]bool, slotsPerMD[j])
		for k := 0; k < slotsPerMD[j]; k++ {
			slotsUsed[j][k] = rawSlots[offset] != 0
			offset++
		}
	}

	result := Result{
		Succeeded:      bool(res.success),
		Objective:      float64(res.objective),
		SolverStatus:   int(res.status_code),
		SolveTimeSecs:  float64(res.solve_time_secs),
		PodAssignments: assignments,
		SlotsUsed:      slotsUsed,
		Message:        "",
	}

	if !result.Succeeded {
		result.Message = fmt.Sprintf("Solver failed with status code %d", result.SolverStatus)
		return result
	}

	result.Message = "Optimisation successful"
	return result
}
