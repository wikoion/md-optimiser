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

    int current_md_assignment;   // Index of MD this pod is currently placed on (-1 if unknown)
} Pod;

typedef struct {
    _Bool success;          // Whether a valid solution was found
    double objective;       // Objective value of the solution
    int status_code;        // Solver status (e.g., 2 = feasible)
    double solve_time_secs; // Time solver spent in seconds
    double current_state_cost; // Cost of current state before optimization
    _Bool already_optimal;     // True if improvement doesn't meet threshold
    _Bool used_greedy;         // True if greedy was used (hint or fallback)
    _Bool greedy_fallback;     // True if pure greedy fallback was used
    int cpsat_attempts;        // Number of CP-SAT attempts made
    int best_attempt;          // Which attempt produced the result
    int unplaced_pods;         // Number of pods that couldn't be placed
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
    const int* max_runtime_secs,
    const double* improvement_threshold,
    const _Bool* score_only,
    const int* max_attempts,
    const _Bool* use_greedy_hint,
    const int* greedy_hint_attempt,
    const _Bool* fallback_to_greedy,
    const _Bool* greedy_only
) {
    // Get the function pointer using dlsym with the specific handle
    void* sym = dlsym(lib_handle, "OptimisePlacement");
    if (!sym) {
        // Try with underscore prefix (macOS)
        sym = dlsym(lib_handle, "_OptimisePlacement");
        if (!sym) {
            SolverResult err_result = {0, 0.0, -1, 0.0, 0.0, 0, 0, 0, 0, 0, 0};
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
        const int*,
        const double*,
        const _Bool*,
        const int*,
        const _Bool*,
        const int*,
        const _Bool*,
        const _Bool*
    );

    OptimisePlacementFunc func = (OptimisePlacementFunc)sym;

    // Call the function
    return func(mds, num_mds, pods, num_pods, plugin_scores,
                allowed_matrix, initial_assignment, out_slot_assignments,
                out_slots_used, max_runtime_secs, improvement_threshold, score_only,
                max_attempts, use_greedy_hint, greedy_hint_attempt,
                fallback_to_greedy, greedy_only);
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
	GetCurrentMDAssignment() int // Returns MD index (-1 if unknown)
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

// OptimizationConfig holds configuration options for the optimization process
type OptimizationConfig struct {
	// Basic optimization controls
	MaxRuntimeSeconds    *int     // Timeout per CP-SAT attempt (default: 15s)
	ImprovementThreshold *float64 // Minimum improvement % to accept (default: 0.0 = any improvement)
	ScoreOnly            *bool    // Just calculate score, no optimization (default: false)

	// Retry controls
	MaxAttempts *int // Number of CP-SAT attempts (default: 1)

	// Greedy heuristic controls
	UseGreedyHint      *bool // Use greedy as hint for CP-SAT attempt N (default: false)
	GreedyHintAttempt  *int  // Which attempt to inject greedy hint (default: 2)
	FallbackToGreedy   *bool // Use pure greedy if all CP-SAT attempts fail (default: false)
	GreedyOnly         *bool // Skip CP-SAT entirely, only run greedy (default: false)
}

type Result struct {
	PodAssignments   []PodSlotAssignment
	SlotsUsed        [][]bool
	Succeeded        bool
	Objective        float64
	SolverStatus     int
	SolveTimeSecs    float64
	Message          string
	CurrentStateCost float64 // Cost of the current state before optimization
	AlreadyOptimal   bool    // True if improvement doesn't meet threshold
	UsedGreedy       bool    // True if greedy algorithm was used (hint or fallback)
	GreedyFallback   bool    // True if pure greedy fallback was used (not just hint)
	CPSATAttempts    int     // Number of CP-SAT attempts made
	BestAttempt      int     // Which attempt produced the result (0 = greedy-only)
	UnplacedPods     int     // Number of pods that couldn't be placed (greedy only)
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

func convertMachineDeployments(mds []MachineDeployment) ([]C.MachineDeployment, []*C.char, func()) {
	numMDs := len(mds)
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

	cleanup := func() {
		for _, cstr := range cNames {
			C.free(unsafe.Pointer(cstr))
		}
	}

	return cMDs, cNames, cleanup
}

func convertPods(pods []Pod) ([]C.Pod, []unsafe.Pointer, []unsafe.Pointer) {
	numPods := len(pods)
	cPods := make([]C.Pod, numPods)
	hardPtrs := []unsafe.Pointer{}
	softPtrs := []unsafe.Pointer{}

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
			cpu:                   C.double(base.GetCPU()),
			memory:                C.double(base.GetMemory()),
			affinity_peers:        cPeers,
			affinity_rules:        cRules,
			affinity_count:        C.int(len(peers)),
			soft_peers:            cSoftPeers,
			soft_affinities:       cSoftWeights,
			soft_affinity_count:   C.int(len(softPeers)),
			current_md_assignment: C.int(base.GetCurrentMDAssignment()),
		}
	}

	return cPods, hardPtrs, softPtrs
}

func prepareRuntimeParameters(maxRuntimeSeconds *int, improvementThreshold *float64) (*C.int, *C.double, func()) {
	// Default to 15s
	if maxRuntimeSeconds == nil {
		defaultRuntime := 15
		maxRuntimeSeconds = &defaultRuntime
	}

	cMaxRuntime := (*C.int)(C.malloc(C.size_t(unsafe.Sizeof(C.int(0)))))
	*cMaxRuntime = C.int(*maxRuntimeSeconds)

	var cImprovementThreshold *C.double
	if improvementThreshold != nil {
		cImprovementThreshold = (*C.double)(C.malloc(C.size_t(unsafe.Sizeof(C.double(0)))))
		*cImprovementThreshold = C.double(*improvementThreshold)
	}

	cleanup := func() {
		C.free(unsafe.Pointer(cMaxRuntime))
		if cImprovementThreshold != nil {
			C.free(unsafe.Pointer(cImprovementThreshold))
		}
	}

	return cMaxRuntime, cImprovementThreshold, cleanup
}

func prepareMatrices(pluginScores []float64, allowedMatrix []int) (*C.double, *C.int, func()) {
	cScores := (*C.double)(C.malloc(C.size_t(len(pluginScores)) * C.size_t(unsafe.Sizeof(C.double(0)))))
	goScores := (*[1 << 30]C.double)(unsafe.Pointer(cScores))[:len(pluginScores):len(pluginScores)]
	for i, s := range pluginScores {
		goScores[i] = C.double(s)
	}

	cAllowed := (*C.int)(C.malloc(C.size_t(len(allowedMatrix)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	goAllowed := (*[1 << 30]C.int)(unsafe.Pointer(cAllowed))[:len(allowedMatrix):len(allowedMatrix)]
	for i, val := range allowedMatrix {
		goAllowed[i] = C.int(val)
	}

	cleanup := func() {
		C.free(unsafe.Pointer(cScores))
		C.free(unsafe.Pointer(cAllowed))
	}

	return cScores, cAllowed, cleanup
}

func prepareHints(pods []Pod, mds []MachineDeployment, slotsPerMD []int, initialAssignment [][][]int) (*C.int, func()) {
	totalHintSize := 0
	for range pods {
		for j := range mds {
			totalHintSize += slotsPerMD[j]
		}
	}

	cHints := (*C.int)(C.malloc(C.size_t(totalHintSize) * C.size_t(unsafe.Sizeof(C.int(0)))))
	goHints := (*[1 << 30]C.int)(unsafe.Pointer(cHints))[:totalHintSize:totalHintSize]

	offset := 0
	for i := range pods {
		for j := range mds {
			for k := 0; k < slotsPerMD[j]; k++ {
				if len(initialAssignment) > i && len(initialAssignment[i]) > j && len(initialAssignment[i][j]) > k {
					goHints[offset] = C.int(initialAssignment[i][j][k])
				} else {
					goHints[offset] = C.int(0)
				}
				offset++
			}
		}
	}

	cleanup := func() {
		C.free(unsafe.Pointer(cHints))
	}

	return cHints, cleanup
}

func prepareOutputBuffers(numPods int, slotsPerMD []int) (*C.int, *C.uint8_t, int, func()) {
	outAssign := (*C.int)(C.malloc(C.size_t(numPods*2) * C.size_t(unsafe.Sizeof(C.int(0)))))

	outSlotsUsedCount := 0
	for _, n := range slotsPerMD {
		outSlotsUsedCount += n
	}
	outSlots := (*C.uint8_t)(C.malloc(C.size_t(outSlotsUsedCount)))

	cleanup := func() {
		C.free(unsafe.Pointer(outAssign))
		C.free(unsafe.Pointer(outSlots))
	}

	return outAssign, outSlots, outSlotsUsedCount, cleanup
}

func parseResults(
	res C.SolverResult,
	outAssign *C.int,
	outSlots *C.uint8_t,
	numPods int,
	slotsPerMD []int,
	outSlotsUsedCount int,
) Result {
	result := Result{
		Succeeded:        bool(res.success),
		Objective:        float64(res.objective),
		SolverStatus:     int(res.status_code),
		SolveTimeSecs:    float64(res.solve_time_secs),
		Message:          "",
		CurrentStateCost: float64(res.current_state_cost),
		AlreadyOptimal:   bool(res.already_optimal),
		UsedGreedy:       bool(res.used_greedy),
		GreedyFallback:   bool(res.greedy_fallback),
		CPSATAttempts:    int(res.cpsat_attempts),
		BestAttempt:      int(res.best_attempt),
		UnplacedPods:     int(res.unplaced_pods),
	}

	// In score-only mode, assignments will be empty (-1 values)
	// Check if this is score-only mode by looking at the first assignment
	rawAssign := (*[1 << 30]C.int)(unsafe.Pointer(outAssign))[: numPods*2 : numPods*2]
	isScoreOnly := numPods > 0 && rawAssign[0] == -1

	if isScoreOnly {
		// Score-only mode: return empty assignments
		result.PodAssignments = []PodSlotAssignment{}
		result.SlotsUsed = make([][]bool, len(slotsPerMD))
		for j := range len(slotsPerMD) {
			result.SlotsUsed[j] = make([]bool, slotsPerMD[j])
		}
		if result.Succeeded {
			result.Message = "Score-only mode: current state score calculated"
		}
		return result
	}

	// Normal optimization mode: parse assignments
	assignments := make([]PodSlotAssignment, numPods)
	for i := range numPods {
		assignments[i] = PodSlotAssignment{
			MD:   int(rawAssign[i*2]),
			Slot: int(rawAssign[i*2+1]),
		}
	}

	rawSlots := (*[1 << 30]C.uint8_t)(unsafe.Pointer(outSlots))[:outSlotsUsedCount:outSlotsUsedCount]
	slotsUsed := make([][]bool, len(slotsPerMD))
	offset := 0
	for j := range len(slotsPerMD) {
		slotsUsed[j] = make([]bool, slotsPerMD[j])
		for k := 0; k < slotsPerMD[j]; k++ {
			slotsUsed[j][k] = rawSlots[offset] != 0
			offset++
		}
	}

	result.PodAssignments = assignments
	result.SlotsUsed = slotsUsed

	if !result.Succeeded {
		result.Message = fmt.Sprintf("Solver failed with status code %d", result.SolverStatus)
		return result
	}

	result.Message = "Optimisation successful"
	return result
}

func OptimisePlacementRaw(
	mds []MachineDeployment,
	pods []Pod,
	pluginScores []float64,
	allowedMatrix []int,
	initialAssignment [][][]int,
	config *OptimizationConfig,
) Result {
	if err := extractAndLoadSharedLibrary(); err != nil {
		return Result{Message: fmt.Sprintf("Failed to load optimiser lib: %v", err)}
	}

	// Extract individual config parameters (nil config is allowed)
	var maxRuntimeSeconds *int
	var improvementThreshold *float64
	var scoreOnly *bool
	var maxAttempts *int
	var useGreedyHint *bool
	var greedyHintAttempt *int
	var fallbackToGreedy *bool
	var greedyOnly *bool

	if config != nil {
		maxRuntimeSeconds = config.MaxRuntimeSeconds
		improvementThreshold = config.ImprovementThreshold
		scoreOnly = config.ScoreOnly
		maxAttempts = config.MaxAttempts
		useGreedyHint = config.UseGreedyHint
		greedyHintAttempt = config.GreedyHintAttempt
		fallbackToGreedy = config.FallbackToGreedy
		greedyOnly = config.GreedyOnly
	}

	numMDs := len(mds)
	slotsPerMD := make([]int, numMDs)
	for i, md := range mds {
		slotsPerMD[i] = md.GetMaxScaleOut()
	}
	numPods := len(pods)

	cMDs, _, cleanupMDs := convertMachineDeployments(mds)
	defer cleanupMDs()

	cPods, hardPtrs, softPtrs := convertPods(pods)
	defer func() {
		for _, ptr := range hardPtrs {
			C.free(ptr)
		}
		for _, ptr := range softPtrs {
			C.free(ptr)
		}
	}()

	cMaxRuntime, cImprovementThreshold, cleanupRuntime := prepareRuntimeParameters(maxRuntimeSeconds, improvementThreshold)
	defer cleanupRuntime()

	cScores, cAllowed, cleanupMatrices := prepareMatrices(pluginScores, allowedMatrix)
	defer cleanupMatrices()

	cHints, cleanupHints := prepareHints(pods, mds, slotsPerMD, initialAssignment)
	defer cleanupHints()

	outAssign, outSlots, outSlotsUsedCount, cleanupOutput := prepareOutputBuffers(numPods, slotsPerMD)
	defer cleanupOutput()

	// Prepare optional boolean parameters
	var cScoreOnly *C.bool
	if scoreOnly != nil && *scoreOnly {
		cScoreOnly = (*C.bool)(C.malloc(C.size_t(unsafe.Sizeof(C.bool(false)))))
		*cScoreOnly = C.bool(true)
		defer C.free(unsafe.Pointer(cScoreOnly))
	}

	var cMaxAttempts *C.int
	if maxAttempts != nil {
		cMaxAttempts = (*C.int)(C.malloc(C.size_t(unsafe.Sizeof(C.int(0)))))
		*cMaxAttempts = C.int(*maxAttempts)
		defer C.free(unsafe.Pointer(cMaxAttempts))
	}

	var cUseGreedyHint *C.bool
	if useGreedyHint != nil {
		cUseGreedyHint = (*C.bool)(C.malloc(C.size_t(unsafe.Sizeof(C.bool(false)))))
		*cUseGreedyHint = C.bool(*useGreedyHint)
		defer C.free(unsafe.Pointer(cUseGreedyHint))
	}

	var cGreedyHintAttempt *C.int
	if greedyHintAttempt != nil {
		cGreedyHintAttempt = (*C.int)(C.malloc(C.size_t(unsafe.Sizeof(C.int(0)))))
		*cGreedyHintAttempt = C.int(*greedyHintAttempt)
		defer C.free(unsafe.Pointer(cGreedyHintAttempt))
	}

	var cFallbackToGreedy *C.bool
	if fallbackToGreedy != nil {
		cFallbackToGreedy = (*C.bool)(C.malloc(C.size_t(unsafe.Sizeof(C.bool(false)))))
		*cFallbackToGreedy = C.bool(*fallbackToGreedy)
		defer C.free(unsafe.Pointer(cFallbackToGreedy))
	}

	var cGreedyOnly *C.bool
	if greedyOnly != nil {
		cGreedyOnly = (*C.bool)(C.malloc(C.size_t(unsafe.Sizeof(C.bool(false)))))
		*cGreedyOnly = C.bool(*greedyOnly)
		defer C.free(unsafe.Pointer(cGreedyOnly))
	}

	res := C.call_optimise_placement_with_handle(
		GetLibHandle(),
		(*C.MachineDeployment)(unsafe.Pointer(&cMDs[0])), C.int(numMDs),
		(*C.Pod)(unsafe.Pointer(&cPods[0])), C.int(numPods),
		cScores, cAllowed, cHints, outAssign, outSlots, cMaxRuntime,
		cImprovementThreshold, cScoreOnly,
		cMaxAttempts, cUseGreedyHint, cGreedyHintAttempt,
		cFallbackToGreedy, cGreedyOnly,
	)

	return parseResults(res, outAssign, outSlots, numPods, slotsPerMD, outSlotsUsedCount)
}

// Helper functions for creating pointers to config values
func IntPtr(i int) *int {
	return &i
}

func BoolPtr(b bool) *bool {
	return &b
}

func FloatPtr(f float64) *float64 {
	return &f
}

// OptimisePlacement provides backward compatibility with the old API
// Deprecated: Use OptimisePlacementRaw with OptimizationConfig instead
func OptimisePlacement(
	mds []MachineDeployment,
	pods []Pod,
	pluginScores []float64,
	allowedMatrix []int,
	initialAssignment [][][]int,
	maxRuntimeSeconds *int,
	improvementThreshold *float64,
	scoreOnly *bool,
) Result {
	config := &OptimizationConfig{
		MaxRuntimeSeconds:    maxRuntimeSeconds,
		ImprovementThreshold: improvementThreshold,
		ScoreOnly:            scoreOnly,
	}
	return OptimisePlacementRaw(mds, pods, pluginScores, allowedMatrix, initialAssignment, config)
}
