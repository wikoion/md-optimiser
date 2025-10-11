# Machine Deployment Optimiser

This project demonstrates how to pack containers onto different machine deployment
(MD) types using Google's OR‑Tools CP‑SAT solver. The core optimiser is written in
C++ (`cpp/optimiser.cpp`) and exposed to Go through dynamic library loading.
It solves a mixed integer program that assigns each pod to exactly one MD slot while
respecting resource limits and optional placement restrictions. The objective is
to minimise a weighted count of the nodes required across all deployments.

## Building

Prebuilt shared libraries for Linux (amd64, arm64) and macOS (arm64) are included in `lib/`. 
To rebuild the C++ solver, run `go generate` in the module root which invokes the
`Makefile` and creates the platform-specific library files.

```bash
# rebuilds platform-specific libraries
go generate
go build
```

## Usage

Import the module as `github.com/wikoion/md-optimiser` and call `OptimisePlacementRaw`. Machine Deployment and Pod interfaces
exist to allow type flexibility. Implement your own types and pass your mds, pods and placement rules as slices.
Plugin scores allow expressing per‑deployment preferences (higher scores = more preferred).

### Basic Example

```go
type md struct {
    Name string
    // Your fields
}

func (m *md) GetName() string {
    return m.Name
}
// Implement interface funcs: GetName(), GetCPU(), GetMemory() and GetMaxScaleOut()

type pod struct {
    // Do the same for pod, including GetCurrentMDAssignment() for warm starts
}

mds := []optimiser.MachineDeployment{
    &md{Name: "md-a", CPU: 16, Memory: 64, MaxScaleOut: 3},
}

pods := []optimiser.Pod{
    &pod{CPU: 2, Memory: 4},
}

scores := []float64{1.0}              // one score per MD
allowed := []int{1}                   // pod 0 may run on MD 0
initial := [][][]int{{{0, 0}}}        // optional warm start: pod 0 -> MD 0, slot 0

// Configure optimization behavior
config := &optimiser.OptimizationConfig{
    MaxRuntimeSeconds:    optimiser.IntPtr(15),    // timeout per CP-SAT attempt
    ImprovementThreshold: optimiser.FloatPtr(5.0), // require 5% improvement
}

result := optimiser.OptimisePlacementRaw(
    mds, pods, scores, allowed, initial, config,
)
if result.Succeeded {
    for i, assignment := range result.PodAssignments {
        fmt.Printf("Pod %d -> MD %d, Slot %d\n", i, assignment.MD, assignment.Slot)
    }
    fmt.Printf("Used greedy: %v, CP-SAT attempts: %d\n",
        result.UsedGreedy, result.CPSATAttempts)
}
```

Refer to `example/main.go` for a more complete flow including scoring
plugins, greedy warm start generation and reporting of resource waste.

## Key Features

### Greedy Heuristic with CP-SAT Optimization
The optimiser now includes a fast greedy placement heuristic that can be used standalone or combined with CP-SAT optimization:

#### Greedy-Only Mode
Use the greedy heuristic alone for fast approximate solutions:
```go
config := &optimiser.OptimizationConfig{
    GreedyOnly: optimiser.BoolPtr(true),
}
result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, nil, config)
```
- Very fast (milliseconds vs seconds for CP-SAT)
- Affinity-aware placement with BFS-based grouping
- Best-fit bin-packing to minimize waste
- Constraint-aware pod ordering (most constrained first)
- Ideal for large workloads or time-sensitive scenarios

#### Greedy Hint Mode
Use greedy to warm-start CP-SAT for improved solution quality:
```go
config := &optimiser.OptimizationConfig{
    MaxAttempts:       optimiser.IntPtr(5),
    UseGreedyHint:     optimiser.BoolPtr(true),
    GreedyHintAttempt: optimiser.IntPtr(2),  // inject greedy hint on attempt 2
    MaxRuntimeSeconds: optimiser.IntPtr(10),
}
result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, nil, config)
```
- Runs greedy algorithm to generate initial solution
- Provides hint to CP-SAT on specified attempt
- Often helps CP-SAT find better solutions faster
- Check `result.UsedGreedy` to see if hint was used

#### Greedy Fallback Mode
Use greedy as a safety net if all CP-SAT attempts fail:
```go
config := &optimiser.OptimizationConfig{
    MaxAttempts:      optimiser.IntPtr(3),
    FallbackToGreedy: optimiser.BoolPtr(true),
    MaxRuntimeSeconds: optimiser.IntPtr(5),
}
result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, nil, config)
```
- Attempts CP-SAT optimization first
- Falls back to greedy if no acceptable solution found
- Ensures you always get a valid placement
- Check `result.GreedyFallback` to see if fallback was used

#### Multiple CP-SAT Attempts
Run CP-SAT multiple times to find better solutions:
```go
config := &optimiser.OptimizationConfig{
    MaxAttempts:       optimiser.IntPtr(5),  // try up to 5 times
    MaxRuntimeSeconds: optimiser.IntPtr(10), // 10 seconds per attempt
}
result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, nil, config)
fmt.Printf("Best solution found on attempt %d\n", result.BestAttempt)
```
- Each attempt uses the same timeout
- Tracks best solution across all attempts
- Returns as soon as an acceptable solution is found
- Use `result.CPSATAttempts` and `result.BestAttempt` for diagnostics

### Slot-Aware Placement
The optimiser explicitly models individual slots within each machine deployment. This provides:
- Fine-grained control over pod placement within scaled-out deployments
- Better resource utilization tracking per slot
- Support for affinity rules at the slot level

### Score-Only Mode
The optimiser can evaluate the current deployment score without performing optimization:
```go
config := &optimiser.OptimizationConfig{
    ScoreOnly: optimiser.BoolPtr(true),
}
result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, nil, config)
fmt.Printf("Current deployment score: %.2f\n", result.Objective)
```
- Returns immediately without running CP-SAT or greedy (very fast, O(n) complexity)
- Use for monitoring, alerting, or deciding whether to trigger optimization
- Result includes the current state score in the `Objective` and `CurrentStateCost` fields
- `PodAssignments` will be empty in score-only mode

### Improvement Threshold
The optimiser can evaluate whether a re-optimization is worthwhile based on a minimum improvement percentage:
- Set `improvementThreshold` to require a minimum cost reduction (e.g., 5%)
- Result includes `AlreadyOptimal` flag when current state is sufficiently optimal
- Helps avoid unnecessary migrations for marginal improvements

### Dynamic Library Loading
The C++ optimiser is loaded dynamically at runtime:
- Automatic platform detection (Linux amd64/arm64, macOS arm64)
- No CGO compilation required for end users
- Simplified deployment and cross-platform support

### Object-Oriented C++ Implementation
The core optimiser has been refactored into a class-based architecture:
- Cleaner separation of concerns
- Improved maintainability and extensibility
- Better encapsulation of solver state

## Running the Example

```bash
go generate
go run example/main.go
```

The example prints the chosen deployment for each pod along with the number of
nodes used and solver statistics.

## License

Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
