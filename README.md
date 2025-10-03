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
maxRuntime := 10                      // optional runtime limit in seconds
improvementThreshold := 5.0           // optional: min improvement % required

result := optimiser.OptimisePlacementRaw(
    mds, pods, scores, allowed, initial, &maxRuntime, &improvementThreshold,
)
if result.Succeeded {
    for i, assignment := range result.PodAssignments {
        fmt.Printf("Pod %d -> MD %d, Slot %d\n", i, assignment.MD, assignment.Slot)
    }
}
```

Refer to `example/main.go` for a more complete flow including scoring
plugins, greedy warm start generation and reporting of resource waste.

## Key Features

### Slot-Aware Placement
The optimiser now explicitly models individual slots within each machine deployment. This provides:
- Fine-grained control over pod placement within scaled-out deployments
- Better resource utilization tracking per slot
- Support for affinity rules at the slot level

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
