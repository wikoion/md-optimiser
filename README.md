# Machine Deployment Optimiser

This project demonstrates how to pack containers onto different machine deployment
(MD) types using Google's OR‑Tools CP‑SAT solver. The core optimiser is written in
C++ (`cpp/optimiser.cpp`) and exposed to Go through cgo (`go/optimiser.go`).
It solves a mixed integer program that assigns each pod to exactly one MD while
respecting resource limits and optional placement restrictions. The objective is
to minimise a weighted count of the nodes required across all deployments.

## Building

Prebuilt shared libraries for Linux and macOS are committed to `go/`. To rebuild
the C++ solver, run `go generate` in the module root which invokes the
`Makefile` and embeds the resulting library files.

```bash
# rebuilds liboptimiser.{so,dylib} and embeds them
go generate
CGO_LDFLAGS="-Wl,-rpath,$PWD" go build
```

## Usage

Import the module as `github.com/wikoion/md-optimiser` and call `OptimisePlacement`. Machine Deployment and Pod interfaces
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
    // Do the same for pod
}

mds := []optimiser.MachineDeployment{
    &md{Name: "md-a", CPU: 16, Memory: 64, MaxScaleOut: 3},
}

pods := []optimiser.Pod{
    &pod{CPU: 2, Memory: 4},
}

scores := []float64{1.0}              // one score per MD
allowed := []int{1}                   // pod 0 may run on MD 0
initial := []int{-1}                  // optional starting hint

result := optimiser.OptimisePlacementRaw(mds, pods, scores, allowed, initial)
if result.Succeeded {
    fmt.Println("Assignments:", result.Assignments)
}
```

Refer to `example/main.go` for a more complete flow including scoring
plugins, greedy warm start generation and reporting of resource waste.

## Running the Example

```bash
go generate
export DYLD_LIBRARY_PATH=$PWD   # use LD_LIBRARY_PATH on Linux
go run example/main.go
```

The example prints the chosen deployment for each pod along with the number of
nodes used and solver statistics.

## License

Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
