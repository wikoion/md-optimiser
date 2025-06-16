# Machine Deployment Optimiser

This project demonstrates how to pack containers onto different machine deployment
(MD) types using Google's OR‑Tools CP‑SAT solver. The core optimiser is written in
C++ (`cpp/optimiser.cpp`) and exposed to Go through cgo (`go/optimiser.go`).
It solves a mixed integer program that assigns each pod to exactly one MD while
respecting resource limits and optional placement restrictions. The objective is
to minimise a weighted count of the nodes required across all deployments.

## Building

Prebuilt shared libraries for Linux and macOS are committed to `go/`. To rebuild
the C++ solver, run `go generate` in the `go` directory which invokes the
`Makefile` and embeds the resulting library files.

```bash
cd go
# rebuilds liboptimiser.{so,dylib} and embeds them
go generate
CGO_LDFLAGS="-Wl,-rpath,$PWD" go build
```

## Usage

Import the module as `md_solver` and call `OptimisePlacement`. Machine Deployment and Pod interfaces
exist to allow type flexibility. Implement your own types and pass your mds, pods and placement rules as slices. 
Plugin scores allow expressing per‑deployment preferences (higher scores = more preferred).

```go
mds := []md_solver.MachineDeployment{
    {Name: "md-a", CPU: 16, Memory: 64, MaxScaleOut: 3},
}

pods := []md_solver.Pod{
    {CPU: 2, Memory: 4},
}

scores := []float64{1.0}              // one score per MD
allowed := []int{1}                   // pod 0 may run on MD 0
initial := []int{-1}                  // optional starting hint

result := md_solver.OptimisePlacementRaw(mds, pods, scores, allowed, initial)
if result.Succeeded {
    fmt.Println("Assignments:", result.Assignments)
}
```

Refer to `go/example/main.go` for a more complete flow including scoring
plugins, greedy warm start generation and reporting of resource waste.

## Running the Example

```bash
cd go
go generate
export DYLD_LIBRARY_PATH=$PWD   # use LD_LIBRARY_PATH on Linux
go run example/main.go
```

The example prints the chosen deployment for each pod along with the number of
nodes used and solver statistics.
