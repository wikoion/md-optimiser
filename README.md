# Capi Machine Deployment Optimiser
This is an example of how you can use Google's or-tools to use custom scoring to bin pack pods onto capi machine deployments (mixed integer linear programming)
build instructions:
```
cd go
go generate
CGO_LDFLAGS="-Wl,-rpath,$PWD" go build && ./md_solver
```
