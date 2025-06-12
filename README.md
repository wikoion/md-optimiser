# Capi Machine Deployment Optimiser
This is an example of how you can use Google's or-tools to use custom scoring to bin pack pods onto capi machine deployments (mixed integer linear programming)
Build instructions:
```
cd go
go generate
CGO_LDFLAGS="-Wl,-rpath,$PWD" go build
```

To run example/main.go:
```
cd go
go generate
export DYLD_LIBRARY_PATH=$PWD
go run example/main.go
```
