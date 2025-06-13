//go:build linux
// +build linux

package md_solver

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdlib.h>

void* load_shared_lib(const char* path) {
    return dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
}
*/
import "C"

import (
	_ "embed"
	"os"
	"path/filepath"
	"unsafe"
)

//go:embed liboptimiser.so
var libLinux []byte

func init() {
	path := filepath.Join(os.TempDir(), "liboptimiser.so")
	if err := os.WriteFile(path, libLinux, 0o755); err != nil {
		panic("failed to write optimiser.so: " + err.Error())
	}

	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	if handle := C.load_shared_lib(cpath); handle == nil {
		panic("failed to dlopen optimiser.so")
	}
}
