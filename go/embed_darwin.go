//go:build darwin
// +build darwin

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

//go:embed liboptimiser.dylib
var libMac []byte

func init() {
	path := filepath.Join(os.TempDir(), "liboptimiser.dylib")
	if err := os.WriteFile(path, libMac, 0o755); err != nil {
		panic("failed to write optimiser.dylib: " + err.Error())
	}

	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	if handle := C.load_shared_lib(cpath); handle == nil {
		panic("failed to dlopen optimiser.dylib")
	}
}
