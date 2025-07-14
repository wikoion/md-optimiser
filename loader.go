package optimiser

/*
#include <dlfcn.h>
#include <stdlib.h>
*/
import "C"

import (
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"
)

//go:embed lib/liboptimiser_*.dylib lib/liboptimiser_*.so
var embeddedLibs embed.FS

var alreadyLoaded = false
var libHandle unsafe.Pointer

func extractAndLoadSharedLibrary() error {
	if alreadyLoaded {
		return nil
	}
	alreadyLoaded = true

	var libName string
	switch {
	case runtime.GOOS == "darwin" && runtime.GOARCH == "arm64":
		libName = "liboptimiser_darwin_arm64.dylib"
	case runtime.GOOS == "linux" && runtime.GOARCH == "amd64":
		libName = "liboptimiser_linux_amd64.so"
	case runtime.GOOS == "linux" && runtime.GOARCH == "arm64":
		libName = "liboptimiser_linux_arm64.so"
	default:
		return fmt.Errorf("unsupported platform: %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	data, err := embeddedLibs.ReadFile("lib/" + libName)
	if err != nil {
		return fmt.Errorf("reading embedded lib: %w", err)
	}

	tmpPath := filepath.Join(os.TempDir(), libName)
	if err := os.WriteFile(tmpPath, data, 0o755); err != nil {
		return fmt.Errorf("writing embedded lib to disk: %w", err)
	}

	cPath := C.CString(tmpPath)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.dlopen(cPath, C.RTLD_NOW|C.RTLD_GLOBAL)
	if handle == nil {
		return fmt.Errorf("dlopen failed for %s", tmpPath)
	}
	
	libHandle = handle
	return nil
}

func getOptimisePlacementFunc() (unsafe.Pointer, error) {
	if libHandle == nil {
		return nil, fmt.Errorf("library not loaded")
	}
	
	symName := C.CString("OptimisePlacement")
	defer C.free(unsafe.Pointer(symName))
	
	sym := C.dlsym(libHandle, symName)
	if sym == nil {
		return nil, fmt.Errorf("symbol OptimisePlacement not found")
	}
	
	return sym, nil
}
