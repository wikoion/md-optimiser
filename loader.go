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
	"sync"
	"unsafe"
)

//go:embed lib/liboptimiser_*.dylib lib/liboptimiser_*.so
var embeddedLibs embed.FS

var alreadyLoaded = false
var libHandle unsafe.Pointer
var preloadOnce sync.Once
var preloadErr error

// GetLibHandle returns the loaded library handle
func GetLibHandle() unsafe.Pointer {
	return libHandle
}

func preloadORTools() error {
	preloadOnce.Do(func() {
		home := os.Getenv("ORTOOLS_HOME")
		if home == "" {
			home = "/opt/or-tools/current"
		}

		var libName string
		switch runtime.GOOS {
		case "linux":
			libName = "libortools.so"
		case "darwin":
			libName = "libortools.dylib"
		default:
			preloadErr = fmt.Errorf("unsupported platform for OR-Tools preload: %s", runtime.GOOS)
			return
		}

		libPath := filepath.Join(home, "lib", libName)
		if _, err := os.Stat(libPath); err != nil {
			preloadErr = fmt.Errorf("expected OR-Tools library at %s: %w", libPath, err)
			return
		}

		cPath := C.CString(libPath)
		defer C.free(unsafe.Pointer(cPath))

		handle := C.dlopen(cPath, C.RTLD_NOW|C.RTLD_GLOBAL)
		if handle == nil {
			preloadErr = fmt.Errorf("dlopen failed for %s: %s", libPath, C.GoString(C.dlerror()))
		}
	})

	return preloadErr
}

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

	if err := preloadORTools(); err != nil {
		return err
	}

	cPath := C.CString(tmpPath)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.dlopen(cPath, C.RTLD_NOW|C.RTLD_GLOBAL)
	if handle == nil {
		errStr := C.GoString(C.dlerror())
		return fmt.Errorf("dlopen failed for %s: %s", tmpPath, errStr)
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
