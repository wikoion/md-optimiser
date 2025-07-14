package optimiser

/*
#cgo LDFLAGS: -ldl
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

	return nil
}
