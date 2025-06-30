GOOS ?= $(shell go env GOOS)
GOARCH ?= $(shell go env GOARCH)

all: build

build:
	@mkdir -p cpp/build lib

	# Build natively
	cd cpp/build && cmake .. && make

	# macOS native .dylib build
	@if [ "$(GOOS)" = "darwin" ] && [ -f cpp/build/liboptimiser.dylib ]; then \
	  echo "macOS: fixing install_name and copying liboptimiser.dylib"; \
	  install_name_tool -id @rpath/liboptimiser_darwin_arm64.dylib cpp/build/liboptimiser.dylib; \
	  cp cpp/build/liboptimiser.dylib lib/liboptimiser_darwin_arm64.dylib; \
	  echo "macOS: building Linux .so using Docker"; \
	  docker buildx build --platform=linux/amd64 \
	    -f Dockerfile.build-linux-so \
	    --output type=local,dest=. \
	    -t optimiser-builder .; \
	fi

	# Linux native .so build
	@if [ "$(GOOS)" = "linux" ] && [ -f cpp/build/liboptimiser.so ]; then \
	  echo "Linux: copying liboptimiser.so"; \
	  cp cpp/build/liboptimiser.so lib/liboptimiser.so; \
	fi

clean:
	@echo "Cleaning build output"
	@if [ "$(GOOS)" = "darwin" ]; then \
	  rm -f lib/liboptimiser_darwin_arm64.dylib \
	fi
	@if [ "$(GOOS)" = "linux" ]; then \
	  rm -f lib/liboptimiser.so; \
	fi
	rm -rf cpp/build
