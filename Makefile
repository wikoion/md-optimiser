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
	fi

	# Linux native .so build
	@if [ "$(GOOS)" = "linux" ] && [ -f cpp/build/liboptimiser.so ]; then \
	  echo "Linux: copying liboptimiser.so"; \
	  cp cpp/build/liboptimiser.so lib/liboptimiser_linux_amd64.so; \
	fi

	# On macOS, cross-compile Linux .so via Docker
	@if [ "$(GOOS)" = "darwin" ]; then \
	  echo "macOS: cross-compiling Linux .so via Docker"; \
	  docker buildx build --platform linux/amd64 -f Dockerfile.build-linux-so -t ortools-builder . ; \
	  docker run --rm --platform linux/amd64 -v "$$(pwd)":/src -w /src ortools-builder bash -c '\
	  mkdir -p /tmp/linux-build && cd /tmp/linux-build && \
	  cmake /src/cpp -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH=/opt/ortools/lib && \
	  make && cp liboptimiser.so /src/lib/liboptimiser_linux_amd64.so' ; \
	fi


clean:
	@echo "Cleaning build output"
	@if [ "$(GOOS)" = "darwin" ]; then \
	  rm -f lib/liboptimiser_darwin_arm64.dylib lib/liboptimiser_linux_amd64.so; \
	fi
	@if [ "$(GOOS)" = "linux" ]; then \
	  rm -f lib/liboptimiser_linux_amd64.so; \
	fi
	rm -rf cpp/build
