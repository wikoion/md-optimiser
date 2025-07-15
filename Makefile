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
	  echo "macOS: building Linux amd64 .so using Docker"; \
	  docker buildx build --platform=linux/amd64 \
	    -f Dockerfile.build-linux-so \
	    --output type=local,dest=. \
	    -t optimiser-builder-amd64 .; \
	  echo "macOS: building Linux arm64 .so using Docker"; \
	  docker buildx build --platform=linux/arm64 \
	    -f Dockerfile.build-linux-arm64-so \
	    --output type=local,dest=. \
	    -t optimiser-builder-arm64 .; \
	fi

	# Linux native .so build
	@if [ "$(GOOS)" = "linux" ] && [ -f cpp/build/liboptimiser.so ]; then \
	  echo "Linux: copying liboptimiser.so"; \
	  if [ "$(GOARCH)" = "amd64" ]; then \
	    cp cpp/build/liboptimiser.so lib/liboptimiser_linux_amd64.so; \
	  elif [ "$(GOARCH)" = "arm64" ]; then \
	    cp cpp/build/liboptimiser.so lib/liboptimiser_linux_arm64.so; \
	  fi; \
	fi

clean:
	@echo "Cleaning build output"
	@if [ "$(GOOS)" = "darwin" ]; then \
	  rm -f lib/liboptimiser_darwin_arm64.dylib \
	fi
	@if [ "$(GOOS)" = "linux" ]; then \
	  rm -f lib/liboptimiser_linux_*.so; \
	fi
	rm -rf cpp/build

test:
	go test ./...

lint:
	golangci-lint run --fix
