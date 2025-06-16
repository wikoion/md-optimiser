all: build

build:
	@mkdir -p cpp/build lib
	cd cpp/build && cmake .. && make

	# macOS ARM
	@if [ -f cpp/build/liboptimiser.dylib ]; then \
	  install_name_tool -id @rpath/liboptimiser_darwin_arm64.dylib cpp/build/liboptimiser.dylib; \
	  cp cpp/build/liboptimiser.dylib lib/liboptimiser_darwin_arm64.dylib; \
	fi

	# Linux x86_64
	@if [ -f cpp/build/liboptimiser.so ]; then \
	  cp cpp/build/liboptimiser.so lib/liboptimiser_linux_amd64.so; \
	fi

clean:
	rm -rf cpp/build lib
