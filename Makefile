.PHONY: all build clean test

CPP_DIR = cpp
GO_DIR = go

all: build

build:
	mkdir -p $(CPP_DIR)/build
	cd $(CPP_DIR)/build && cmake .. && make
	cp $(CPP_DIR)/build/liboptimiser.dylib $(GO_DIR)/

clean:
	rm -rf $(CPP_DIR)/build
	rm -f $(GO_DIR)/liboptimiser.dylib

