.PHONY: all build clean

CPP_DIR := cpp
GO_DIR := .
BUILD_DIR := $(CPP_DIR)/build

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    LIB_FILES := liboptimiser.so
else ifeq ($(UNAME_S),Darwin)
    LIB_FILES := liboptimiser.dylib liboptimiser.so
else ifeq ($(OS),Windows_NT)
    LIB_FILES := optimiser.dll
else
    $(error Unsupported OS: $(UNAME_S))
endif

all: build

build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make
	cp $(addprefix $(BUILD_DIR)/, $(LIB_FILES)) $(GO_DIR)/

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(addprefix $(GO_DIR)/, $(LIB_FILES))
