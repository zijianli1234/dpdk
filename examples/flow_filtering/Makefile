# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2017 Mellanox Technologies, Ltd

APP = flow

SRCS-y := main.c flow_skeleton.c
subdir := snippets
snippet_objects := $(addprefix build/,$(notdir $(patsubst %.c,%.o,$(wildcard $(subdir)/*.c))))

PKGCONF ?= pkg-config

# Build using pkg-config variables if possible
ifneq ($(shell $(PKGCONF) --exists libdpdk && echo 0),0)
$(error "no installation of DPDK found")
endif

all: shared
.PHONY: shared static
shared: build/$(APP)-shared
	ln -sf $(APP)-shared build/$(APP)
static: build/$(APP)-static
	ln -sf $(APP)-static build/$(APP)

PC_FILE := $(shell $(PKGCONF) --path libdpdk 2>/dev/null)
CFLAGS += -O3 $(shell $(PKGCONF) --cflags libdpdk)
LDFLAGS_SHARED = $(shell $(PKGCONF) --libs libdpdk)
LDFLAGS_STATIC = $(shell $(PKGCONF) --static --libs libdpdk)

CFLAGS += -DALLOW_EXPERIMENTAL_API

build/$(APP)-shared: $(SRCS-y) $(snippet_objects) Makefile $(PC_FILE) | build
	$(CC) $(CFLAGS) $(SRCS-y) $(snippet_objects) -o $@ $(LDFLAGS) $(LDFLAGS_SHARED)

build/$(APP)-static: $(SRCS-y) $(snippet_objects) Makefile $(PC_FILE) | build
	$(CC) $(CFLAGS) $(SRCS-y) $(snippet_objects) -o $@ $(LDFLAGS) $(LDFLAGS_STATIC)

build/snippet_%.o: $(subdir)/snippet_%.c | build
	$(CC) -c $(CFLAGS) -Wno-unused-function $< -o $@

build:
	@mkdir -p $@

.PHONY: clean
clean:
	rm -f build/$(APP) build/$(APP)-static build/$(APP)-shared build/*.o
	test -d build && rmdir -p build || true
