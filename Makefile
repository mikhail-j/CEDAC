# SPDX-License-Identifier: GPL-3.0-only

LIB_DIR:=
INCLUDE_DIR:=/usr/include
SHARED_LIB_EXT:=


ifeq ($(OS),Windows_NT)
SHARED_LIB_EXT:=.dll
else
	ifeq ($(shell uname -s),Darwin)
SHARED_LIB_EXT:=.dylib
	else
SHARED_LIB_EXT:=.so
	endif
endif

ifeq ($(OS),Windows_NT)
$(error Error: Windows operating systems are currently unsupported!)
else
	ifeq ($(shell [ -e "/usr/lib64" ] && echo "lib64"),lib64)
LIB_DIR:=/usr/lib64
	else
LIB_DIR:=/usr/lib
	endif
endif


.PHONY: all test install uninstall clean

all:
	@$(MAKE) -C opencl
	@$(MAKE) -C cuda

install:
	install -m 644 include/clecc.h $(INCLUDE_DIR)
	install -m 644 include/cuecc.h $(INCLUDE_DIR)
	install -s opencl/libclecc$(SHARED_LIB_EXT) $(LIB_DIR)
	install -s cuda/libcuecc$(SHARED_LIB_EXT) $(LIB_DIR)
	@ldconfig

uninstall:
	-rm -f $(INCLUDE_DIR)/clecc.h
	-rm -f $(INCLUDE_DIR)/cuecc.h
	-rm -f $(LIB_DIR)/libclecc$(SHARED_LIB_EXT)
	-rm -f $(LIB_DIR)/libcuecc$(SHARED_LIB_EXT)
	@ldconfig

test:
	@$(MAKE) -C opencl test
	@$(MAKE) -C cuda test

clean:
	@-$(MAKE) -C opencl clean
	@-$(MAKE) -C cuda clean
