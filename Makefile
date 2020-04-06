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
# skip setting library directory in Windows
else
	ifeq ($(shell [ -e "/lib64" ] && echo "lib64"),lib64)
LIB_DIR:=/lib64
	else
LIB_DIR:=/lib
	endif
endif


.PHONY: all test install uninstall clean

all:
	@$(MAKE) -C opencl
	@$(MAKE) -C cuda

ifeq ($(OS),Windows_NT)
install: opencl/libclecc$(SHARED_LIB_EXT) cuda/libcuecc$(SHARED_LIB_EXT)
	$(error Error: Makefile 'install' target does not support Windows operating systems!)

uninstall:
	$(error Error: Makefile 'uninstall' target does not support Windows operating systems!)
else
install: opencl/libclecc$(SHARED_LIB_EXT) cuda/libcuecc$(SHARED_LIB_EXT)
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
endif

test:
	@$(MAKE) -C opencl test
	@$(MAKE) -C cuda test

clean:
	@-$(MAKE) -C opencl clean
	@-$(MAKE) -C cuda clean
