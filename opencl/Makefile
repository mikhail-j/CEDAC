# SPDX-License-Identifier: GPL-3.0-only

OS_SHARED_LIBRARY_FLAGS=
test_LDFLAGS=
CFLAGS=-std=c99 -O2 -pthread

RETPOLINE_THUNK_INLINE_CFLAGS:=
RETPOLINE_THUNK_EXTERN_CFLAGS:=

# determine shared library extension based on operating system
ifeq ($(OS),Windows_NT)
	SHARED_LIB_EXT = .dll
else
	OS_NAME := $(shell uname -s)
	ifeq ($(OS_NAME),Darwin)
		SHARED_LIB_EXT = .dylib
		OS_SHARED_LIBRARY_FLAGS = -dynamiclib
	else
		SHARED_LIB_EXT = .so
		test_LDFLAGS = -Wl,-rpath,"\$$ORIGIN/../"
		ifeq ($(shell $(CC) --version 2>&1 | grep -o clang),clang)
			OS_SHARED_LIBRARY_FLAGS = -Wl,--export-dynamic
		else
			# check for RedHat GCC version string
			ifeq ($(if $(shell $(CC) --version 2>&1 | grep -o GCC),gcc,$(shell $(CC) --version 2>&1 | grep -o gcc)),gcc)
				OS_SHARED_LIBRARY_FLAGS = -rdynamic
			endif
		endif
	endif
endif

# check C compiler repoline using version information
ifeq ($(shell $(CC) --version 2>&1 | grep -o clang),clang)
	# check for retpoline support (clang)
	ifeq ($(shell $(CC) -Werror \
			-mretpoline -mretpoline-external-thunk \
			-E -x c /dev/null -o /dev/null >/dev/null 2>&1 && echo $$? || echo $$?),0)
RETPOLINE_CFLAGS := -mretpoline -mretpoline-external-thunk -fcf-protection=full
RETPOLINE_THUNK_INLINE_CFLAGS := -mretpoline -mno-retpoline-external-thunk -fcf-protection=full
RETPOLINE_THUNK_EXTERN_CFLAGS := -mretpoline -mretpoline-external-thunk -fcf-protection=full
	else
$(error Error: $(CC) (clang) does not support retpoline!)
	endif
else
	# check for RedHat/Ubuntu GCC version string
	ifeq ($(if $(shell $(CC) --version 2>&1 | grep -o GCC),gcc,$(if $(shell $(CC) -v 2>&1 | grep -o gcc),gcc,$(shell $(CC) --version 2>&1 | grep -o gcc))),gcc)
		# check for retpoline support (gcc)
		ifeq ($(shell $(CC) -Werror \
				-mindirect-branch=thunk-extern -mindirect-branch-register \
				-E -x c /dev/null -o /dev/null >/dev/null 2>&1 && echo $$? || echo $$?),0)
RETPOLINE_CFLAGS := -mindirect-branch=thunk-extern -mindirect-branch-register -fcf-protection=none
RETPOLINE_THUNK_INLINE_CFLAGS := -mindirect-branch=thunk-inline -mindirect-branch-register -fcf-protection=none
RETPOLINE_THUNK_EXTERN_CFLAGS := -mindirect-branch=thunk-extern -mindirect-branch-register -fcf-protection=none
			# check for Intel Skylake+ from model 78 (cpu family 6 extended model 4 and model e)
			ifeq ($(shell [ -e "/proc/cpuinfo" ] && echo cpuinfo),cpuinfo)
				ifeq ($(shell cat /proc/cpuinfo | grep vendor_id | head -n 1 | grep GenuineIntel > /dev/null && echo "GenuineIntel"),GenuineIntel)
					ifeq ($(shell [ "$$(cat /proc/cpuinfo | grep 'cpu family[[:space:]]*: 6' > /dev/null && \
						cat /proc/cpuinfo | grep '^model[[:space:]]*:' | head -n 1 | awk '{ print $$3; }')" -ge "78" ] && \
						echo "skylake+"),skylake+)
RETPOLINE_CFLAGS += -mfunction-return=thunk-extern
RETPOLINE_THUNK_INLINE_CFLAGS += -mfunction-return=thunk-inline
RETPOLINE_THUNK_EXTERN_CFLAGS += -mfunction-return=thunk-extern
					endif
				endif
			endif
		else
$(error Error: $(CC) (gcc) does not support retpoline!)
		endif
	# found unexpected C compiler
	else
$(error Error: Detected unexpected C compiler (unknown retpoline support)!)
	endif
endif

OPENCL_KERNEL_OBJ:=$(patsubst %.opencl, %_opencl.c, $(wildcard kernels/*.opencl))
CLECC_OBJ:=$(patsubst %.c,%.o,$(wildcard *.c))

.PHONY: all test debug-test clean

all:
	@$(MAKE) -C kernels
	@$(MAKE) libclecc$(SHARED_LIB_EXT)

%.o: %.c
	$(CC) $(RETPOLINE_THUNK_INLINE_CFLAGS) $(CFLAGS) -c -fPIC -o $@ $<

# "-Wl,--nxcompat" specified to enable Windows DEP for the executable code
ifeq ($(OS),Windows_NT)
libclecc$(SHARED_LIB_EXT): $(CLECC_OBJ) $(OPENCL_KERNEL_OBJ)
	$(CC) $(RETPOLINE_THUNK_EXTERN_CFLAGS) $(CFLAGS) -Wl,--nxcompat -shared -fPIC -o libclecc$(SHARED_LIB_EXT) $(OPENCL_KERNEL_OBJ) $(CLECC_OBJ) -l OpenCL -l pthread
else
libclecc$(SHARED_LIB_EXT): $(CLECC_OBJ) $(OPENCL_KERNEL_OBJ)
	$(CC) $(RETPOLINE_THUNK_EXTERN_CFLAGS) $(CFLAGS) $(OS_SHARED_LIBRARY_FLAGS) -shared -fPIC -o libclecc$(SHARED_LIB_EXT) $(OPENCL_KERNEL_OBJ) $(CLECC_OBJ) -l OpenCL -l pthread
endif

test: test/opencl_edac_test

test/opencl_edac_test: libclecc$(SHARED_LIB_EXT)
	$(CC) $(CFLAGS) $(test_LDFLAGS) -I ../include -o $@ test/opencl_edac_test.c -l OpenCL -l pthread -L ./ -l clecc
	@$@

debug-test: libclecc$(SHARED_LIB_EXT)
	$(CC) -g -fsanitize=address $(RETPOLINE_THUNK_EXTERN_CFLAGS) $(CFLAGS) $(test_LDFLAGS) -I ../include -o test/opencl_edac_test test/opencl_edac_test.c -l OpenCL -l pthread -L ./ -l clecc -l asan
	env ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0 test/opencl_edac_test

clean:
	@-$(MAKE) -C kernels clean
	-rm -f *.o
	-rm -f *$(SHARED_LIB_EXT)
	-rm -f test/opencl_edac_test
