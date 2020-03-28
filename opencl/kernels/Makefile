# SPDX-License-Identifier: GPL-3.0-only

.PHONY: all clean

OPENCL_C:=$(patsubst %.opencl, %_opencl.c, $(wildcard *.opencl))

all:
	$(MAKE) $(OPENCL_C)

%_opencl.c: %.opencl
	perl opencl_to_c.pl $< > $@

clean:
	-rm -f *_opencl.c
