# SPDX-License-Identifier: GPL-3.0-only

.PHONY: all clean

PTX:=$(patsubst %.cu, %.ptx, $(wildcard *.cu))
PTX_C:=$(patsubst %.cu, %_cuda.c, $(wildcard *.cu))

all:
	$(MAKE) $(PTX)
	$(MAKE) $(PTX_C)

%.ptx: %.cu
	nvcc -ptx -Xptxas -dlcm=cg -o $@ $<

%_cuda.c: %.ptx
	perl ptx_to_c.pl $< > $@

clean:
	-rm -f *.ptx
	-rm -f *_cuda.c
