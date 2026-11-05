TORCH=0
CXX=g++
CXXFLAGS=-fopenmp -std=c++20 -g -O0 -DDEBUG -Wno-unused-result -DTORCH=$(TORCH)
BLAS=-lblas
LAPACK=-llapack
LDFLAGS=-fopenmp
LIBS=-lm

#
# Use optimized flags and libraries on M2
#
ifeq ($(patsubst m3login%,m3login,$(shell hostname)),m3login)
	CXX=g++
	CXXFLAGS=-fopenmp -std=c++20 -march=haswell -g -O2 -DDEBUG-DMARRAY_ENABLE_ASSERTS -DTORCH=$(TORCH) -DPROFILE
	BLAS=-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core
	LAPACK=
	TORCH_DIR=~/group/apps/libtorch
else
#
# Otherwise find libtorch wherever it may be
#
	TORCH_DIR=$(HOME)/libtorch
endif

CXXFLAGS+=-I../common
CXXFLAGS+=-I../docopt.cpp/

ifeq ($(TORCH),1)
	CXXFLAGS+=-I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include
	LIBS+=-L$(TORCH_DIR)/lib -Wl,-rpath,$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10
endif

#Hack for macOS "g++"
IS_APPLE=$(shell g++ --version 2>/dev/null | grep -q Apple && echo yes || echo no)
ifeq ($(IS_APPLE),yes)
	CXXFLAGS:=$(subst -fopenmp,,$(CXXFLAGS))
	LDFLAGS:=$(subst -fopenmp,,$(LDFLAGS))
endif

exe:=../bin/pivoting
src:=$(shell find . -iname "*.cxx") $(shell find ../common -iname "*.cxx") $(shell find ../docopt.cpp -iname "*.cxx")
objs:=$(src:.cxx=.o)
deps:=$(join $(addsuffix .deps/,$(dir $(objs))),$(notdir $(objs:.o=.d)))

.PHONY: all
all: $(exe)

-include $(deps)

.PHONY: clean
clean:
	rm -f $(objs) $(exe)

$(exe): $(objs)
	@mkdir -p $(dir $(exe))
	$(CXX) $(LDFLAGS) -o $(exe) $(objs) $(BLAS) $(LAPACK) $(LIBS)

%.o: %.cxx
	@mkdir -p $(dir $@).deps
	$(CXX) $(CXXFLAGS) -MT $@ -MP -MMD -MF $(dir $@).deps/$(notdir $(@:.o=.d)) -o $@ -c $<

