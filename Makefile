CC=clang
CCFLAGS_SHARED = -Wall -Wpedantic -Wextra -Wuninitialized -O2 -pthread -fno-strict-aliasing -std=c23 -Wno-unused-parameter\
		 -I./inc/ -I./inc/external -Wno-bitwise-instead-of-logical
CCFLAGS= -g -pg -pthread -fno-strict-aliasing -std=c23 -DDBG -fsanitize=address,undefined
CCFLAGS_PROD=-DPROD -Wno-format -DNOERRCHECK -march=native -ffast-math -ffp-contract=fast
CCFLAGS_BENCH=-DPROD -Wno-format -DNOERRCHECK -DBENCH
CCFLAGS_NOSANITIZE = -g -pg -pthread -fno-strict-aliasing -std=c23 -DDBG
CCFLAGS_E = $(CCFLAGS_PROD) -E

LIB_FILE=cablegen.so
BUILD=prod
PLATFORM=linux
LDFLAGS=-flto
LIBTYPE=static
BUILD_OPENCL=0

ifeq ($(BUILD_OPENCL),1)
CCFLAGS_SHARED += -DUSE_OPENCL
LDFLAGS += -lOpenCL
endif

ifeq ($(BUILD),e)
FILES=$(addsuffix .c,$(addprefix ppc/,$(notdir $(basename $(wildcard src/*.c)))))
else
FILES=$(addsuffix .o,$(addprefix build/,$(notdir $(basename $(wildcard src/*.c))))) build/ini.o 
endif

CCFLAGS_SHARED += -DVERSION=$$(git describe --tags --always --dirty)
.PHONY: all clean default gen_conf frontends bench bench_full

ifeq ($(PLATFORM),windows)
CC=x86_64-w64-mingw32-gcc
LDFLAGS += -lm
endif
ifeq ($(PLATFORM),linux)
LDFLAGS += -lm -lc # I'm not sure why only clang needs an explicit -lc
endif

ifeq ($(LIBTYPE),dynamic)
LDFLAGS += -fpic
CABLEGEN_LDFLAGS = -shared
FILES += build/logging.o
endif
ifeq ($(LIBTYPE),static)
CABLEGEN_LDFLAGS = -r
LDFLAGS += -fpie
ifeq ($(BUILD),debug)
$(error Debug mode doesn't work with static linkage! (Asan will complain))
endif
endif

ifeq ($(BUILD),prod)
CCFLAGS = $(CCFLAGS_PROD)
endif
ifeq ($(BUILD),bench)
CCFLAGS = $(CCFLAGS_BENCH)
endif
ifeq ($(BUILD),nosanitize)
CCFLAGS = $(CCFLAGS_NOSANITIZE)
endif
ifeq ($(BUILD),e)
CCFLAGS = $(CCFLAGS_E)
endif

ifeq (, $(shell which $(CC)))
$(error "No CC in $(PATH), consider changing CC in the makefile for your operating system")
endif

# build cablegen.so, which the frontends depend on
default: $(FILES) $(LIB_FILE)
	@mkdir -p build/

all: $(FILES) $(LIB_FILE)
	@mkdir -p build/
	@make gen_conf
	@make frontends

frontends:
	@echo [frontends]
	@make -C frontends/ all PLATFORM=$(PLATFORM) BUILD=$(BUILD) LIBTYPE=$(LIBTYPE) BUILD_OPENCL=$(BUILD_OPENCL)

gen_conf:
	@echo [gen_conf]
	@make -C gen_conf/ all PLATFORM=$(PLATFORM)

BENCH_SRCS = $(wildcard src/*.c) $(wildcard inc/*.h) $(wildcard frontends/cli/*.c) $(wildcard frontends/cli/*.h)

bench_cpu: $(BENCH_SRCS)
	@echo [BENCH] building CPU binary
	@$(MAKE) clean > /dev/null
	@$(MAKE) BUILD=bench BUILD_OPENCL=0 > /dev/null
	@$(MAKE) -C frontends/ all BUILD=bench LIBTYPE=static BUILD_OPENCL=0 > /dev/null
	@cp frontends/cli/cli $@

bench_gpu: $(BENCH_SRCS)
	@echo [BENCH] building GPU binary
	@$(MAKE) clean > /dev/null
	@$(MAKE) BUILD=bench BUILD_OPENCL=1 > /dev/null
	@$(MAKE) -C frontends/ all BUILD=bench LIBTYPE=static BUILD_OPENCL=1 > /dev/null
	@cp frontends/cli/cli $@

bench: bench_cpu bench_gpu
	@bash bench_cpu_vs_gpu.sh

bench_full: bench_cpu bench_gpu
	@bash bench_cpu_vs_gpu.sh 300

clean: 
	@rm -f build/*
	@make -C gen_conf/ clean
	@make -C frontends/ clean

build/ini.o: src/external/ini.c 
	@echo [CC] $@
	@$(CC) $< $(CCFLAGS_SHARED) $(CCFLAGS) -c -fpic -o $@

build/logging.o: src/dynamic/logging.c 
	@echo [CC] $@
	@$(CC) $< $(CCFLAGS_SHARED) $(CCFLAGS) -c -fpic -o $@

build/ui.o: src/ui.c 
	@echo [CC] $@
	@$(CC) $< $(CCFLAGS_SHARED) $(CCFLAGS) -Wno-missing-field-initializers -Wno-missing-braces -c -fpic -o $@

build/%.o: src/%.c 
	@echo [CC] $@
	@$(CC) $< $(CCFLAGS_SHARED) $(CCFLAGS) -c -fpic -o $@

ppc/ini.c: src/external/ini.c 
	@echo [PPC] $@
	@$(CC) $< $(CCFLAGS_SHARED) $(CCFLAGS) -fpic -o $@

ppc/ui.c: src/ui.c 
	@echo [PPC] $@
	@$(CC) $< $(CCFLAGS_SHARED) $(CCFLAGS) -Wno-missing-field-initializers -Wno-missing-braces -fpic -o $@

ppc/%.c: src/%.c 
	@echo [PPC] $@
	@$(CC) $< $(CCFLAGS_SHARED) $(CCFLAGS) -fpic -o $@


$(LIB_FILE): $(FILES)
ifeq (e,$(BUILD))

else
	@echo [LD] $@
	@ld $(wildcard build/*.o) $(CABLEGEN_LDFLAGS) -o $(LIB_FILE) # TODO correct file ext
endif
