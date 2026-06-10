CC=clang
TABLEGEN=../../cablegen.so
CCFLAGS=-I../../api/
FILES=main.c
CCFLAGS_DBG=-lasan -lubsan
LIBTYPE=static
BUILD_OPENCL=0

ifeq ($(PLATFORM),windows)
CC=x86_64-w64-mingw32-gcc
endif
ifeq ($(PLATFORM),linux)
CC=clang
endif

ifeq ($(BUILD),debug)
CCFLAGS += $(CCFLAGS_DBG)
endif


ifeq (, $(shell which $(CC)))
$(error "No CC in $(PATH), consider changing CC in the makefile for your operating system")
endif

ifeq ($(LIBTYPE),dynamic)
CCFLAGS += -DDYNAMIC
endif
ifeq ($(LIBTYPE),static)
ifneq ($(BUILD_OPENCL),1)
CCFLAGS += -static
endif
endif

ifeq ($(BUILD_OPENCL),1)
OPENCL_LDFLAGS = -lOpenCL
endif

.PHONY: all

all: main.c
	@echo [CC] CLI
	@$(CC) main.c parse.c -o cli $(TABLEGEN) $(CCFLAGS) $(OPENCL_LDFLAGS)

clean:
	@rm -f cli
