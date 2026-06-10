FRONTENDS=cli/
BUILD=debug
PLATFORM=linux
LIBTYPE=static
BUILD_OPENCL=0

.PHONY: all clean 

all:
	@make -C cli/ all BUILD=$(BUILD) PLATFORM=$(PLATFORM) LIBTYPE=$(LIBTYPE) BUILD_OPENCL=$(BUILD_OPENCL)

clean:
	@make -C cli/ clean
