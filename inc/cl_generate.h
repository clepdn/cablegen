#ifndef CL_GENERATE_H
#define CL_GENERATE_H
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include "array.h"

/*
 * When compiled with -DUSE_OPENCL (make BUILD_OPENCL=1), generate.c uses
 * a device-resident pipeline (cl_move_dev / cl_spawn_dev) that keeps
 * boards on the GPU across move and spawn, falling back to the
 * host-resident cl_move_boards / cl_spawn_boards, then to CPU pthreads.
 * All entry points are no-ops when USE_OPENCL is not defined.
 */

bool cl_init(void);

// Must be called whenever generate_lut() is (re)invoked; the LUT on the
// device reflects the settings active at upload time.
void cl_upload_luts(void);

void cl_shutdown(void);

dynamic_arr_info cl_move_boards(const uint64_t *boards,
                                size_t n_boards,
                                bool prune,
                                long stsl,
                                long ltc,
                                long smallest_large);

bool cl_spawn_boards(const uint64_t *boards,
                     size_t n_boards,
                     dynamic_arr_info *out_n2,
                     dynamic_arr_info *out_n4);

// Opaque GPU-resident board array. cl_spawn_dev and cl_darr_download
// both consume (release) the handle; use cl_darr_to_host to peek without
// releasing.
typedef struct cl_darr cl_darr;

bool cl_move_dev(const uint64_t *boards,
                 size_t n,
                 bool prune,
                 long stsl, long ltc, long smallest_large,
                 cl_darr **out);

// Consumes `in` regardless of outcome. Returns false on SPAWN_SLOTS
// overflow (boards with more empty tiles than the kernel slot count) as
// well as on any other failure; caller must fall back to host spawn.
bool cl_spawn_dev(cl_darr *in,
                  cl_darr **out_n2,
                  cl_darr **out_n4);

// Downloads and releases `in`.
bool cl_darr_download(cl_darr *in, dynamic_arr_info *out);

// Downloads without releasing; use when `in` is still needed on device.
bool cl_darr_to_host(cl_darr *in, dynamic_arr_info *out);

void cl_darr_release(cl_darr *in);

// Buffers only grow during a run, so the value at shutdown is peak VRAM.
size_t cl_vram_bytes(void);

#endif
