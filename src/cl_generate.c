#include "cl_generate.h"

#ifndef USE_OPENCL

// CPU build: stubs so the translation unit is trivially empty.
#include <stddef.h>
bool cl_init(void) { return false; }
void cl_upload_luts(void) { }
void cl_shutdown(void) { }
dynamic_arr_info cl_move_boards(const uint64_t *boards,
                                size_t n_boards,
                                bool prune,
                                long stsl,
                                long ltc,
                                long smallest_large) {
    (void)boards; (void)n_boards; (void)prune;
    (void)stsl; (void)ltc; (void)smallest_large;
    dynamic_arr_info r = { .valid = false, .bp = NULL, .sp = NULL, .size = 0 };
    return r;
}
bool cl_spawn_boards(const uint64_t *boards,
                     size_t n_boards,
                     dynamic_arr_info *out_n2,
                     dynamic_arr_info *out_n4) {
    (void)boards; (void)n_boards;
    if (out_n2) *out_n2 = (dynamic_arr_info){ .valid = false, .bp = NULL, .sp = NULL, .size = 0 };
    if (out_n4) *out_n4 = (dynamic_arr_info){ .valid = false, .bp = NULL, .sp = NULL, .size = 0 };
    return false;
}

bool cl_move_dev(const uint64_t *boards, size_t n, bool prune,
                 long stsl, long ltc, long smallest_large, cl_darr **out) {
    (void)boards; (void)n; (void)prune; (void)stsl; (void)ltc; (void)smallest_large;
    if (out) *out = NULL;
    return false;
}
bool cl_spawn_dev(cl_darr *in, cl_darr **out_n2, cl_darr **out_n4) {
    (void)in;
    if (out_n2) *out_n2 = NULL;
    if (out_n4) *out_n4 = NULL;
    return false;
}
bool cl_darr_download(cl_darr *in, dynamic_arr_info *out) {
    (void)in;
    if (out) *out = (dynamic_arr_info){ .valid = false, .bp = NULL, .sp = NULL, .size = 0 };
    return false;
}
bool cl_darr_to_host(cl_darr *in, dynamic_arr_info *out) {
    (void)in;
    if (out) *out = (dynamic_arr_info){ .valid = false, .bp = NULL, .sp = NULL, .size = 0 };
    return false;
}
void cl_darr_release(cl_darr *in) { (void)in; }
size_t cl_vram_bytes(void) { return 0; }

#else /* USE_OPENCL */

#include "array.h"
#include "bench.h"
#include "board.h"
#define LOG_H_ENUM_PREFIX_
#define LOG_H_NAMESPACE_
#include "logging.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

// Forward decls for the verify path
#include "board.h"
bool prune_board(const uint64_t board, const long stsl, const long ltc, const long smallest_large);

static bool          g_cl_initialised = false;
static bool          g_cl_init_tried  = false;
static cl_context    g_ctx     = NULL;
static cl_device_id  g_dev     = NULL;
static cl_command_queue g_q    = NULL;
static cl_program    g_prog    = NULL;
static cl_kernel     g_kern    = NULL;
static cl_kernel     g_kern_spawn = NULL;

// Persistent LUT buffers (uploaded once per generate_lut() call)
static cl_mem g_buf_move_lut   = NULL; // ushort[2*65536]
static cl_mem g_buf_locked_lut = NULL; // uchar[2*65536]
static bool   g_luts_uploaded  = false;

/* Persistent batch buffers, grown lazily. Allocating/freeing cl_mem and
 * host scratch on every call is expensive on NVIDIA (DMA pinning,
 * kernel mappings). Reusing buffers across layers saves a chunk of
 * 'sys' time. */
typedef struct {
    cl_mem    dev;
    void     *host;
    size_t    bytes;
    cl_mem_flags flags;
} cl_buf;

static cl_buf g_move_in       = { 0 };
static cl_buf g_move_out      = { 0 };
static cl_buf g_move_counts   = { 0 };
static cl_buf g_spawn_in      = { 0 };
static cl_buf g_spawn_o2      = { 0 };
static cl_buf g_spawn_o4      = { 0 };
static cl_buf g_spawn_counts  = { 0 };

/* Device-resident path buffers. DEV_SLOTS large ulong slots
 * ping-pong through compact/sort/uniq for both move and spawn. */
#define DEV_SLOTS       4
#define MAX_SCAN_LEVELS 4
static cl_buf g_dev_slots[DEV_SLOTS]; // ulong[N*4] each
static cl_buf g_dev_input   = { 0 };  // ulong[N] -- initial upload
static cl_buf g_dev_counts  = { 0 };  // uchar[N] -- shared counts
static cl_buf g_dev_prefix  = { 0 };  // uint[N] -- compact prefix
static cl_buf g_dev_uflags  = { 0 };  // uint[N] -- uniq flags / prefix
static cl_buf g_dev_hist    = { 0 };  // uint[256 * n_blocks_max]
static cl_buf g_dev_scan_l[MAX_SCAN_LEVELS]; // per-recursion-level block sums
static cl_buf g_dev_overflow = { 0 }; // uint[1] -- spawn overflow flag

#define WG_SIZE      256u
#define RADIX_BINS   256u
#define WG_SIZE_SMALL 64u
#define SPAWN_SLOTS    4u

// Slot allocator: a tiny RAII-ish bitmask of in-use device slots.
static uint32_t g_slot_busy = 0;

static int slot_acquire(void) {
    for (int i = 0; i < DEV_SLOTS; i++) {
        if (!(g_slot_busy & (1u << i))) {
            g_slot_busy |= (1u << i);
            return i;
        }
    }
    return -1;
}
static void slot_release(int s) {
    if (s >= 0 && s < DEV_SLOTS) g_slot_busy &= ~(1u << s);
}

// Concrete definition of the opaque handle declared in cl_generate.h.
struct cl_darr {
    int    slot;
    size_t n;
};

static cl_kernel g_kern_scan_inc_uint  = NULL;
static cl_kernel g_kern_scan_inc_uchar = NULL;
static cl_kernel g_kern_scan_add       = NULL;
static cl_kernel g_kern_compact        = NULL;
static cl_kernel g_kern_compact2       = NULL;
static cl_kernel g_kern_clamp_counts   = NULL;
static cl_kernel g_kern_radix_hist     = NULL;
static cl_kernel g_kern_radix_scatter  = NULL;
static cl_kernel g_kern_uniq_flag      = NULL;
static cl_kernel g_kern_uniq_scatter   = NULL;

static bool cl_buf_ensure(cl_buf *b, size_t bytes, cl_mem_flags flags, bool want_host) {
    cl_int err;
    if (b->bytes >= bytes && b->dev) return true;
    if (b->dev) { clReleaseMemObject(b->dev); b->dev = NULL; }
    if (b->host) { free(b->host); b->host = NULL; }
    b->dev = clCreateBuffer(g_ctx, flags, bytes, NULL, &err);
    if (err != CL_SUCCESS) { b->dev = NULL; return false; }
    if (want_host) {
        b->host = malloc(bytes);
        if (!b->host) { clReleaseMemObject(b->dev); b->dev = NULL; return false; }
    }
    b->bytes = bytes;
    b->flags = flags;
    return true;
}

static void cl_buf_release(cl_buf *b) {
    if (b->dev) clReleaseMemObject(b->dev);
    if (b->host) free(b->host);
    b->dev = NULL; b->host = NULL; b->bytes = 0;
}

#define CL_CHECK(call, ret_on_fail) do {                                   \
    cl_int _err = (call);                                                  \
    if (_err != CL_SUCCESS) {                                              \
        logf_out("OpenCL call failed (%d) at %s:%d: " #call,               \
                 LOG_ERROR, (int)_err, __FILE__, __LINE__);                \
        return ret_on_fail;                                                \
    }                                                                      \
} while(0)

static const char *const k_search_paths[] = {
    "src/cl/generate.cl",
    "./generate.cl",
    "/usr/local/share/cablegen/generate.cl",
    "/usr/share/cablegen/generate.cl",
    NULL,
};

static char *slurp_kernel(size_t *out_len) {
    const char *override = getenv("CABLEGEN_CL_PATH");
    const char *path = NULL;
    FILE *fp = NULL;
    if (override && *override) {
        fp = fopen(override, "rb");
        if (fp) path = override;
    }
    for (int i = 0; !fp && k_search_paths[i]; i++) {
        fp = fopen(k_search_paths[i], "rb");
        if (fp) { path = k_search_paths[i]; break; }
    }
    if (!fp) {
        log_out("Could not locate generate.cl (set CABLEGEN_CL_PATH).", LOG_ERROR);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    rewind(fp);
    if (sz <= 0) { fclose(fp); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        free(buf); fclose(fp); return NULL;
    }
    buf[sz] = '\0';
    fclose(fp);
    logf_out("Loaded OpenCL kernel from %s (%ld bytes)", LOG_INFO, path, sz);
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

bool cl_init(void) {
    if (g_cl_initialised) return true;
    if (g_cl_init_tried)  return false;
    g_cl_init_tried = true;

    cl_int err;
    cl_platform_id platforms[8];
    cl_uint n_plat = 0;
    err = clGetPlatformIDs(8, platforms, &n_plat);
    if (err != CL_SUCCESS || n_plat == 0) {
        log_out("No OpenCL platforms available; falling back to CPU.", LOG_WARN);
        return false;
    }

    // Prefer GPU devices; fall back to any.
    cl_device_id dev = NULL;
    for (cl_uint i = 0; i < n_plat && !dev; i++) {
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &dev, NULL)
            != CL_SUCCESS) dev = NULL;
    }
    for (cl_uint i = 0; i < n_plat && !dev; i++) {
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1, &dev, NULL)
            != CL_SUCCESS) dev = NULL;
    }
    if (!dev) {
        log_out("No OpenCL devices found.", LOG_WARN);
        return false;
    }
    g_dev = dev;

    char dev_name[256] = {0};
    clGetDeviceInfo(g_dev, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    logf_out("OpenCL device: %s", LOG_INFO, dev_name);

    g_ctx = clCreateContext(NULL, 1, &g_dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) { log_out("clCreateContext failed.", LOG_ERROR); return false; }

    /* OpenCL 2.0+ deprecated clCreateCommandQueue but we keep it for
     * 1.2 portability; the deprecation warning is harmless. */
#ifdef CL_VERSION_2_0
    g_q = clCreateCommandQueueWithProperties(g_ctx, g_dev, NULL, &err);
#else
    g_q = clCreateCommandQueue(g_ctx, g_dev, 0, &err);
#endif
    if (err != CL_SUCCESS) { log_out("clCreateCommandQueue failed.", LOG_ERROR); return false; }

    size_t src_len = 0;
    char *src = slurp_kernel(&src_len);
    if (!src) return false;

    const char *srcs[1] = { src };
    const size_t lens[1] = { src_len };
    g_prog = clCreateProgramWithSource(g_ctx, 1, srcs, lens, &err);
    free(src);
    if (err != CL_SUCCESS) { log_out("clCreateProgramWithSource failed.", LOG_ERROR); return false; }

    err = clBuildProgram(g_prog, 1, &g_dev, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(g_prog, g_dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char *log = malloc(log_sz + 1);
        if (log) {
            clGetProgramBuildInfo(g_prog, g_dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
            log[log_sz] = '\0';
            logf_out("OpenCL build failed:\n%s", LOG_ERROR, log);
            free(log);
        }
        return false;
    }

    g_kern = clCreateKernel(g_prog, "move_kernel", &err);
    if (err != CL_SUCCESS) { log_out("clCreateKernel(move_kernel) failed.", LOG_ERROR); return false; }

    g_kern_spawn = clCreateKernel(g_prog, "spawn_kernel", &err);
    if (err != CL_SUCCESS) { log_out("clCreateKernel(spawn_kernel) failed.", LOG_ERROR); return false; }

    struct { cl_kernel *slot; const char *name; } kerns[] = {
        { &g_kern_scan_inc_uint,  "scan_inc_uint_kernel"  },
        { &g_kern_scan_inc_uchar, "scan_inc_uchar_kernel" },
        { &g_kern_scan_add,       "scan_add_kernel"       },
        { &g_kern_compact,        "compact_kernel"        },
        { &g_kern_compact2,       "compact2_kernel"       },
        { &g_kern_clamp_counts,   "clamp_counts_kernel"   },
        { &g_kern_radix_hist,     "radix_hist_kernel"     },
        { &g_kern_radix_scatter,  "radix_scatter_kernel"  },
        { &g_kern_uniq_flag,      "uniq_flag_kernel"      },
        { &g_kern_uniq_scatter,   "uniq_scatter_kernel"   },
    };
    for (size_t i = 0; i < sizeof(kerns)/sizeof(kerns[0]); i++) {
        *kerns[i].slot = clCreateKernel(g_prog, kerns[i].name, &err);
        if (err != CL_SUCCESS) {
            logf_out("clCreateKernel(%s) failed (%d).", LOG_ERROR, kerns[i].name, (int)err);
            return false;
        }
    }

    g_cl_initialised = true;
    cl_upload_luts();
    return true;
}

void cl_shutdown(void) {
    cl_buf_release(&g_move_in);
    cl_buf_release(&g_move_out);
    cl_buf_release(&g_move_counts);
    cl_buf_release(&g_spawn_in);
    cl_buf_release(&g_spawn_o2);
    cl_buf_release(&g_spawn_o4);
    cl_buf_release(&g_spawn_counts);
    for (int i = 0; i < DEV_SLOTS; i++) cl_buf_release(&g_dev_slots[i]);
    cl_buf_release(&g_dev_input);
    cl_buf_release(&g_dev_counts);
    cl_buf_release(&g_dev_prefix);
    cl_buf_release(&g_dev_uflags);
    cl_buf_release(&g_dev_hist);
    for (int i = 0; i < MAX_SCAN_LEVELS; i++) cl_buf_release(&g_dev_scan_l[i]);
    cl_buf_release(&g_dev_overflow);
    g_slot_busy = 0;
    if (g_buf_move_lut)   clReleaseMemObject(g_buf_move_lut);
    if (g_buf_locked_lut) clReleaseMemObject(g_buf_locked_lut);
    if (g_kern) clReleaseKernel(g_kern);
    if (g_kern_spawn) clReleaseKernel(g_kern_spawn);
    cl_kernel *new_kerns[] = {
        &g_kern_scan_inc_uint, &g_kern_scan_inc_uchar, &g_kern_scan_add,
        &g_kern_compact, &g_kern_compact2, &g_kern_clamp_counts,
        &g_kern_radix_hist, &g_kern_radix_scatter,
        &g_kern_uniq_flag, &g_kern_uniq_scatter,
    };
    for (size_t i = 0; i < sizeof(new_kerns)/sizeof(new_kerns[0]); i++) {
        if (*new_kerns[i]) clReleaseKernel(*new_kerns[i]);
        *new_kerns[i] = NULL;
    }
    if (g_prog) clReleaseProgram(g_prog);
    if (g_q)    clReleaseCommandQueue(g_q);
    if (g_ctx)  clReleaseContext(g_ctx);
    g_buf_move_lut = g_buf_locked_lut = NULL;
    g_kern = NULL; g_kern_spawn = NULL; g_prog = NULL; g_q = NULL; g_ctx = NULL; g_dev = NULL;
    g_luts_uploaded = false;
    g_cl_initialised = false;
    g_cl_init_tried = false;
}

void cl_upload_luts(void) {
    if (!g_cl_initialised) {
        // Will be retried at the end of cl_init().
        g_luts_uploaded = false;
        return;
    }
    cl_int err;
    if (g_buf_move_lut)   { clReleaseMemObject(g_buf_move_lut);   g_buf_move_lut = NULL; }
    if (g_buf_locked_lut) { clReleaseMemObject(g_buf_locked_lut); g_buf_locked_lut = NULL; }

    /* _move_lut is uint16_t[2][65536]; flatten as a single contiguous host
     * region (it already is one, since C arrays are row-major). */
    g_buf_move_lut = clCreateBuffer(
        g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint16_t) * 2 * 65536, (void*)_move_lut, &err);
    if (err != CL_SUCCESS) {
        log_out("Failed to upload move LUT.", LOG_ERROR);
        return;
    }

    /* _locked_lut is bool[2][65536]. sizeof(bool) is implementation-defined
     * (typically 1) but to guarantee a uchar layout on the device we copy
     * to a contiguous uchar buffer. */
    uint8_t *flat = malloc(2 * 65536);
    if (!flat) { log_out("OOM packing locked LUT.", LOG_ERROR); return; }
    for (int d = 0; d < 2; d++)
        for (int i = 0; i < 65536; i++)
            flat[d * 65536 + i] = _locked_lut[d][i] ? 1 : 0;

    g_buf_locked_lut = clCreateBuffer(
        g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        2 * 65536, flat, &err);
    free(flat);
    if (err != CL_SUCCESS) {
        log_out("Failed to upload locked LUT.", LOG_ERROR);
        return;
    }
    g_luts_uploaded = true;
}

dynamic_arr_info cl_move_boards(const uint64_t *boards,
                                size_t n_boards,
                                bool prune,
                                long stsl,
                                long ltc,
                                long smallest_large)
{
    dynamic_arr_info bad = { .valid = false, .bp = NULL, .sp = NULL, .size = 0 };
    if (!cl_init() || !g_luts_uploaded || n_boards == 0) {
        if (n_boards == 0) {
            dynamic_arr_info empty = init_darr(false, 0);
            return empty;
        }
        return bad;
    }

    /* Chunk to keep peak GPU memory bounded. The output buffer is 4x the
     * input size in uint64_t, so for very large layers we batch. */
    const size_t MAX_BATCH = 1u << 22; // 4M input boards -> 128 MiB output
    dynamic_arr_info out = init_darr(false, n_boards); // will grow
    out.sp = out.bp; // push_back appends from sp

    size_t batch_cap = n_boards < MAX_BATCH ? n_boards : MAX_BATCH;

    if (!cl_buf_ensure(&g_move_in,     sizeof(uint64_t) * batch_cap,        CL_MEM_READ_ONLY,  false)) goto fail;
    if (!cl_buf_ensure(&g_move_out,    sizeof(uint64_t) * batch_cap * 4,    CL_MEM_WRITE_ONLY, true))  goto fail;
    if (!cl_buf_ensure(&g_move_counts, sizeof(uint8_t)  * batch_cap,        CL_MEM_WRITE_ONLY, true))  goto fail;

    cl_mem    buf_in     = g_move_in.dev;
    cl_mem    buf_out    = g_move_out.dev;
    cl_mem    buf_counts = g_move_counts.dev;
    uint64_t *host_out    = (uint64_t*)g_move_out.host;
    uint8_t  *host_counts = (uint8_t *)g_move_counts.host;

    for (size_t off = 0; off < n_boards; off += batch_cap) {
        size_t this_batch = n_boards - off;
        if (this_batch > batch_cap) this_batch = batch_cap;

        if (clEnqueueWriteBuffer(g_q, buf_in, CL_FALSE, 0,
                                 sizeof(uint64_t) * this_batch,
                                 boards + off, 0, NULL, NULL) != CL_SUCCESS) goto fail;

        cl_uint  arg_n     = (cl_uint)this_batch;
        cl_uint  arg_prune = prune ? 1u : 0u;
        cl_long  arg_stsl  = (cl_long)stsl;
        cl_long  arg_ltc   = (cl_long)ltc;
        cl_long  arg_sl    = (cl_long)smallest_large;

        if (clSetKernelArg(g_kern, 0, sizeof(cl_mem), &buf_in)            != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern, 1, sizeof(cl_mem), &buf_out)           != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern, 2, sizeof(cl_mem), &buf_counts)        != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern, 3, sizeof(cl_mem), &g_buf_move_lut)    != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern, 4, sizeof(cl_mem), &g_buf_locked_lut)  != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern, 5, sizeof(cl_uint), &arg_n)            != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern, 6, sizeof(cl_uint), &arg_prune)        != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern, 7, sizeof(cl_long), &arg_stsl)         != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern, 8, sizeof(cl_long), &arg_ltc)          != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern, 9, sizeof(cl_long), &arg_sl)           != CL_SUCCESS) goto fail;

        size_t local = WG_SIZE_SMALL;
        size_t global = ((this_batch + local - 1) / local) * local;
        if (clEnqueueNDRangeKernel(g_q, g_kern, 1, NULL, &global, &local,
                                   0, NULL, NULL) != CL_SUCCESS) goto fail;

        if (clEnqueueReadBuffer(g_q, buf_out, CL_FALSE, 0,
                                sizeof(uint64_t) * this_batch * 4,
                                host_out, 0, NULL, NULL) != CL_SUCCESS) goto fail;
        if (clEnqueueReadBuffer(g_q, buf_counts, CL_TRUE, 0,
                                sizeof(uint8_t) * this_batch,
                                host_counts, 0, NULL, NULL) != CL_SUCCESS) goto fail;

        for (size_t i = 0; i < this_batch; i++) {
            uint8_t c = host_counts[i];
            for (uint8_t k = 0; k < c; k++)
                push_back(&out, host_out[i * 4 + k]);
        }
    }

    // Sort to match the CPU-thread contract (each thread returns sorted).
    qs_sort_h(out.bp, out.sp - out.bp);

    /* Optional CPU cross-check: with CABLEGEN_CL_VERIFY=1 the same input
     * is also run through the CPU move; if the sorted result differs we
     * dump the diff and abort. Kept around because it catches kernel
     * regressions quickly. */
    if (getenv("CABLEGEN_CL_VERIFY")) {
        size_t cap = n_boards * 4;
        uint64_t *cpu = malloc(cap * sizeof(uint64_t));
        size_t cn = 0;
        for (size_t i = 0; i < n_boards; i++) {
            uint64_t orig = boards[i];
            if (prune && prune_board(orig, stsl, ltc, smallest_large)) continue;
            for (int d = 0; d < 4; d++) {
                uint64_t b = orig;
                if (movedir_unstable(&b, d)) {
                    if (prune && prune_board(b, stsl, ltc, smallest_large)) continue;
                    canonicalize_b(&b);
                    cpu[cn++] = b;
                }
            }
        }
        qs_sort_h(cpu, cn);
        size_t gn = out.sp - out.bp;
        if (gn != cn || (gn && memcmp(out.bp, cpu, gn * sizeof(uint64_t)))) {
            logf_out("VERIFY MISMATCH: gpu=%zu cpu=%zu input=%zu",
                     LOG_ERROR, gn, cn, n_boards);
            size_t i = 0, j = 0, shown = 0;
            while ((i < gn || j < cn) && shown < 10) {
                if (j >= cn || (i < gn && out.bp[i] < cpu[j])) {
                    logf_out("  GPU-only: %016lx", LOG_ERROR, out.bp[i]); i++; shown++;
                } else if (i >= gn || (j < cn && cpu[j] < out.bp[i])) {
                    logf_out("  CPU-only: %016lx", LOG_ERROR, cpu[j]); j++; shown++;
                } else { i++; j++; }
            }
            abort();
        }
        free(cpu);
    }
    return out;

fail:
    log_out("OpenCL move dispatch failed; falling back to CPU.", LOG_ERROR);
    // Persistent buffers are owned by the global pool; not freed here.
    destroy_darr(&out);
    return bad;
}

bool cl_spawn_boards(const uint64_t *boards,
                     size_t n_boards,
                     dynamic_arr_info *out_n2,
                     dynamic_arr_info *out_n4)
{
    if (out_n2) *out_n2 = (dynamic_arr_info){ .valid = false, .bp = NULL, .sp = NULL, .size = 0 };
    if (out_n4) *out_n4 = (dynamic_arr_info){ .valid = false, .bp = NULL, .sp = NULL, .size = 0 };
    if (!cl_init() || n_boards == 0) {
        if (n_boards == 0) {
            if (out_n2) *out_n2 = init_darr(false, 0);
            if (out_n4) *out_n4 = init_darr(false, 0);
            return true;
        }
        return false;
    }
    if (!out_n2 || !out_n4) return false;

    /* ~99.9% of cablegen post-move boards have <=4 empties; the rare
     * overflow boards are detected by host (out_counts > SPAWN_SLOTS)
     * and re-run on CPU. 4M inputs -> 128 MiB per output stream, 256 MiB total. */
    const size_t MAX_BATCH = 1u << 22;
    /* Cablegen's typical post-move boards have 1-3 empty tiles (median 1,
     * p99.99 = 5). 4x is generous. */
    dynamic_arr_info n2 = init_darr(false, n_boards * 4);
    dynamic_arr_info n4 = init_darr(false, n_boards * 4);
    n2.sp = n2.bp;
    n4.sp = n4.bp;

    size_t batch_cap = n_boards < MAX_BATCH ? n_boards : MAX_BATCH;

    if (!cl_buf_ensure(&g_spawn_in,     sizeof(uint64_t) * batch_cap,                   CL_MEM_READ_ONLY,  false)) goto fail;
    if (!cl_buf_ensure(&g_spawn_o2,     sizeof(uint64_t) * batch_cap * SPAWN_SLOTS,     CL_MEM_WRITE_ONLY, true))  goto fail;
    if (!cl_buf_ensure(&g_spawn_o4,     sizeof(uint64_t) * batch_cap * SPAWN_SLOTS,     CL_MEM_WRITE_ONLY, true))  goto fail;
    if (!cl_buf_ensure(&g_spawn_counts, sizeof(uint8_t)  * batch_cap,                   CL_MEM_WRITE_ONLY, true))  goto fail;

    cl_mem    buf_in     = g_spawn_in.dev;
    cl_mem    buf_o2     = g_spawn_o2.dev;
    cl_mem    buf_o4     = g_spawn_o4.dev;
    cl_mem    buf_counts = g_spawn_counts.dev;
    uint64_t *host_o2    = (uint64_t*)g_spawn_o2.host;
    uint64_t *host_o4    = (uint64_t*)g_spawn_o4.host;
    uint8_t  *host_counts = (uint8_t*)g_spawn_counts.host;

    for (size_t off = 0; off < n_boards; off += batch_cap) {
        size_t this_batch = n_boards - off;
        if (this_batch > batch_cap) this_batch = batch_cap;

        if (clEnqueueWriteBuffer(g_q, buf_in, CL_FALSE, 0,
                                 sizeof(uint64_t) * this_batch,
                                 boards + off, 0, NULL, NULL) != CL_SUCCESS) goto fail;

        cl_uint arg_n = (cl_uint)this_batch;
        if (clSetKernelArg(g_kern_spawn, 0, sizeof(cl_mem), &buf_in)     != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern_spawn, 1, sizeof(cl_mem), &buf_o2)     != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern_spawn, 2, sizeof(cl_mem), &buf_o4)     != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern_spawn, 3, sizeof(cl_mem), &buf_counts) != CL_SUCCESS) goto fail;
        if (clSetKernelArg(g_kern_spawn, 4, sizeof(cl_uint), &arg_n)     != CL_SUCCESS) goto fail;

        size_t local = WG_SIZE_SMALL;
        size_t global = ((this_batch + local - 1) / local) * local;
        if (clEnqueueNDRangeKernel(g_q, g_kern_spawn, 1, NULL, &global, &local,
                                   0, NULL, NULL) != CL_SUCCESS) goto fail;

        if (clEnqueueReadBuffer(g_q, buf_o2, CL_FALSE, 0,
                                sizeof(uint64_t) * this_batch * SPAWN_SLOTS,
                                host_o2, 0, NULL, NULL) != CL_SUCCESS) goto fail;
        if (clEnqueueReadBuffer(g_q, buf_o4, CL_FALSE, 0,
                                sizeof(uint64_t) * this_batch * SPAWN_SLOTS,
                                host_o4, 0, NULL, NULL) != CL_SUCCESS) goto fail;
        if (clEnqueueReadBuffer(g_q, buf_counts, CL_TRUE, 0,
                                sizeof(uint8_t) * this_batch,
                                host_counts, 0, NULL, NULL) != CL_SUCCESS) goto fail;

        for (size_t i = 0; i < this_batch; i++) {
            uint8_t c = host_counts[i];
            if (c > SPAWN_SLOTS) {
                /* Overflow: kernel only wrote the first SPAWN_SLOTS
                 * slots; redo this board on the host. Tiny relative
                 * cost since these are <0.2% of boards. */
                uint64_t orig = boards[off + i];
                for (int t = 0; t < 16; t++) {
                    if (GET_TILE(orig, t) == 0) {
                        uint64_t tmp;
                        tmp = orig; SET_TILE(tmp, t, 1); canonicalize_b(&tmp);
                        push_back(&n2, tmp);
                        tmp = orig; SET_TILE(tmp, t, 2); canonicalize_b(&tmp);
                        push_back(&n4, tmp);
                    }
                }
            } else {
                for (uint8_t k = 0; k < c; k++) {
                    push_back(&n2, host_o2[i * SPAWN_SLOTS + k]);
                    push_back(&n4, host_o4[i * SPAWN_SLOTS + k]);
                }
            }
        }
    }

    // Match CPU-thread contract: each returns sorted.
    qs_sort_h(n2.bp, n2.sp - n2.bp);
    qs_sort_h(n4.bp, n4.sp - n4.bp);

    if (getenv("CABLEGEN_CL_VERIFY")) {
        size_t cap = n_boards * 16 + 16;
        uint64_t *c2 = malloc(cap * sizeof(uint64_t));
        uint64_t *c4 = malloc(cap * sizeof(uint64_t));
        size_t cn = 0;
        for (size_t i = 0; i < n_boards; i++) {
            uint64_t old = boards[i];
            for (int t = 0; t < 16; t++) {
                if (GET_TILE(old, t) == 0) {
                    uint64_t tmp;
                    tmp = old; SET_TILE(tmp, t, 1); canonicalize_b(&tmp); c2[cn] = tmp;
                    tmp = old; SET_TILE(tmp, t, 2); canonicalize_b(&tmp); c4[cn] = tmp;
                    cn++;
                }
            }
        }
        qs_sort_h(c2, cn);
        qs_sort_h(c4, cn);
        size_t gn2 = n2.sp - n2.bp;
        size_t gn4 = n4.sp - n4.bp;
        if (gn2 != cn || gn4 != cn ||
            (cn && memcmp(n2.bp, c2, cn * sizeof(uint64_t))) ||
            (cn && memcmp(n4.bp, c4, cn * sizeof(uint64_t)))) {
            logf_out("SPAWN VERIFY MISMATCH: gpu=%zu/%zu cpu=%zu input=%zu",
                     LOG_ERROR, gn2, gn4, cn, n_boards);
            abort();
        }
        free(c2);
        free(c4);
    }

    *out_n2 = n2;
    *out_n4 = n4;
    return true;

fail:
    log_out("OpenCL spawn dispatch failed; falling back to CPU.", LOG_ERROR);
    // Persistent buffers are owned by the global pool; not freed here.
    destroy_darr(&n2);
    destroy_darr(&n4);
    return false;
}

// Device-resident path

/* Layout note: the device-resident pipeline never crosses to host between
 * the move kernel and the final download of n2/n4. Each phase uses two or
 * three of the four DEV_SLOTS as scratch + result, freeing them as soon
 * as they're consumed so the next phase can grab them. */

static inline size_t ceil_div_sz(size_t a, size_t b) { return (a + b - 1) / b; }
static inline size_t round_up_sz(size_t a, size_t b) { return ceil_div_sz(a, b) * b; }

// Recursive in-place inclusive scan of a uint array on the device.
static bool cl_scan_uint_inc_dev(cl_mem keys, size_t n, int level) {
    if (n == 0) return true;
    size_t n_blocks = ceil_div_sz(n, WG_SIZE);
    if (level >= MAX_SCAN_LEVELS) {
        log_out("Scan recursion too deep.", LOG_ERROR);
        return false;
    }
    if (!cl_buf_ensure(&g_dev_scan_l[level], sizeof(cl_uint) * n_blocks,
                       CL_MEM_READ_WRITE, false)) return false;
    cl_mem block_sums = g_dev_scan_l[level].dev;

    cl_uint arg_n = (cl_uint)n;
    if (clSetKernelArg(g_kern_scan_inc_uint, 0, sizeof(cl_mem), &keys)       != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_scan_inc_uint, 1, sizeof(cl_mem), &keys)       != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_scan_inc_uint, 2, sizeof(cl_mem), &block_sums) != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_scan_inc_uint, 3, sizeof(cl_uint), &arg_n)     != CL_SUCCESS) return false;
    size_t local = WG_SIZE;
    size_t global = round_up_sz(n, WG_SIZE);
    if (clEnqueueNDRangeKernel(g_q, g_kern_scan_inc_uint, 1, NULL,
                               &global, &local, 0, NULL, NULL) != CL_SUCCESS) return false;

    if (n_blocks <= 1) return true;

    if (!cl_scan_uint_inc_dev(block_sums, n_blocks, level + 1)) return false;

    if (clSetKernelArg(g_kern_scan_add, 0, sizeof(cl_mem), &keys)        != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_scan_add, 1, sizeof(cl_mem), &block_sums)  != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_scan_add, 2, sizeof(cl_uint), &arg_n)      != CL_SUCCESS) return false;
    if (clEnqueueNDRangeKernel(g_q, g_kern_scan_add, 1, NULL,
                               &global, &local, 0, NULL, NULL) != CL_SUCCESS) return false;
    return true;
}

// Inclusive scan of uchar input into uint output.
static bool cl_scan_uchar_inc_dev(cl_mem in_uc, cl_mem out_ui, size_t n) {
    if (n == 0) return true;
    size_t n_blocks = ceil_div_sz(n, WG_SIZE);
    if (!cl_buf_ensure(&g_dev_scan_l[0], sizeof(cl_uint) * n_blocks,
                       CL_MEM_READ_WRITE, false)) return false;
    cl_mem block_sums = g_dev_scan_l[0].dev;

    cl_uint arg_n = (cl_uint)n;
    if (clSetKernelArg(g_kern_scan_inc_uchar, 0, sizeof(cl_mem), &in_uc)     != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_scan_inc_uchar, 1, sizeof(cl_mem), &out_ui)    != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_scan_inc_uchar, 2, sizeof(cl_mem), &block_sums)!= CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_scan_inc_uchar, 3, sizeof(cl_uint), &arg_n)    != CL_SUCCESS) return false;
    size_t local = WG_SIZE;
    size_t global = round_up_sz(n, WG_SIZE);
    if (clEnqueueNDRangeKernel(g_q, g_kern_scan_inc_uchar, 1, NULL,
                               &global, &local, 0, NULL, NULL) != CL_SUCCESS) return false;

    if (n_blocks <= 1) return true;
    if (!cl_scan_uint_inc_dev(block_sums, n_blocks, 1)) return false;

    if (clSetKernelArg(g_kern_scan_add, 0, sizeof(cl_mem), &out_ui)      != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_scan_add, 1, sizeof(cl_mem), &block_sums)  != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_scan_add, 2, sizeof(cl_uint), &arg_n)      != CL_SUCCESS) return false;
    if (clEnqueueNDRangeKernel(g_q, g_kern_scan_add, 1, NULL,
                               &global, &local, 0, NULL, NULL) != CL_SUCCESS) return false;
    return true;
}

// Blocking read of one uint from buf[idx].
static bool cl_read_last_u32(cl_mem buf, size_t idx, cl_uint *out) {
    return clEnqueueReadBuffer(g_q, buf, CL_TRUE,
                               sizeof(cl_uint) * idx, sizeof(cl_uint),
                               out, 0, NULL, NULL) == CL_SUCCESS;
}

// Compact in[i*stride .. i*stride+counts[i]] into out[0..total).
static bool cl_compact_dev(cl_mem in, cl_mem counts, size_t stride,
                            cl_mem out, size_t n_in, size_t *out_len) {
    if (n_in == 0) { *out_len = 0; return true; }
    if (!cl_buf_ensure(&g_dev_prefix, sizeof(cl_uint) * n_in,
                       CL_MEM_READ_WRITE, false)) return false;
    cl_mem prefix = g_dev_prefix.dev;
    if (!cl_scan_uchar_inc_dev(counts, prefix, n_in)) return false;

    cl_uint total = 0;
    if (!cl_read_last_u32(prefix, n_in - 1, &total)) return false;
    *out_len = total;
    if (total == 0) return true;

    cl_uint arg_stride = (cl_uint)stride;
    cl_uint arg_n = (cl_uint)n_in;
    if (clSetKernelArg(g_kern_compact, 0, sizeof(cl_mem),  &in)         != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact, 1, sizeof(cl_mem),  &counts)     != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact, 2, sizeof(cl_mem),  &prefix)     != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact, 3, sizeof(cl_mem),  &out)        != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact, 4, sizeof(cl_uint), &arg_stride) != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact, 5, sizeof(cl_uint), &arg_n)      != CL_SUCCESS) return false;

    size_t local = WG_SIZE_SMALL;
    size_t global = round_up_sz(n_in, local);
    if (clEnqueueNDRangeKernel(g_q, g_kern_compact, 1, NULL,
                               &global, &local, 0, NULL, NULL) != CL_SUCCESS) return false;
    return true;
}

// Dual-stream compact (spawn n2 + n4 share counts/prefix).
static bool cl_compact2_dev(cl_mem in2, cl_mem in4, cl_mem counts, size_t stride,
                             cl_mem out2, cl_mem out4, size_t n_in, size_t *out_len) {
    if (n_in == 0) { *out_len = 0; return true; }
    if (!cl_buf_ensure(&g_dev_prefix, sizeof(cl_uint) * n_in,
                       CL_MEM_READ_WRITE, false)) return false;
    cl_mem prefix = g_dev_prefix.dev;
    if (!cl_scan_uchar_inc_dev(counts, prefix, n_in)) return false;

    cl_uint total = 0;
    if (!cl_read_last_u32(prefix, n_in - 1, &total)) return false;
    *out_len = total;
    if (total == 0) return true;

    cl_uint arg_stride = (cl_uint)stride;
    cl_uint arg_n = (cl_uint)n_in;
    if (clSetKernelArg(g_kern_compact2, 0, sizeof(cl_mem),  &in2)        != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact2, 1, sizeof(cl_mem),  &in4)        != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact2, 2, sizeof(cl_mem),  &counts)     != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact2, 3, sizeof(cl_mem),  &prefix)     != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact2, 4, sizeof(cl_mem),  &out2)       != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact2, 5, sizeof(cl_mem),  &out4)       != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact2, 6, sizeof(cl_uint), &arg_stride) != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_compact2, 7, sizeof(cl_uint), &arg_n)      != CL_SUCCESS) return false;

    size_t local = WG_SIZE_SMALL;
    size_t global = round_up_sz(n_in, local);
    if (clEnqueueNDRangeKernel(g_q, g_kern_compact2, 1, NULL,
                               &global, &local, 0, NULL, NULL) != CL_SUCCESS) return false;
    return true;
}

/* 8-bit-digit, 8-pass LSB radix sort with ping-pong. After 8 (even)
 * passes, the sorted result is back in keys_a. keys_b is scratch. */
static bool cl_radix_sort_dev(cl_mem keys_a, cl_mem keys_b, size_t n) {
    if (n <= 1) return true;
    size_t n_blocks = ceil_div_sz(n, WG_SIZE);
    size_t hist_n = (size_t)RADIX_BINS * n_blocks;

    if (!cl_buf_ensure(&g_dev_hist, sizeof(cl_uint) * hist_n,
                       CL_MEM_READ_WRITE, false)) return false;
    cl_mem hist = g_dev_hist.dev;

    cl_mem src = keys_a, dst = keys_b;
    size_t local = WG_SIZE;
    size_t global = round_up_sz(n, WG_SIZE);
    cl_uint arg_n = (cl_uint)n;
    cl_uint arg_nblocks = (cl_uint)n_blocks;

    for (cl_uint pass = 0; pass < 8u; pass++) {
        cl_uint shift = pass * 8u;

        if (clSetKernelArg(g_kern_radix_hist, 0, sizeof(cl_mem),  &src)         != CL_SUCCESS) return false;
        if (clSetKernelArg(g_kern_radix_hist, 1, sizeof(cl_mem),  &hist)        != CL_SUCCESS) return false;
        if (clSetKernelArg(g_kern_radix_hist, 2, sizeof(cl_uint), &arg_n)       != CL_SUCCESS) return false;
        if (clSetKernelArg(g_kern_radix_hist, 3, sizeof(cl_uint), &shift)       != CL_SUCCESS) return false;
        if (clSetKernelArg(g_kern_radix_hist, 4, sizeof(cl_uint), &arg_nblocks) != CL_SUCCESS) return false;
        if (clEnqueueNDRangeKernel(g_q, g_kern_radix_hist, 1, NULL,
                                   &global, &local, 0, NULL, NULL) != CL_SUCCESS) return false;

        if (!cl_scan_uint_inc_dev(hist, hist_n, 0)) return false;

        if (clSetKernelArg(g_kern_radix_scatter, 0, sizeof(cl_mem),  &src)         != CL_SUCCESS) return false;
        if (clSetKernelArg(g_kern_radix_scatter, 1, sizeof(cl_mem),  &dst)         != CL_SUCCESS) return false;
        if (clSetKernelArg(g_kern_radix_scatter, 2, sizeof(cl_mem),  &hist)        != CL_SUCCESS) return false;
        if (clSetKernelArg(g_kern_radix_scatter, 3, sizeof(cl_uint), &arg_n)       != CL_SUCCESS) return false;
        if (clSetKernelArg(g_kern_radix_scatter, 4, sizeof(cl_uint), &shift)       != CL_SUCCESS) return false;
        if (clSetKernelArg(g_kern_radix_scatter, 5, sizeof(cl_uint), &arg_nblocks) != CL_SUCCESS) return false;
        if (clEnqueueNDRangeKernel(g_q, g_kern_radix_scatter, 1, NULL,
                                   &global, &local, 0, NULL, NULL) != CL_SUCCESS) return false;

        cl_mem tmp = src; src = dst; dst = tmp;
    }
    return true;
}

/* keys_in is sorted; write unique values to keys_out. *out_len receives
 * the number of unique elements. */
static bool cl_uniq_dev(cl_mem keys_in, cl_mem keys_out, size_t n, size_t *out_len) {
    if (n == 0) { *out_len = 0; return true; }
    if (!cl_buf_ensure(&g_dev_uflags, sizeof(cl_uint) * n,
                       CL_MEM_READ_WRITE, false)) return false;
    cl_mem flags = g_dev_uflags.dev;

    cl_uint arg_n = (cl_uint)n;
    if (clSetKernelArg(g_kern_uniq_flag, 0, sizeof(cl_mem),  &keys_in) != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_uniq_flag, 1, sizeof(cl_mem),  &flags)   != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_uniq_flag, 2, sizeof(cl_uint), &arg_n)   != CL_SUCCESS) return false;
    size_t local = WG_SIZE_SMALL;
    size_t global = round_up_sz(n, local);
    if (clEnqueueNDRangeKernel(g_q, g_kern_uniq_flag, 1, NULL,
                               &global, &local, 0, NULL, NULL) != CL_SUCCESS) return false;

    if (!cl_scan_uint_inc_dev(flags, n, 0)) return false;

    cl_uint total = 0;
    if (!cl_read_last_u32(flags, n - 1, &total)) return false;
    *out_len = total;
    if (total == 0) return true;

    if (clSetKernelArg(g_kern_uniq_scatter, 0, sizeof(cl_mem),  &keys_in)  != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_uniq_scatter, 1, sizeof(cl_mem),  &flags)    != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_uniq_scatter, 2, sizeof(cl_mem),  &keys_out) != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_uniq_scatter, 3, sizeof(cl_uint), &arg_n)    != CL_SUCCESS) return false;
    if (clEnqueueNDRangeKernel(g_q, g_kern_uniq_scatter, 1, NULL,
                               &global, &local, 0, NULL, NULL) != CL_SUCCESS) return false;
    return true;
}

/* Clamp counts > max_val to max_val in place, raising the overflow flag
 * if any are clamped. */
static bool cl_clamp_counts_dev(cl_mem counts, uint8_t max_val, size_t n, bool *overflow_out) {
    if (!cl_buf_ensure(&g_dev_overflow, sizeof(cl_uint), CL_MEM_READ_WRITE, false))
        return false;
    cl_uint zero = 0;
    if (clEnqueueFillBuffer(g_q, g_dev_overflow.dev, &zero, sizeof(cl_uint),
                             0, sizeof(cl_uint), 0, NULL, NULL) != CL_SUCCESS) return false;

    cl_uint arg_n = (cl_uint)n;
    cl_uchar arg_max = (cl_uchar)max_val;
    cl_mem ovf = g_dev_overflow.dev;
    if (clSetKernelArg(g_kern_clamp_counts, 0, sizeof(cl_mem),   &counts) != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_clamp_counts, 1, sizeof(cl_mem),   &ovf)    != CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_clamp_counts, 2, sizeof(cl_uchar), &arg_max)!= CL_SUCCESS) return false;
    if (clSetKernelArg(g_kern_clamp_counts, 3, sizeof(cl_uint),  &arg_n)  != CL_SUCCESS) return false;
    size_t local = WG_SIZE_SMALL;
    size_t global = round_up_sz(n, local);
    if (clEnqueueNDRangeKernel(g_q, g_kern_clamp_counts, 1, NULL,
                               &global, &local, 0, NULL, NULL) != CL_SUCCESS) return false;

    cl_uint ovf_val = 0;
    if (clEnqueueReadBuffer(g_q, g_dev_overflow.dev, CL_TRUE, 0,
                            sizeof(cl_uint), &ovf_val, 0, NULL, NULL) != CL_SUCCESS) return false;
    *overflow_out = ovf_val != 0;
    return true;
}

/* Ensure each free DEV_SLOT is at least max_ulongs * 8 bytes. Busy slots
 * are skipped because growing them via clCreateBuffer would discard the
 * live data they hold (e.g. the spawn input between cl_move_dev and the
 * spawn kernel). */
static bool dev_slots_ensure(size_t max_ulongs) {
    size_t bytes = sizeof(uint64_t) * max_ulongs;
    if (bytes == 0) bytes = sizeof(uint64_t);
    for (int i = 0; i < DEV_SLOTS; i++) {
        if (g_slot_busy & (1u << i)) continue;
        if (!cl_buf_ensure(&g_dev_slots[i], bytes, CL_MEM_READ_WRITE, false))
            return false;
    }
    return true;
}

// Public device-resident API

bool cl_move_dev(const uint64_t *boards, size_t n, bool prune,
                 long stsl, long ltc, long smallest_large, cl_darr **out)
{
    if (out) *out = NULL;
    if (!cl_init() || !g_luts_uploaded) return false;
    if (n == 0) {
        cl_darr *h = malloc(sizeof(*h));
        if (!h) return false;
        h->slot = -1; h->n = 0;
        if (out) *out = h;
        return true;
    }

    if (!cl_buf_ensure(&g_dev_input,  sizeof(uint64_t) * n, CL_MEM_READ_ONLY,  false)) return false;
    if (!cl_buf_ensure(&g_dev_counts, sizeof(uint8_t)  * n, CL_MEM_READ_WRITE, false)) return false;
    if (!dev_slots_ensure(n * 4)) return false;

    if (clEnqueueWriteBuffer(g_q, g_dev_input.dev, CL_FALSE, 0,
                             sizeof(uint64_t) * n, boards, 0, NULL, NULL) != CL_SUCCESS) return false;

    int s_raw = slot_acquire();
    if (s_raw < 0) return false;
    cl_mem buf_raw = g_dev_slots[s_raw].dev;
    cl_mem buf_in  = g_dev_input.dev;
    cl_mem buf_counts = g_dev_counts.dev;

    start_node(GEN_MOVE);
    cl_uint arg_n = (cl_uint)n;
    cl_uint arg_prune = prune ? 1u : 0u;
    cl_long arg_stsl  = (cl_long)stsl;
    cl_long arg_ltc   = (cl_long)ltc;
    cl_long arg_sl    = (cl_long)smallest_large;
    if (clSetKernelArg(g_kern, 0, sizeof(cl_mem),  &buf_in)            != CL_SUCCESS) goto fail;
    if (clSetKernelArg(g_kern, 1, sizeof(cl_mem),  &buf_raw)           != CL_SUCCESS) goto fail;
    if (clSetKernelArg(g_kern, 2, sizeof(cl_mem),  &buf_counts)        != CL_SUCCESS) goto fail;
    if (clSetKernelArg(g_kern, 3, sizeof(cl_mem),  &g_buf_move_lut)    != CL_SUCCESS) goto fail;
    if (clSetKernelArg(g_kern, 4, sizeof(cl_mem),  &g_buf_locked_lut)  != CL_SUCCESS) goto fail;
    if (clSetKernelArg(g_kern, 5, sizeof(cl_uint), &arg_n)             != CL_SUCCESS) goto fail;
    if (clSetKernelArg(g_kern, 6, sizeof(cl_uint), &arg_prune)         != CL_SUCCESS) goto fail;
    if (clSetKernelArg(g_kern, 7, sizeof(cl_long), &arg_stsl)          != CL_SUCCESS) goto fail;
    if (clSetKernelArg(g_kern, 8, sizeof(cl_long), &arg_ltc)           != CL_SUCCESS) goto fail;
    if (clSetKernelArg(g_kern, 9, sizeof(cl_long), &arg_sl)            != CL_SUCCESS) goto fail;
    {
        size_t local = WG_SIZE_SMALL;
        size_t global = round_up_sz(n, local);
        if (clEnqueueNDRangeKernel(g_q, g_kern, 1, NULL, &global, &local, 0, NULL, NULL) != CL_SUCCESS) goto fail;
    }
    if (clFinish(g_q) != CL_SUCCESS) goto fail;
    end_node(GEN_MOVE);

    start_node(COMBINE_MOVE);
    int s_compact = slot_acquire();
    if (s_compact < 0) goto fail;
    cl_mem buf_compact = g_dev_slots[s_compact].dev;
    size_t n_compacted = 0;
    if (!cl_compact_dev(buf_raw, buf_counts, 4, buf_compact, n, &n_compacted)) {
        slot_release(s_compact);
        goto fail;
    }
    slot_release(s_raw); // raw move output no longer needed
    if (clFinish(g_q) != CL_SUCCESS) { slot_release(s_compact); goto fail; }
    end_node(COMBINE_MOVE);

    start_node(DEDUPE_MOVE);
    int s_scratch = slot_acquire();
    if (s_scratch < 0) { slot_release(s_compact); goto fail; }
    cl_mem buf_scratch = g_dev_slots[s_scratch].dev;
    if (n_compacted > 1) {
        if (!cl_radix_sort_dev(buf_compact, buf_scratch, n_compacted)) {
            slot_release(s_compact); slot_release(s_scratch); goto fail;
        }
    }
    size_t n_unique = 0;
    int s_uniq = slot_acquire();
    if (s_uniq < 0) { slot_release(s_compact); slot_release(s_scratch); goto fail; }
    cl_mem buf_uniq = g_dev_slots[s_uniq].dev;
    if (!cl_uniq_dev(buf_compact, buf_uniq, n_compacted, &n_unique)) {
        slot_release(s_compact); slot_release(s_scratch); slot_release(s_uniq); goto fail;
    }
    slot_release(s_compact);
    slot_release(s_scratch);
    if (clFinish(g_q) != CL_SUCCESS) { slot_release(s_uniq); goto fail; }
    end_node(DEDUPE_MOVE);

    cl_darr *h = malloc(sizeof(*h));
    if (!h) { slot_release(s_uniq); goto fail; }
    h->slot = s_uniq;
    h->n    = n_unique;
    if (out) *out = h;
    return true;
fail:
    slot_release(s_raw);
    log_out("cl_move_dev failed.", LOG_ERROR);
    return false;
}

bool cl_spawn_dev(cl_darr *in, cl_darr **out_n2, cl_darr **out_n4)
{
    if (out_n2) *out_n2 = NULL;
    if (out_n4) *out_n4 = NULL;
    if (!in) return false;
    if (!cl_init()) return false;
    if (in->n == 0) {
        cl_darr *e2 = malloc(sizeof(*e2)); if (!e2) return false;
        cl_darr *e4 = malloc(sizeof(*e4)); if (!e4) { free(e2); return false; }
        e2->slot = -1; e2->n = 0;
        e4->slot = -1; e4->n = 0;
        if (out_n2) *out_n2 = e2; else free(e2);
        if (out_n4) *out_n4 = e4; else free(e4);
        cl_darr_release(in);
        return true;
    }

    size_t n = in->n;
    cl_mem buf_in = g_dev_slots[in->slot].dev;

    if (!cl_buf_ensure(&g_dev_counts, sizeof(uint8_t) * n, CL_MEM_READ_WRITE, false)) return false;
    if (!dev_slots_ensure(n * SPAWN_SLOTS)) return false;

    start_node(GEN_SPAWN);
    int s_o2 = slot_acquire();
    int s_o4 = slot_acquire();
    if (s_o2 < 0 || s_o4 < 0) {
        slot_release(s_o2); slot_release(s_o4);
        return false;
    }
    cl_mem buf_o2 = g_dev_slots[s_o2].dev;
    cl_mem buf_o4 = g_dev_slots[s_o4].dev;
    cl_mem buf_counts = g_dev_counts.dev;

    cl_uint arg_n = (cl_uint)n;
    if (clSetKernelArg(g_kern_spawn, 0, sizeof(cl_mem),  &buf_in)     != CL_SUCCESS) goto fail2;
    if (clSetKernelArg(g_kern_spawn, 1, sizeof(cl_mem),  &buf_o2)     != CL_SUCCESS) goto fail2;
    if (clSetKernelArg(g_kern_spawn, 2, sizeof(cl_mem),  &buf_o4)     != CL_SUCCESS) goto fail2;
    if (clSetKernelArg(g_kern_spawn, 3, sizeof(cl_mem),  &buf_counts) != CL_SUCCESS) goto fail2;
    if (clSetKernelArg(g_kern_spawn, 4, sizeof(cl_uint), &arg_n)      != CL_SUCCESS) goto fail2;
    {
        size_t local = WG_SIZE_SMALL;
        size_t global = round_up_sz(n, local);
        if (clEnqueueNDRangeKernel(g_q, g_kern_spawn, 1, NULL, &global, &local, 0, NULL, NULL) != CL_SUCCESS)
            goto fail2;
    }

    // Detect / clamp overflow (boards with > SPAWN_SLOTS empty tiles).
    bool overflow = false;
    if (!cl_clamp_counts_dev(buf_counts, (uint8_t)SPAWN_SLOTS, n, &overflow)) goto fail2;
    if (overflow) {
        log_out("cl_spawn_dev: input has boards with > SPAWN_SLOTS empties; falling back.", LOG_DBG);
        goto fail2_quiet;
    }
    if (clFinish(g_q) != CL_SUCCESS) goto fail2;
    end_node(GEN_SPAWN);

    /* Spawn input no longer needed. Release the input slot for reuse.
     * The freed slot may be smaller than the spawn output (it was sized
     * by cl_move_dev to n_move_input*4 ulongs, but spawn output streams
     * are n*4 ulongs where n is the move-deduped count). Re-ensure all
     * free slots are large enough before further allocations consume
     * them as compact/sort scratch. */
    cl_darr_release(in);
    in = NULL;
    if (!dev_slots_ensure(n * SPAWN_SLOTS)) return false;

    start_node(COMBINE_SPAWN);

    // Compact n2 stream.
    int s_c2 = slot_acquire();
    if (s_c2 < 0) goto fail2;
    cl_mem buf_c2 = g_dev_slots[s_c2].dev;
    size_t n_c2 = 0;
    if (!cl_compact_dev(buf_o2, buf_counts, SPAWN_SLOTS, buf_c2, n, &n_c2)) {
        slot_release(s_c2); goto fail2;
    }
    slot_release(s_o2); // raw o2 no longer needed

    // Sort + uniq n2: ping-pong with another scratch slot, write to s_n2.
    int s_x2 = slot_acquire();
    if (s_x2 < 0) { slot_release(s_c2); goto fail2; }
    if (n_c2 > 1) {
        if (!cl_radix_sort_dev(buf_c2, g_dev_slots[s_x2].dev, n_c2)) {
            slot_release(s_c2); slot_release(s_x2); goto fail2;
        }
    }
    slot_release(s_x2);
    int s_n2 = slot_acquire();
    if (s_n2 < 0) { slot_release(s_c2); goto fail2; }
    size_t n_n2 = 0;
    if (!cl_uniq_dev(buf_c2, g_dev_slots[s_n2].dev, n_c2, &n_n2)) {
        slot_release(s_c2); slot_release(s_n2); goto fail2;
    }
    slot_release(s_c2);

    // Compact n4 stream.
    int s_c4 = slot_acquire();
    if (s_c4 < 0) { slot_release(s_n2); goto fail2; }
    cl_mem buf_c4 = g_dev_slots[s_c4].dev;
    size_t n_c4 = 0;
    if (!cl_compact_dev(buf_o4, buf_counts, SPAWN_SLOTS, buf_c4, n, &n_c4)) {
        slot_release(s_c4); slot_release(s_n2); goto fail2;
    }
    slot_release(s_o4);

    // Sort + uniq n4 -> s_n4.
    int s_x4 = slot_acquire();
    if (s_x4 < 0) { slot_release(s_c4); slot_release(s_n2); goto fail2; }
    if (n_c4 > 1) {
        if (!cl_radix_sort_dev(buf_c4, g_dev_slots[s_x4].dev, n_c4)) {
            slot_release(s_c4); slot_release(s_x4); slot_release(s_n2); goto fail2;
        }
    }
    slot_release(s_x4);
    int s_n4 = slot_acquire();
    if (s_n4 < 0) { slot_release(s_c4); slot_release(s_n2); goto fail2; }
    size_t n_n4 = 0;
    if (!cl_uniq_dev(buf_c4, g_dev_slots[s_n4].dev, n_c4, &n_n4)) {
        slot_release(s_c4); slot_release(s_n4); slot_release(s_n2); goto fail2;
    }
    slot_release(s_c4);
    if (clFinish(g_q) != CL_SUCCESS) { slot_release(s_n2); slot_release(s_n4); goto fail2; }
    end_node(COMBINE_SPAWN);
    start_node(DEDUPE_SPAWN);
    end_node(DEDUPE_SPAWN);

    cl_darr *h2 = malloc(sizeof(*h2));
    cl_darr *h4 = malloc(sizeof(*h4));
    if (!h2 || !h4) {
        free(h2); free(h4);
        slot_release(s_n2); slot_release(s_n4);
        return false;
    }
    h2->slot = s_n2; h2->n = n_n2;
    h4->slot = s_n4; h4->n = n_n4;
    if (out_n2) *out_n2 = h2; else { slot_release(s_n2); free(h2); }
    if (out_n4) *out_n4 = h4; else { slot_release(s_n4); free(h4); }
    return true;

fail2_quiet:
    slot_release(s_o2); slot_release(s_o4);
    if (in) cl_darr_release(in);
    return false;
fail2:
    slot_release(s_o2); slot_release(s_o4);
    log_out("cl_spawn_dev failed.", LOG_ERROR);
    if (in) cl_darr_release(in);
    return false;
}

bool cl_darr_to_host(cl_darr *in, dynamic_arr_info *out)
{
    if (!out) return false;
    *out = (dynamic_arr_info){ .valid = false, .bp = NULL, .sp = NULL, .size = 0 };
    if (!in) return false;
    if (in->n == 0) {
        *out = init_darr(false, 0);
        return true;
    }
    dynamic_arr_info d = init_darr(false, in->n);
    if (!d.valid) return false;
    d.sp = d.bp + in->n;
    cl_mem src = g_dev_slots[in->slot].dev;
    if (clEnqueueReadBuffer(g_q, src, CL_TRUE, 0,
                            sizeof(uint64_t) * in->n, d.bp, 0, NULL, NULL) != CL_SUCCESS) {
        destroy_darr(&d);
        return false;
    }
    *out = d;
    return true;
}

bool cl_darr_download(cl_darr *in, dynamic_arr_info *out)
{
    bool ok = cl_darr_to_host(in, out);
    cl_darr_release(in);
    return ok;
}

void cl_darr_release(cl_darr *in)
{
    if (!in) return;
    if (in->slot >= 0) slot_release(in->slot);
    free(in);
}

size_t cl_vram_bytes(void)
{
    size_t total = 0;
    cl_buf *bufs[] = {
        &g_move_in, &g_move_out, &g_move_counts,
        &g_spawn_in, &g_spawn_o2, &g_spawn_o4, &g_spawn_counts,
        &g_dev_input, &g_dev_counts, &g_dev_prefix, &g_dev_uflags,
        &g_dev_hist, &g_dev_overflow,
    };
    for (size_t i = 0; i < sizeof(bufs)/sizeof(bufs[0]); i++) total += bufs[i]->bytes;
    for (int i = 0; i < DEV_SLOTS; i++) total += g_dev_slots[i].bytes;
    for (int i = 0; i < MAX_SCAN_LEVELS; i++) total += g_dev_scan_l[i].bytes;
    /* LUTs uploaded by cl_upload_luts() are clCreateBuffer'd directly,
     * not via the cl_buf pool. Add their known sizes if present. */
    if (g_buf_move_lut)   total += sizeof(uint16_t) * 2 * 65536;
    if (g_buf_locked_lut) total += 2 * 65536;
    return total;
}

#endif /* USE_OPENCL */
