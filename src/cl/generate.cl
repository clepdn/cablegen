/*
 * Device-side port of:
 *   - movedir_unstable (board.c)
 *   - canonicalize_b   (scalar fallback path in board.c)
 *   - prune_board      (generate.c)
 * plus the per-input-board loop body from generation_thread_move /
 * generation_thread_movep.
 *
 * One work-item == one input board. Each work-item writes up to 4 result
 * boards into out_boards[4*gid + d] and the count into out_counts[gid].
 * The host then stream-compacts (CPU side, cheap relative to the move
 * itself) and sorts.
 *
 * Conventions match board.h:
 *   GET_TILE(b, x) = nibble x, where x=0 is the *high* nibble.
 *   _move_lut[dir][16-bit row] = 16-bit row after a left/right shift.
 *   _locked_lut[dir][row] = false if the row would have shifted a 0xf
 *     tile (which is illegal in non-free-formation play).
 *
 * dir encoding (must match board.h): left=0, right=1, up=2, down=3.
 */

#define DIR_LEFT  0u
#define DIR_RIGHT 1u
#define DIR_UP    2u
#define DIR_DOWN  3u

#define BIT_MASK  0xf000000000000000UL
#define OFFSET(x) ((x) * 4u)
#define GET_TILE(b, x) ((uchar)((((b) << OFFSET(x)) & BIT_MASK) >> OFFSET(15)))
#define SET_TILE(b, x, v) \
    ((b) = (~(~(b) | (BIT_MASK >> OFFSET(x))) | \
            ((BIT_MASK & ((ulong)(v) << OFFSET(15))) >> OFFSET(x))))

// Rotations (bitwise, taken verbatim from board.c)

static inline ulong rot_cw(ulong b) {
    b = ((b & 0xff00ff0000000000UL) >> 8 ) |
        ((b & 0x00ff00ff00000000UL) >> 32) |
        ((b & 0x00000000ff00ff00UL) << 32) |
        ((b & 0x0000000000ff00ffUL) << 8 );
    b = ((b & 0xf0f00000f0f00000UL) >> 4 ) |
        ((b & 0x0f0f00000f0f0000UL) >> 16) |
        ((b & 0x0000f0f00000f0f0UL) << 16) |
        ((b & 0x00000f0f00000f0fUL) << 4 );
    return b;
}

static inline ulong rot_ccw(ulong b) {
    b = ((b & 0xff00ff0000000000UL) >> 32) |
        ((b & 0x00ff00ff00000000UL) << 8 ) |
        ((b & 0x00000000ff00ff00UL) >> 8 ) |
        ((b & 0x0000000000ff00ffUL) << 32);
    b = ((b & 0xf0f00000f0f00000UL) >> 16) |
        ((b & 0x0f0f00000f0f0000UL) << 4 ) |
        ((b & 0x0000f0f00000f0f0UL) >> 4 ) |
        ((b & 0x00000f0f00000f0fUL) << 16);
    return b;
}

static inline ulong rot_180(ulong b) {
    b = ((b & 0xffffffff00000000UL) >> 32) | ((b & 0x00000000ffffffffUL) << 32);
    b = ((b & 0xffff0000ffff0000UL) >> 16) | ((b & 0x0000ffff0000ffffUL) << 16);
    b = ((b & 0xff00ff00ff00ff00UL) >> 8 ) | ((b & 0x00ff00ff00ff00ffUL) << 8 );
    b = ((b & 0xf0f0f0f0f0f0f0f0UL) >> 4 ) | ((b & 0x0f0f0f0f0f0f0f0fUL) << 4 );
    return b;
}

static inline ulong flip_b(ulong b) {
    // mirror left<->right (matches the fast path of flip() in board.c)
    b = ((b & 0xffffffff00000000UL) >> 32) | ((b & 0x00000000ffffffffUL) << 32);
    b = ((b & 0xffff0000ffff0000UL) >> 16) | ((b & 0x0000ffff0000ffffUL) << 16);
    /* The CPU flip() stops here -- it's actually flipping top<->bottom and
     * then up/down are normalized by the rotations. Keep the exact same
     * behaviour so canonical boards are bit-identical with the CPU path. */
    return b;
}

// Row-LUT move (mirrors movedir_hori in board.c)

/* direction is 0 (left) or 1 (right) here -- caller has already
 * rotated the board for up/down. Returns the new board; sets *changed
 * if any row's bits differ; sets *legal=false if any row was locked.
 * On illegal, board is returned to its pre-move state. */
static inline ulong movedir_hori(ulong board,
                                 uint dir01,
                                 __global const ushort *move_lut,   /* [2][65536] */
                                 __global const uchar  *locked_lut, /* [2][65536] */
                                 bool *changed_out,
                                 bool *legal_out) {
    const ulong pre = board;
    bool changed = false;
    const uint base = dir01 * 65536u;
    for (int i = 0; i < 4; i++) {
        ushort row = (ushort)((board >> (16 * i)) & 0xFFFFUL);
        ushort newrow = move_lut[base + row];
        if (newrow != row) {
            changed = true;
            board &= ~((ulong)0xFFFFUL << (16 * i));
            board |= ((ulong)newrow) << (16 * i);
        }
        if (!locked_lut[base + row]) {
            *changed_out = false;
            *legal_out = false;
            return pre;
        }
    }
    *changed_out = changed;
    *legal_out = true;
    return board;
}

/* Equivalent of movedir_unstable: returns the moved board, and writes
 * whether the move was legal+changed to *valid. The board may be left
 * rotated (matches the CPU "unstable" contract -- canonicalize_b
 * collapses the orbit so this doesn't matter). */
static inline ulong movedir_unstable_dev(ulong board,
                                         uint dir,
                                         __global const ushort *move_lut,
                                         __global const uchar  *locked_lut,
                                         bool *valid) {
    bool changed = false, legal = false;
    ulong out;
    switch (dir) {
        case DIR_LEFT:
            out = movedir_hori(board, 0u, move_lut, locked_lut, &changed, &legal);
            break;
        case DIR_RIGHT:
            out = movedir_hori(board, 1u, move_lut, locked_lut, &changed, &legal);
            break;
        case DIR_UP:
            out = movedir_hori(rot_cw(board), 1u, move_lut, locked_lut, &changed, &legal);
            break;
        case DIR_DOWN:
        default:
            out = movedir_hori(rot_cw(board), 0u, move_lut, locked_lut, &changed, &legal);
            break;
    }
    *valid = legal && changed;
    return out;
}

// canonicalize (scalar fallback; AVX-512 path is host-only)

static inline ulong canonicalize(ulong b) {
    ulong mx = b;

    ulong r = rot_cw(b);
    ulong c0 = r;
    if (r > mx) mx = r;

    r = rot_cw(r);
    ulong c1 = r;
    if (r > mx) mx = r;

    r = rot_cw(r);
    ulong c2 = r;
    if (r > mx) mx = r;

    ulong f = flip_b(b);
    if (f > mx) mx = f;
    f = flip_b(c0);
    if (f > mx) mx = f;
    f = flip_b(c1);
    if (f > mx) mx = f;
    f = flip_b(c2);
    if (f > mx) mx = f;

    return mx;
}

// prune_board (port of generate.c)

static inline bool prune_board(ulong board, long stsl, long ltc, long smallest_large) {
    short large_tiles = 0;
    int smallest = 0xff;
    int sts = 0;
    ushort tiles = 0;
    ushort tiles2 = 0;
    char c64 = 0;
    for (short i = 0; i < 16; i++) {
        uchar tmp = GET_TILE(board, (uint)i);
        if (tmp >= (uchar)smallest_large && tmp < 0xe) {
            if (tmp < smallest) smallest = tmp;
            large_tiles++;
            if (tmp == (uchar)smallest_large && c64 < 3) {
                c64++;
            } else if (tmp == (uchar)smallest_large && c64 == 2) {
                return true;
            } else if (!(tiles & (1u << tmp))) {
                tiles |= (1u << tmp);
            } else if (!(tiles2 & (1u << tmp))) {
                tiles2 |= (1u << tmp);
            } else {
                return true;
            }
        } else if (tmp < 0xe) {
            sts += 1 << tmp;
        }
    }
    if (sts > stsl + 64) return true;
    if (large_tiles > ltc) return true;
    return false;
}

// Spawn kernel

/* One work-item per input board. For every empty (==0) tile in the
 * input, emit one canonicalized board with that tile set to 1 (spawn-2)
 * into out_2 and one with that tile set to 2 (spawn-4) into out_4.
 * out_counts[gid] = number of empty tiles found.
 *
 * SLOTS is the per-input slot count; the host validates and falls back
 * to CPU for any input that exceeded SLOTS. Host compacts. */
#ifndef SPAWN_SLOTS
/* Most cablegen workloads have <=4 empty tiles per board (p99.89);
 * larger inputs are handled by the host as a leftover list. */
#define SPAWN_SLOTS 4
#endif
__kernel void spawn_kernel(__global const ulong *in_boards,
                           __global       ulong *out_2,       /* [SPAWN_SLOTS*N] */
                           __global       ulong *out_4,       /* [SPAWN_SLOTS*N] */
                           __global       uchar *out_counts,  /* [N]    */
                                   const uint    n)
{
    const uint gid = get_global_id(0);
    if (gid >= n) return;

    const ulong board = in_boards[gid];
    const ulong base  = (ulong)gid * SPAWN_SLOTS;

    uchar count = 0;
    for (uint t = 0; t < 16; t++) {
        if (GET_TILE(board, t) == 0) {
            if (count < SPAWN_SLOTS) {
                ulong b2 = board; SET_TILE(b2, t, 1); b2 = canonicalize(b2);
                ulong b4 = board; SET_TILE(b4, t, 2); b4 = canonicalize(b4);
                out_2[base + count] = b2;
                out_4[base + count] = b4;
            }
            count++;
        }
    }
    out_counts[gid] = count;
}

// Move kernel

__kernel void move_kernel(__global const ulong  *in_boards,
                          __global       ulong  *out_boards,   /* [4*N]    */
                          __global       uchar  *out_counts,   /* [N]      */
                          __global const ushort *move_lut,     /* [2*65536]*/
                          __global const uchar  *locked_lut,   /* [2*65536]*/
                                  const uint    n,
                                  const uint    prune,         /* 0 or 1   */
                                  const long    stsl,
                                  const long    ltc,
                                  const long    smallest_large)
{
    const uint gid = get_global_id(0);
    if (gid >= n) return;

    const ulong board = in_boards[gid];
    const ulong base = (ulong)gid * 4u;

    if (prune && prune_board(board, stsl, ltc, smallest_large)) {
        out_counts[gid] = 0;
        return;
    }

    uchar count = 0;
    for (uint d = 0; d < 4; d++) {
        bool valid = false;
        ulong moved = movedir_unstable_dev(board, d, move_lut, locked_lut, &valid);
        if (!valid) continue;
        if (prune && prune_board(moved, stsl, ltc, smallest_large)) continue;
        moved = canonicalize(moved);
        out_boards[base + count] = moved;
        count++;
    }
    out_counts[gid] = count;
}

// Device-resident scan / compact / sort / uniq

#ifndef WG_SIZE
#define WG_SIZE 256
#endif
#define RADIX_BINS 256u
#define RADIX_MASK 0xFFu

/* Inclusive Hillis-Steele scan over a single workgroup-sized block.
 * Writes the per-element inclusive scan to out[] and the block total
 * (== inclusive[last element of block]) to block_sums[wgid]. */
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void scan_inc_uint_kernel(__global const uint *in,
                          __global       uint *out,
                          __global       uint *block_sums,
                                          uint n)
{
    __local uint sdata[2 * WG_SIZE];
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wgid = get_group_id(0);

    uint val = (gid < n) ? in[gid] : 0u;

    uint pin = 0;
    sdata[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < WG_SIZE; offset <<= 1) {
        uint pout = 1u - pin;
        uint a = sdata[pin * WG_SIZE + lid];
        uint b = (lid >= offset) ? sdata[pin * WG_SIZE + lid - offset] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        sdata[pout * WG_SIZE + lid] = a + b;
        barrier(CLK_LOCAL_MEM_FENCE);
        pin = pout;
    }

    uint inc = sdata[pin * WG_SIZE + lid];
    if (gid < n) out[gid] = inc;
    if (lid == WG_SIZE - 1u) block_sums[wgid] = inc;
}

// Same as scan_inc_uint_kernel but reads uchar input (per-input counts).
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void scan_inc_uchar_kernel(__global const uchar *in,
                           __global       uint  *out,
                           __global       uint  *block_sums,
                                           uint  n)
{
    __local uint sdata[2 * WG_SIZE];
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wgid = get_group_id(0);

    uint val = (gid < n) ? (uint)in[gid] : 0u;

    uint pin = 0;
    sdata[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < WG_SIZE; offset <<= 1) {
        uint pout = 1u - pin;
        uint a = sdata[pin * WG_SIZE + lid];
        uint b = (lid >= offset) ? sdata[pin * WG_SIZE + lid - offset] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        sdata[pout * WG_SIZE + lid] = a + b;
        barrier(CLK_LOCAL_MEM_FENCE);
        pin = pout;
    }

    uint inc = sdata[pin * WG_SIZE + lid];
    if (gid < n) out[gid] = inc;
    if (lid == WG_SIZE - 1u) block_sums[wgid] = inc;
}

/* After recursively scanning block_sums, add the per-block base to every
 * element of out[]. block_sums_inc[wgid] is the inclusive scan of block
 * totals, so the base to add to block wgid is block_sums_inc[wgid-1]
 * (== 0 for wgid==0). */
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void scan_add_kernel(__global       uint *out,
                     __global const uint *block_sums_inc,
                                     uint n)
{
    const uint gid = get_global_id(0);
    const uint wgid = get_group_id(0);
    if (gid >= n) return;
    if (wgid == 0u) return;
    out[gid] += block_sums_inc[wgid - 1u];
}

/* Stream-compaction by per-input counts: copy
 *   in[i*stride .. i*stride + counts[i])
 * into out[prefix_inc[i] - counts[i] .. prefix_inc[i]). */
__kernel void compact_kernel(__global const ulong *in,
                             __global const uchar *counts,
                             __global const uint  *prefix_inc,
                             __global       ulong *out,
                                             uint  stride,
                                             uint  n)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    uchar c = counts[i];
    if (c > stride) c = (uchar)stride;
    uint base = prefix_inc[i] - (uint)c;
    for (uint k = 0; k < c; k++)
        out[base + k] = in[i * stride + k];
}

// Same but for two parallel streams sharing one counts/prefix (spawn n2+n4).
__kernel void compact2_kernel(__global const ulong *in2,
                              __global const ulong *in4,
                              __global const uchar *counts,
                              __global const uint  *prefix_inc,
                              __global       ulong *out2,
                              __global       ulong *out4,
                                              uint  stride,
                                              uint  n)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    uchar c = counts[i];
    if (c > stride) c = (uchar)stride;
    uint base = prefix_inc[i] - (uint)c;
    for (uint k = 0; k < c; k++) {
        out2[base + k] = in2[i * stride + k];
        out4[base + k] = in4[i * stride + k];
    }
}

/* If any per-input count exceeds max_val, set *overflow to 1, and clamp
 * counts in place. Used by the device-resident spawn path so the rest of
 * the pipeline can scan a bounded count. */
__kernel void clamp_counts_kernel(__global       uchar *counts,
                                  __global       uint  *overflow,
                                          const uchar  max_val,
                                          const uint   n)
{
    const uint gid = get_global_id(0);
    if (gid >= n) return;
    uchar c = counts[gid];
    if (c > max_val) {
        atomic_or(overflow, 1u);
        counts[gid] = max_val;
    }
}

/* Per-block, per-bin histogram. Bin-major global layout
 * hist[bin * n_blocks + wgid] so a single 1-D inclusive scan yields the
 * exclusive base for (bin, block) at hist_inc[bin*n_blocks+wgid - 1]. */
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void radix_hist_kernel(__global const ulong *keys,
                       __global       uint  *hist,
                                       uint  n,
                                       uint  shift,
                                       uint  n_blocks)
{
    __local uint lhist[RADIX_BINS];
    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint wgid = get_group_id(0);

    // WG_SIZE == RADIX_BINS so each work-item zeroes one bin.
    lhist[lid] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < n) {
        uint bin = (uint)((keys[gid] >> shift) & (ulong)RADIX_MASK);
        atomic_inc(&lhist[bin]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    hist[lid * n_blocks + wgid] = lhist[lid];
}

/* Stable scatter using the inclusive-scanned histogram. Stability is
 * essential for LSB radix sort -- each pass must preserve the order
 * established by the previous pass.
 *
 * For each element we need rank_within_block_within_bin: the count of
 * preceding work-items in the same workgroup whose bin equals ours.
 * We compute this with a serial walk over the per-block bin array in
 * local memory; cheap because WG_SIZE is small (256) and the bin array
 * fits in __local. atomic_inc would be faster but isn't order-stable. */
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void radix_scatter_kernel(__global const ulong *keys_in,
                          __global       ulong *keys_out,
                          __global const uint  *hist_inc,
                                          uint  n,
                                          uint  shift,
                                          uint  n_blocks)
{
    __local uint lbins[WG_SIZE];

    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint wgid = get_group_id(0);

    bool valid = gid < n;
    ulong key = valid ? keys_in[gid] : 0UL;
    uint bin = valid ? (uint)((key >> shift) & (ulong)RADIX_MASK) : RADIX_BINS;
    lbins[lid] = bin;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (valid) {
        uint loff = 0u;
        for (uint j = 0; j < lid; j++)
            if (lbins[j] == bin) loff++;

        uint idx = bin * n_blocks + wgid;
        uint base = (idx == 0u) ? 0u : hist_inc[idx - 1u];
        keys_out[base + loff] = key;
    }
}

/* For sorted keys, flag[i] = 1 if keys[i] != keys[i-1] (or i == 0).
 * Followed by a scan + uniq_scatter to keep one of each value. */
__kernel void uniq_flag_kernel(__global const ulong *keys,
                               __global       uint  *flags,
                                               uint  n)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    if (i == 0u) { flags[i] = 1u; return; }
    flags[i] = (keys[i] != keys[i - 1u]) ? 1u : 0u;
}

/* Scatter for uniq: write the key whenever the inclusive prefix
 * increases (i.e. flag[i] == 1). */
__kernel void uniq_scatter_kernel(__global const ulong *keys_in,
                                  __global const uint  *prefix_inc,
                                  __global       ulong *keys_out,
                                                  uint  n)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    uint cur = prefix_inc[i];
    uint prev = (i == 0u) ? 0u : prefix_inc[i - 1u];
    if (cur > prev) {
        keys_out[cur - 1u] = keys_in[i];
    }
}
