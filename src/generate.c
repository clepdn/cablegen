#include "generate.h"
#include "array.h"
#include "bench.h"
#include <time.h>
#define LOG_H_ENUM_PREFIX_
#define LOG_H_NAMESPACE_ 
#include "logging.h"
#include "format.h"
#include "board.h"
#include "settings.h"
#include "cl_generate.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <fcntl.h>
#include <sys/resource.h>
#define STR(x) #x
#define EXPAND_STR(x) STR(x)
#define VERSION_STR EXPAND_STR(VERSION)

typedef struct {
	static_arr_info n; 
	dynamic_arr_info nret;
	dynamic_arr_info n2; 
	dynamic_arr_info n4;
	size_t start; 
	size_t end; 
	long stsl; 
	long smallest_large; 
	long ltc;
	char nox;
	long layer;
	pthread_t thread;
} arguments;
enum thread_op {
	move,
	movep,
	spawn,
};

void print_speed(uint64_t size){
	static bool init = false;
	static bool enabled = true;
	static struct timespec old_time;
	struct timespec time;
	if(!enabled)
		return;
	if(clock_gettime(CLOCK_MONOTONIC, &time)){
		log_out("Could not get system time!", LOG_WARN);
		return;
	}
	if(!init){
		init = true;
		if(clock_gettime(CLOCK_MONOTONIC, &old_time)){
			log_out("Could not get system time, speed information will be disabled!", LOG_WARN);
			enabled = false;
			return;
		}
		return; // dont need to display on the first layer bc we don't know when startup was
	}
	struct timespec diff = { 
		.tv_sec  = time.tv_sec  - old_time.tv_sec,
		.tv_nsec = time.tv_nsec - old_time.tv_nsec };
	old_time = time;
	long totalns = (1'000'000'000 * diff.tv_sec) + diff.tv_nsec;
	if(totalns == 0){
		return;
	}
	logf_out("Speed: %ld thousand boards per second", LOG_INFO, (long)((float)((float)size / 1000)/ ((float)totalns / (float)1000000000)));
}

void write_boards(const static_arr_info n, const char* fmt, const int layer){
	char* filename = format_str(fmt, layer);
	logf_out("Writing %lu boards to %s (%lu bytes)", LOG_INFO, n.size, filename, sizeof(uint64_t) * n.size);
	FILE *file = fopen(filename, "wb");
	if(file == NULL){
		logf_out("Couldn't write to %s!", LOG_WARN, filename);
		free(filename);
		return;
	}
	fwrite(n.bp, n.size, sizeof(uint64_t), file);
	free(filename);
	fclose(file);
	print_speed(n.size);
}

bool checkx(uint64_t board, char x){
	for(int i = 0; i < 16; i++){
		if((GET_TILE(board, i)) == x)
			return false;
	}
	logf_out("Board %016lx does not contain %d", LOG_TRACE, board, x);
	return true;
}

bool prune_board(const uint64_t board, const long stsl, const long ltc, const long smallest_large){
	short tmp = 0; // TODO: masking
	short large_tiles = 0;
	int smallest = 0xff;
	int sts = 0; // small tile sum
	uint16_t tiles = 0;
	uint16_t tiles2 = 0;
	char c64 = 0;
	for(short i = 0; i < 16; i++){
		if((tmp = GET_TILE(board, i)) >= smallest_large && tmp < 0xe){
			smallest = tmp > smallest ? smallest : tmp;
			large_tiles++;
			if (tmp == smallest_large && c64 < 3)
				c64++;
			else if (tmp == smallest_large && c64 == 2)
				return true;
			else if(!GETBIT(tiles, tmp))
				SETBIT(tiles, tmp);
			else if(!GETBIT(tiles2, tmp))
				SETBIT(tiles2, tmp);
			else
				return true;
		}
		else if(tmp < 0xe){
			sts += 1 << tmp;
		}
	}
	if(sts > stsl + 64)
		return true;
	if(large_tiles > ltc)
		return true;
	// condition number three seems impossible??
	return false;
}

void *generation_thread_move(void* data){ // n, nret
	arguments *args = data;
	uint64_t tmp;
	uint64_t old;
	for(size_t i = args->start; i < args->end; i++){
		old = args->n.bp[i];
		for(dir d = left; d <= down; d++){
			tmp = old; /* reset every iteration: movedir_unstable can leave the
			            * board rotated even on failure (up/down rotate before
			            * the lock check), which would corrupt the input for
			            * subsequent directions. */
			if(movedir_unstable(&tmp, d)){
				canonicalize_b(&tmp); // TODO it's not necessary to gen *all* boards in nox
				push_back(&args->nret, tmp);
			}
		}
	}
	qs_sort_h(args->nret.bp, args->nret.sp - args->nret.bp);
	return NULL;
}

void *generation_thread_movep(void* data){ // n, nret, stsl, ltc, smallest_large
	arguments *args = data;
	uint64_t tmp;
	uint64_t old;
	for(size_t i = args->start; i < args->end; i++){
		old = args->n.bp[i];
		if(prune_board(old, args->stsl, args->ltc, args->smallest_large))
			continue;
		for(dir d = left; d <= down; d++){
			tmp = old; /* reset every iteration -- see comment in
			            * generation_thread_move() above. */
			if(movedir_unstable(&tmp, d)){
				if(prune_board(tmp, args->stsl, args->ltc, args->smallest_large))
					continue;
				canonicalize_b(&tmp);
				push_back(&args->nret, tmp);
			}
		}
	}
	qs_sort_h(args->nret.bp, args->nret.sp - args->nret.bp);
	return NULL;
}

void *generation_thread_spawn(void* data){
	arguments *args = data;
	uint64_t tmp;
	uint64_t old;
	for(size_t i = args->start; i < args->end; i++){
		old = args->n.bp[i];
		for(int tile = 0; tile < 16; tile++){
			if(GET_TILE(old, tile) == 0){
				tmp = old;
				SET_TILE(tmp, tile, 1);
				canonicalize_b(&tmp);
				push_back(&args->n2, tmp);
				tmp = old;
				SET_TILE(tmp, tile, 2);
				canonicalize_b(&tmp);
				push_back(&args->n4, tmp);
			}
		}
	}
	qs_sort_h(args->n2.bp, args->n2.sp - args->n2.bp);
	qs_sort_h(args->n4.bp, args->n4.sp - args->n4.bp);
	return NULL;
}

static void init_threads(const dynamic_arr_info *n, const unsigned int core_count, enum thread_op op, arguments *cores, char nox) {
	// TODO make these ops work with solving too?
	void *(*fn)(void*);
	switch(op){
	case movep:
		fn = generation_thread_movep;
		break;
	case move:
		fn = generation_thread_move;
		break;
	case spawn:
		fn = generation_thread_spawn;
		break;
	}
	for(unsigned i = 0; i < core_count; i++){ // initialize worker threads
		cores[i].n = (static_arr_info){.valid = n->valid, .bp = n->bp, .size = n->sp - n->bp};
		switch(op){
		case movep:
			cores[i].stsl = get_settings()->stsl;
			cores[i].ltc = get_settings()->ltc;
			__attribute__ ((fallthrough));
		case move:
			cores[i].nret = init_darr(0, 3 * (n->sp - n->bp) / core_count);
			break;
		case spawn:
			cores[i].n2 = init_darr(0, 4 * (n->sp - n->bp) / core_count);
			cores[i].n4 = init_darr(0, 4 * (n->sp - n->bp) / core_count);
			cores[i].nox = nox;
			break;
		}
		// divide up [0,n.size)
		// cores work in [start,end)
		int block_size = (n->sp - n->bp) / core_count;
		cores[i].start = i * block_size;
		cores[i].end = (i + 1) * block_size;
		// core_count * n.size / core_count = n.size
		// make sure that the last thread covers all of the array
		if(i + 1 == core_count){
			cores[i].end = n->sp - n->bp;
		}
	}
	for(unsigned i = 0; i < core_count; i++){
		[[maybe_unused]] int e;
		log_out("Creating thread", LOG_TRACE);
		if((e = pthread_create(&cores[i].thread, NULL, fn, (void*)(cores + i)))){
#ifndef NOERRCHECK
			log_out("Failed creating thread!", LOG_ERROR);
			exit(EXIT_FAILURE);
#endif
		}
	}
}

static void wait(arguments *cores, size_t core_count){
	for(size_t i = 0; i < core_count; i++){
		pthread_join(cores[i].thread, NULL);
	}
}

dynamic_arr_info *get_darr_arr_and(const arguments *cores, const size_t core_count, const size_t extra, const bool n2){
	dynamic_arr_info *arrs = malloc(sizeof(dynamic_arr_info) * (core_count + extra));
	if(!arrs){
		log_out("Couldn't allocate arrays!", LOG_ERROR);
		exit(EXIT_FAILURE);
	}
	for(size_t i = 0; i < core_count; i++){
		if(n2){
			arrs[i] = cores[i].n2;
		}
		else{
			arrs[i] = cores[i].n4;
		}
	}
	return arrs;
}
dynamic_arr_info *get_darr_arr(const arguments *cores, const size_t core_count){
	dynamic_arr_info *arrs = malloc(sizeof(dynamic_arr_info) * core_count);
	if(!arrs){
		log_out("Couldn't allocate arrays!", LOG_ERROR);
		exit(EXIT_FAILURE);
	}
	for(size_t i = 0; i < core_count; i++){
		arrs[i] = cores[i].nret;
	}
	return arrs;
}

static void replace_n(dynamic_arr_info *n, arguments *cores, const unsigned int core_count){
	wait(cores, core_count);
	end_node(GEN_MOVE);
	start_node(COMBINE_MOVE);
	start_node(DEDUPE_MOVE);
	destroy_darr(n);
	dynamic_arr_info *arrs = get_darr_arr(cores, core_count);

	*n = deduplicate_threads(arrs, core_count);
	for(size_t i = 0; i < core_count; i++){
		destroy_darr(arrs + i);
	}
	free(arrs);
}

#ifdef USE_OPENCL
static bool try_dev_resident_layer(dynamic_arr_info* n,
                                   dynamic_arr_info* n2,
                                   dynamic_arr_info* n4,
                                   const unsigned core_count,
                                   const char *fmt_dir,
                                   const int layer,
                                   arguments *cores,
                                   char nox)
{
	bool prune = get_settings()->prune;
	start_node(MOVE);
	cl_darr *moved = NULL;
	bool ok = cl_move_dev(n->bp,
	                     (size_t)(n->sp - n->bp),
	                     prune,
	                     (long)get_settings()->stsl,
	                     (long)get_settings()->ltc,
	                     (long)get_settings()->smallest_large,
	                     &moved);
	end_node(MOVE);
	if (!ok) return false;

	/* Copy device move result to host for write_boards. cl_spawn_dev
	 * still consumes the device handle. */
	dynamic_arr_info host_moved = { .valid = false };
	if (!cl_darr_to_host(moved, &host_moved)) {
		cl_darr_release(moved);
		return false;
	}
	destroy_darr(n);
	*n = host_moved;

	start_node(SPAWN);
	start_node(GEN_SPAWN);

	cl_darr *gpu_n2 = NULL, *gpu_n4 = NULL;
	bool spawn_ok = cl_spawn_dev(moved, &gpu_n2, &gpu_n4);
	/* moved is consumed (released) by cl_spawn_dev whether it
	 * succeeded or not, per its contract. */
	moved = NULL;

	start_node(WRITE);
	write_boards((static_arr_info){.valid = n->valid, .bp = n->bp, .size = n->sp - n->bp}, fmt_dir, layer);
	end_node(WRITE);
	end_node(GEN_SPAWN);

	start_node(COMBINE_SPAWN);
	start_node(DEDUPE_SPAWN);

	if (spawn_ok) {
		dynamic_arr_info host_n2 = { .valid = false };
		dynamic_arr_info host_n4 = { .valid = false };
		bool d2 = cl_darr_download(gpu_n2, &host_n2);
		bool d4 = cl_darr_download(gpu_n4, &host_n4);
		if (!d2 || !d4) {
			if (d2) destroy_darr(&host_n2);
			if (d4) destroy_darr(&host_n4);
			end_node(COMBINE_SPAWN);
			end_node(DEDUPE_SPAWN);
			end_node(SPAWN);
			log_out("cl_darr_download failed; falling back to CPU spawn.", LOG_WARN);
			goto cpu_spawn_fallback;
		}

		dynamic_arr_info arrs2[2] = { host_n2, *n2 };
		*n2 = deduplicate_threads(arrs2, 2);
		destroy_darr(&arrs2[0]);
		destroy_darr(&arrs2[1]);

		dynamic_arr_info arrs4[2] = { host_n4, *n4 };
		*n4 = deduplicate_threads(arrs4, 2);
		destroy_darr(&arrs4[0]);
		destroy_darr(&arrs4[1]);

		end_node(COMBINE_SPAWN);
		end_node(DEDUPE_SPAWN);
		end_node(SPAWN);
		return true;
	}

cpu_spawn_fallback:
	/* Move already happened on device; *n holds the host copy of the
	 * sorted+deduped move result. Run CPU spawn on it. */
	log_out("cl_spawn_dev failed; running CPU spawn on device-resident move output.", LOG_DBG);
	init_threads(n, core_count, spawn, cores, nox);
	wait(cores, core_count);
	{
		dynamic_arr_info *arrs = get_darr_arr_and(cores, core_count, 1, true);
		arrs[core_count] = *n2;
		*n2 = deduplicate_threads(arrs, core_count + 1);
		for(size_t i = 0; i < core_count + 1; i++) destroy_darr(arrs + i);
		free(arrs);
		arrs = get_darr_arr_and(cores, core_count, 1, false);
		arrs[core_count] = *n4;
		*n4 = deduplicate_threads(arrs, core_count + 1);
		for(size_t i = 0; i < core_count + 1; i++) destroy_darr(arrs + i);
		free(arrs);
	}
	end_node(COMBINE_SPAWN);
	end_node(DEDUPE_SPAWN);
	end_node(SPAWN);
	return true;
}
#endif /* USE_OPENCL */

void generate_layer(dynamic_arr_info* n, dynamic_arr_info* n2, dynamic_arr_info* n4,
		const unsigned core_count, const char *fmt_dir, const int layer, arguments *cores, char nox){
#ifdef USE_OPENCL
	const bool use_gpu = get_settings()->use_gpu;
	if (use_gpu) {
		if (try_dev_resident_layer(n, n2, n4, core_count, fmt_dir, layer, cores, nox))
			return;
		log_out("Device-resident layer path failed; falling back to old path.", LOG_WARN);
	}
#endif
	start_node(MOVE);
	start_node(GEN_MOVE);

#ifdef USE_OPENCL
	{
		dynamic_arr_info gpu_out = { .valid = false };
		if (use_gpu) {
			bool prune = get_settings()->prune;
			gpu_out = cl_move_boards(
				n->bp,
				(size_t)(n->sp - n->bp),
				prune,
				(long)get_settings()->stsl,
				(long)get_settings()->ltc,
				(long)get_settings()->smallest_large);
		}
		if(gpu_out.valid){
			end_node(GEN_MOVE);
			start_node(COMBINE_MOVE);
			start_node(DEDUPE_MOVE);
			destroy_darr(n);
			/* deduplicate_threads expects per-thread sorted arrays.
			 * cl_move_boards returns a single sorted array, so feed it
			 * as a one-element arrs[] to keep the dedupe path identical. */
			dynamic_arr_info one[1] = { gpu_out };
			*n = deduplicate_threads(one, 1);
			destroy_darr(&one[0]);
		} else {
			if (use_gpu)
				log_out("OpenCL move failed; falling back to CPU pthread path.", LOG_WARN);
			if(get_settings()->prune)
				init_threads(n, core_count, movep, cores, nox);
			else
				init_threads(n, core_count, move, cores, nox);
			replace_n(n, cores, core_count);
		}
	}
#else
	if(get_settings()->prune)
		init_threads(n, core_count, movep, cores, nox);
	else
		init_threads(n, core_count, move, cores, nox);
	replace_n(n, cores, core_count);
#endif

	end_node(COMBINE_MOVE);
	end_node(DEDUPE_MOVE);
	end_node(MOVE);

	start_node(SPAWN);
	start_node(GEN_SPAWN);

#ifdef USE_OPENCL
	{
		dynamic_arr_info gpu_n2 = { .valid = false };
		dynamic_arr_info gpu_n4 = { .valid = false };
		bool gpu_ok = use_gpu &&
		              cl_spawn_boards(n->bp,
		                              (size_t)(n->sp - n->bp),
		                              &gpu_n2, &gpu_n4);

		start_node(WRITE);
		write_boards((static_arr_info){.valid = n->valid, .bp = n->bp, .size = n->sp - n->bp}, fmt_dir, layer);
		end_node(WRITE);
		end_node(GEN_SPAWN);
		start_node(COMBINE_SPAWN);
		start_node(DEDUPE_SPAWN);

		if (gpu_ok) {
			/* Merge GPU result with the n2/n4 carried over from the
			 * previous layer. deduplicate_threads expects sorted inputs;
			 * the GPU helper already sorted gpu_n2/gpu_n4. */
			dynamic_arr_info arrs2[2] = { gpu_n2, *n2 };
			*n2 = deduplicate_threads(arrs2, 2);
			destroy_darr(&arrs2[0]);
			destroy_darr(&arrs2[1]);

			dynamic_arr_info arrs4[2] = { gpu_n4, *n4 };
			*n4 = deduplicate_threads(arrs4, 2);
			destroy_darr(&arrs4[0]);
			destroy_darr(&arrs4[1]);
		} else {
			if (use_gpu)
				log_out("OpenCL spawn failed; falling back to CPU pthread path.", LOG_WARN);
			init_threads(n, core_count, spawn, cores, nox);
			wait(cores, core_count);
			dynamic_arr_info *arrs = get_darr_arr_and(cores, core_count, 1, true);
			arrs[core_count] = *n2;
			*n2 = deduplicate_threads(arrs, core_count + 1);
			for(size_t i = 0; i < core_count + 1; i++) destroy_darr(arrs + i);
			free(arrs);
			arrs = get_darr_arr_and(cores, core_count, 1, false);
			arrs[core_count] = *n4;
			*n4 = deduplicate_threads(arrs, core_count + 1);
			for(size_t i = 0; i < core_count + 1; i++) destroy_darr(arrs + i);
			free(arrs);
		}
	}
#else
	init_threads(n, core_count, spawn, cores, nox);
	// write while waiting for spawns
	
	start_node(WRITE);

	write_boards((static_arr_info){.valid = n->valid, .bp = n->bp, .size = n->sp - n->bp}, fmt_dir, layer);
	
	end_node(WRITE);
	end_node(GEN_SPAWN);
	start_node(COMBINE_SPAWN);
	start_node(DEDUPE_SPAWN);

	wait(cores,core_count);
	dynamic_arr_info *arrs = get_darr_arr_and(cores, core_count, 1, true);
	arrs[core_count] = *n2;
	*n2 = deduplicate_threads(arrs, core_count + 1);
	for(size_t i = 0; i < core_count + 1; i++){
		destroy_darr(arrs + i);
	}
	free(arrs);
	arrs = get_darr_arr_and(cores, core_count, 1, false);
	arrs[core_count] = *n4;
	*n4 = deduplicate_threads(arrs, core_count + 1);
	for(size_t i = 0; i < core_count + 1; i++){
		destroy_darr(arrs + i);
	}
	free(arrs);
#endif

	end_node(COMBINE_SPAWN);
	end_node(DEDUPE_SPAWN);
	end_node(SPAWN);
}
void generate(const int start, const int end, const char *fmt, const static_arr_info *initial){
	// GENERATE: write all sub-boards where it is the computer's move
	open_bench("bench/"VERSION_STR".gv", "generate");

	generate_lut();
#ifdef USE_OPENCL
	if(get_settings()->use_gpu && cl_init()) cl_upload_luts();
#endif
	static const size_t DARR_INITIAL_SIZE = 100;
	long long core_count = get_settings()->min.cores;
	long long nox = get_settings()->min.nox;
	dynamic_arr_info n  = init_darr(false, 0);
	free(n.bp);
	n.bp = malloc_errcheck(initial->size * sizeof(uint64_t));
	memcpy(n.bp, initial->bp, initial->size * sizeof(uint64_t));
	n.size = initial->size;
	n.sp = n.size + n.bp;
	dynamic_arr_info n2 = init_darr(false, DARR_INITIAL_SIZE);
	dynamic_arr_info n4 = init_darr(false, DARR_INITIAL_SIZE);
	arguments *cores = malloc_errcheck(sizeof(arguments) * core_count);
	if(get_settings()->premove){
		arguments premove_args;
		premove_args.n = (static_arr_info){.valid = n.valid, .bp = n.bp, .size = n.sp - n.bp};
		premove_args.n2 = n2;
		premove_args.n4 = n4;
		premove_args.start = 0;
		premove_args.end = premove_args.n.size;
		generation_thread_move(&premove_args);
	}
	for(int i = start; i <= end; i += 2){
		set_layer(i);
		start_node(GEN_LAYER);

		generate_layer(&n, &n2, &n4, core_count, fmt, i, cores, nox);

		end_node(GEN_LAYER);

		destroy_darr(&n);
		n = n2;
		n2 = n4;
		n4 = init_darr(false, DARR_INITIAL_SIZE);
	}
	destroy_darr(&n);
	destroy_darr(&n2);
	destroy_darr(&n4);
	free(cores);
#ifdef USE_OPENCL
	{
		size_t vram = cl_vram_bytes();
		logf_out("Peak VRAM use: %.2f MiB", LOG_INFO, vram / (1024.0 * 1024.0));
	}
	cl_shutdown();
#endif
	{
		struct rusage ru;
		if (getrusage(RUSAGE_SELF, &ru) == 0) {
			/* Linux reports ru_maxrss in KiB; macOS in bytes. We're on
			 * Linux per the build target, so multiply by 1024 for bytes. */
			double mib = (double)ru.ru_maxrss / 1024.0;
			logf_out("Peak host RSS:  %.2f MiB", LOG_INFO, mib);
		}
	}

	end_bench();
}
static_arr_info read_boards(const char *dir){
	FILE *fp = fopen(dir, "rb");
	if(fp == NULL){
		goto fail;
	}
	fseek(fp, 0L, SEEK_END);
	size_t sz = ftell(fp);
	rewind(fp);
	if(sz % 8 != 0)
		log_out("sz %%8 != 0, this is probably not a real table!", LOG_WARN);
	uint64_t* data = malloc_errcheck(sz);
	if(fread(data, 1, sz, fp) != sz){
		goto fail;
	}
	logf_out("Read %ld bytes (%ld boards) from %s", LOG_INFO, sz, sz / 8, dir);
	fclose(fp);
	static_arr_info res = {true, data, sz / 8}; 
#ifdef DBG
	uint64_t tmp;
	for(size_t i = 0; i < sz / 8; i++){
		tmp = res.bp[i];
		canonicalize_b(&tmp);
		if(tmp != res.bp[i]){
			log_out("Reading non canonicalized board!!!!", LOG_WARN);
		}
	}
	
#endif
	return res;
fail:
	logf_out("Couldn't read %s!", LOG_WARN, dir);
	return (static_arr_info){.valid = false};
}

void read_boards2(static_arr_info *b, const char *dir){
	*b = read_boards(dir);
}

double pun_uint64(uint64_t num){
	return *(double*)(&num);
}
