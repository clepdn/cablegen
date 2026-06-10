#ifndef SETTINGS_H
#define SETTINGS_H
#include <stdbool.h>
#define REMOVE_STRUCT_DECL
#include "array.h"
#include "board.h"
#include "solve.h"
#include "../api/cablegen.h"
#include "cfgpath.h"
extern bool settings_read;
extern bool custom_path;
extern char cfgdir[MAX_PATH];
char *strlwr_(char *str);
void change_config(char *cfg);
typedef struct {
	min_settings_t min;

	bool delete_boards;
	bool mask;
	bool compress;
	long long max_prealloc;
	bool use_gpu;

	bool premove;
	long long stsl;
	long long ltc;
	long long smallest_large;
	bool prune;

	bool score;
} settings_t;

settings_t *get_settings(void);
void init_settings(void);

#endif
