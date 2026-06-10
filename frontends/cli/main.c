#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include "cablegen.h"
#include "parse.h"
#define LOG_H_IMPLEMENTATION
#define LOG_H_ENUM_PREFIX_
#define LOG_H_NAMESPACE_ 
#include "logging.h"
#include "format.h"
#include <string.h>
#ifndef WIN32
#include <termios.h>
#endif
#ifdef WIN32
#include <conio.h>
#endif
#include <errno.h>
#include <time.h>

static int parseCfg(char *arg, void *data){
	option_t *opt = data;
	size_t eq = opt->eq;
	char *buf = malloc(strlen(arg) - eq);
	if(!buf){
		printf("Could not allocate string buffer for argument!\n");
		return -1;
	}
	strcpy(buf, arg + eq + 1);
	change_config(buf);
	free(buf);
	return 0;
}

static void help(void){
	logf_out("Tablegen frontend by ember/emelia/cattodoameow running %s", LOG_INFO, get_version());
	log_out("Usage: [cablegen] [flags] [command]", LOG_INFO); // TODO I would like these to be self-documenting somehow
	log_out("Flags:", LOG_INFO);
	log_out("-h          --help               -- this output", LOG_INFO);
	log_out("-C=[FILE]   --config             -- specifies a config to read flags from, behaviour if unspecified is specified in the README", LOG_INFO);
	log_out("-f=[bool]   --free-formation     -- specifies whether unmergable (0xf) tiles should be allowed to move", LOG_INFO);
	log_out("-v=[bool]   --ignore-unmergeable -- treats unmergable tiles as walls", LOG_INFO);
	log_out("-c=[int]    --cores              -- chooses the amount of cores to use while generating or solving", LOG_INFO);
	log_out("-n=[int]    --nox                -- treats a tile with this tile on it as dead (0 to disable)", LOG_INFO);
	log_out("-b=[string] --bdir               -- select a directory for where to put generated positions", LOG_INFO);
	log_out("-t=[string] --tdir               -- select a directory for where to put solved tables", LOG_INFO);
	log_out("-i=[string] --initial            -- select a file to start generation from", LOG_INFO);
	log_out("-e=[int]    --end-gen            -- choose the tile sum to end generation", LOG_INFO);
	log_out("-E=[int]    --end-solve          -- choose the tile sum to end solving", LOG_INFO);
	log_out("-w=[string] --winstate           -- select a file to consider winstates from", LOG_INFO);
	log_out("-d=[bool]   --delete             -- delete positions after solving them", LOG_INFO);
	log_out("Commands:", LOG_INFO);
	log_out("help                             -- this output", LOG_INFO);
	log_out("generate                         -- generate boards", LOG_INFO);
	log_out("solve                            -- solve boards backwards", LOG_INFO);
	log_out("generate_solve                   -- generate boards and then solves", LOG_INFO);
	log_out("test                             -- run self test", LOG_INFO);
	log_out("read [FILE]                      -- read board(s) from a file", LOG_INFO);
	log_out("write [FILE] [BOARD] (BOARD) ... -- write board(s) to a file", LOG_INFO);
	log_out("lookup [BOARD]                   -- look a board up", LOG_INFO);
	log_out("lookup_spawn [BOARD]             -- look a board up", LOG_INFO);
	log_out("explore [TABLE]                  -- show all the solved boards in TABLE", LOG_INFO);
	log_out("train [BOARD]                    -- play a game starting with BOARD, optionally specifying CONFIG, with live feedback", LOG_INFO);
	log_out("play [BOARD]                     -- simulate an optimal game of BOARD with random spawns", LOG_INFO);
	log_out("benchmark                        -- benchmark cablegen", LOG_INFO);
}

#if __has_include(<unistd.h>)
#include <unistd.h>
#elif __has_include(<windows.h>)
#include <windows.h>
#else
#warning "Sleep will not work on non posix, non-windows systems!"
#endif
static void parseGenerate(){
	set_log_level(LOG_INFO);
	min_settings_t settings = *get_settings_min();
	char *default_postfix = "%d.boards";
	char *fmt = format_str("%s%s", settings.bdir, default_postfix);

	// assume all boards in our file are the same sum (!!)
	static_arr_info boards = read_boards(settings.initial);
	for(size_t i = 0; i < boards.size; i++){
		if(get_sum(boards.bp[i]) != get_sum(boards.bp[0])){
			log_out("Boards in Generate.initial have mixed sums, refusing to generate!", LOG_ERROR);
			exit(EXIT_FAILURE);
		}
	}
	if(boards.size < 1){
		logf_out("No boards in %s!", LOG_ERROR, settings.initial);
		return;
	}

	if(get_sum(boards.bp[0]) < settings.end_solve){
		log_out("Solving will continue below initial board, consider changing Solve.end!", LOG_WARN);
		log_out("Press Ctrl+C to cancel, otherwise beginning generation in 5 seconds...", LOG_WARN);
		// TODO: platform independent sleep
#if __has_include(<unistd.h>)
		sleep(5);
#elif __has_include(<windows.h>)
		Sleep(5000);
#else
#warning "Sleep will not work on non posix, non-windows systems!"
		log_out("Just kidding! I don't know how to sleep on your stupid OS!", LOG_WARN);
#endif
	}
	int layer = get_sum(boards.bp[0]);
	generate(layer, settings.end_gen, fmt, &boards);
	free(fmt);
}

static void parseSolve(){
	set_log_level(LOG_INFO);
	min_settings_t settings = *get_settings_min();
	char *default_board_postfix = "%d.boards";
	char* posfmt = format_str("%s%s", settings.bdir, default_board_postfix);

	char* default_table_postfix = "%d.tables";
	char* table_fmt = format_str("%s%s", settings.tdir, default_table_postfix);

	static_arr_info boards = read_boards(settings.winstates);
	if(boards.size < 1){
		log_out(format_str("No boards in %s!", settings.initial), LOG_WARN);
		return;
	}
	for(size_t i = 0; i < boards.size; i++){
		if(get_sum(boards.bp[i]) > get_settings_min()->end_gen){
			log_out("Winstate will never be generated, consider changing Generate.end!", LOG_WARN);
			output_board(boards.bp[i]);
		}
	}
	solve(settings.end_gen, settings.end_solve, posfmt, table_fmt, &boards);
	free(posfmt);
	free(table_fmt);
}

static void parseWrite(int argc, char **argv){
	if(argc < 2){ log_out("Not enough arguments!", LOG_ERROR); help(); exit(EXIT_FAILURE); }
	dynamic_arr_info boards = init_darr(0, 1);
	for(int i = 2; i < argc; i++){
		errno = 0;
		uint64_t board = strtoull(argv[i], NULL, 16); // interpret as hex string
		if (errno != 0){
			printf("Error parsing string! %s\n", strerror(errno));
		}
		push_back(&boards, board);
	}
	write_boards((static_arr_info){.valid = boards.valid, .bp = boards.bp, .size = boards.sp - boards.bp}, argv[1], 0);
}

struct dirprob {
	dir d;
	double prob;
};

static char* dirtos(dir d){
	switch(d){
		default:
		case left:
			return "left\0";
		case right:
			return "right\0";
		case up:
			return "up\0";
		case down:
			return "down\0";
	}
}

static struct dirprob best(uint64_t board, table *n){
	uint64_t tmp = board;
	dir maxd = left;
	double maxp = 0;
	double tmpp = 0;
	for(dir d = left; d <= down; d++){
		tmp = board;
		if(!movedir(&tmp, d))
			continue;
		if((tmpp = lookup(tmp, n, true)) > maxp){
			maxd = d;
			maxp = tmpp;
		}
	}
	return (struct dirprob){maxd, maxp};
}

static void parseLookup(int argc, char **argv, bool spawn){ // TODO refactor
	if(argc < 2){ log_out("Not enough arguments!", LOG_ERROR); help(); exit(EXIT_FAILURE); }
	char *tabledir;
	min_settings_t settings = *get_settings_min();
	tabledir = settings.tdir;
	uint64_t board = strtoull(argv[1], NULL, 16); // interpret as hex string
	uint64_t original = board;
	set_log_level(LOG_INFO);
	if(spawn){ // this board has a spawn on it
		for(int i = 0; i < 16; i++){
			if(GET_TILE(board, i) <= 2){
				SET_TILE(board, i, 0);
				break;
			}
		}
	}
	generate_lut();
	uint64_t sum = get_sum(board);

	char* default_table_postfix = "%d.tables";
	char* table_fmt = format_str("%s%s", tabledir, default_table_postfix);
	char* tablestr = format_str(table_fmt, sum);
	log_out(tablestr, LOG_TRACE);
	
	table *t  = malloc_errcheck(sizeof(table));
	table *t2 = malloc_errcheck(sizeof(table));
	table *t4 = malloc_errcheck(sizeof(table));
	read_table(t, tablestr);
	double res = lookup(board, t, true);
	free(tablestr);
	tablestr = format_str(table_fmt, sum + 2);
	read_table(t2, tablestr);
	free(tablestr);
	tablestr = format_str(table_fmt, sum + 4);
	read_table(t4, tablestr);


	if(spawn)
		output_board(original);
	else{
		logf_out("Board(%0.10lf):\n", LOG_INFO, res);
		output_board(board);
		logf_out("Spawns:\n", LOG_INFO);
	}
	for(int i = 0; i < 16; i++){
		if(GET_TILE(board, i))
			continue;
		SET_TILE(board, i, 1);
		struct dirprob pb = best(board, t2);
		if(!spawn || original == board){
			printf("Best move: %s (%.10lf)\n", dirtos(pb.d), pb.prob);
			if(!spawn)
				output_board(board);
		}
		SET_TILE(board, i, 2);
		pb = best(board, t4);
		if(!spawn || original == board){
			printf("Best move: %s (%.10lf)\n", dirtos(pb.d), pb.prob);
			if(!spawn)
				output_board(board);
		}
		SET_TILE(board, i, 0);
	}
	free(table_fmt);
	free(tablestr);
	free(t->key.bp);
	free(t->value.bp);
	free(t);
	free(t2->key.bp);
	free(t2->value.bp);
	free(t2);
	free(t4->key.bp);
	free(t4->value.bp);
	free(t4);
}

static void parseLookupSpawn(int argc, char **argv){
	parseLookup(argc, argv, true);
}
static void parseLookupMove(int argc, char **argv){
	parseLookup(argc, argv, false);
}


static void parseExplore(int argc, char **argv){
	if(argc < 1){ log_out("Not enough arguments!", LOG_ERROR); help(); exit(EXIT_FAILURE); }
	table *t = malloc_errcheck(sizeof(table));
	read_table(t, argv[1]);
	for(size_t i = 0; i < t->key.size; i++){
		printf("Board(%0.20lf):\n", *(double*)(t->value.bp + i));
		output_board(t->key.bp[i]);
	}
	free(t->key.bp);
	free(t->value.bp);
	free(t);
}

static void parseRead(int argc, char **argv){
	if(argc < 1){ log_out("Not enough arguments!", LOG_ERROR); help(); exit(EXIT_FAILURE); }

	static_arr_info t = read_boards(argv[1]);
	for(size_t i = 0; i < t.size; i++){
		output_board(t.bp[i]);
		if(i + 1 < t.size)
			printf("----\n");
	}
	free(t.bp);
}

static bool canMove(uint64_t board){
	for(dir d = left; d <= down; d++){
		if(movedir(&board, d))
			return true;
	}
	return false;
}

static void spawn(uint64_t *board){
	short spaces = 0;
	for(int i = 0; i < 16; i++){
		if(!GET_TILE((*board), i))
			spaces++;
	}
	if(spaces == 0){
		log_out("No space!", LOG_INFO);
		output_board(*board);
		exit(0);
	}
	short position = rand() % spaces;
	short tile = rand() % 10;
	short curr = 0;
	for(int i = 0; i < 16; i++){
		if(!GET_TILE((*board), i) && curr++ == position){
			if(tile == 9){
				SET_TILE((*board), i, 2);
			}
			else
				SET_TILE((*board), i, 1);
		}
	}
}
char getch__(void) // https://stackoverflow.com/a/7469410 
{
#ifndef WIN32
	bool echo = false;
	static struct termios current, old;
	tcgetattr(0, &old); /* grab old terminal i/o settings */
	current = old; /* make new settings same as old settings */
	current.c_lflag &= ~ICANON; /* disable buffered i/o */
	if (echo) {
	  current.c_lflag |= ECHO; /* set echo mode */
	} else {
	  current.c_lflag &= ~ECHO; /* set no echo mode */
	}
	tcsetattr(0, TCSANOW, &current); /* use these new terminal i/o settings now */
	char res = getchar();
	tcsetattr(0, TCSANOW, &old);
	return res;
#endif
#ifdef WIN32
	return getch();
#endif
}
static void parseTrain(int argc, char **argv){
	if(argc < 1){ log_out("Not enough arguments!", LOG_ERROR); help(); exit(EXIT_FAILURE); }
	set_log_level(LOG_INFO);
	srand(time(0));
	generate_lut();
	double cumacc = 1;
	bool validMove = 0;
	char inp;

	char* default_table_postfix = "%d.tables"; // TODO pull out
	char* table_fmt = format_str("%s%s", get_settings_min()->tdir, default_table_postfix);

	char *table_dir_str;
	table *t  = malloc_errcheck(sizeof(table));
	uint64_t board = strtoull(argv[1], NULL, 16);

	struct dirprob res;
	double tmp = 0;
	log_out("Welcome to cablegen training mode! Move with WASD or HJKL", LOG_INFO);
	while(canMove(board)){
		printf("-------------\n");
		output_board(board);
		uint64_t premove = board;
		dir move = 0;
		do{
			inp = getch__();
			switch(inp){
			case 'w':
			case 'k':
				move = up;
				validMove = true;
				break;
			case 'a':
			case 'h':
				move = left;
				validMove = true;
				break;
			case 's':
			case 'j':
				move = down;
				validMove = true;
				break;
			case 'd':
			case 'l':
				move = right;
				validMove = true;
				break;
			default:
				validMove = false;
			}
			if(validMove)
				validMove = movedir(&board, move);
			if(validMove == false)
				log_out("Invalid move!", LOG_INFO);
		}while(!validMove);
		table_dir_str = format_str(table_fmt, get_sum(board));
		read_table(t, table_dir_str);
		free(table_dir_str);
		res = best(premove, t);
		if(tmp == 1){
			log_out("You Win!", LOG_INFO);
			printf("[INFO] Cumulative accuracy: %lf\n", cumacc);
			exit(0);
		}
		if((res.prob - (tmp = lookup(board, t, true))) > 0.005){ // TODO setting?
			log_out("Suboptimal move!", LOG_INFO);
			printf("[INFO] Table: %lf(%s), You %lf(%s)\n", res.prob, dirtos(res.d), tmp, dirtos(move));
			printf("[INFO] Cumulative accuracy: %lf\n", cumacc *= (tmp/res.prob));
			printf("[INFO] One step loss: %lf\n", 1 - (tmp/res.prob));
		}
		else
			cumacc *= (tmp/res.prob);
		spawn(&board);
		free(t->key.bp);
		free(t->value.bp);
	}
	output_board(board);
	log_out("You Died!", LOG_INFO);
	printf("[INFO] Cumulative accuracy: %lf\n", cumacc);
	exit(0);
}

static void parsePlay(int argc, char **argv){
	if(argc < 1){ log_out("Not enough arguments!", LOG_ERROR); help(); exit(EXIT_FAILURE); }
	set_log_level(LOG_INFO);
	min_settings_t settings = *get_settings_min();
	srand(time(NULL));
	char *tdir = settings.tdir;
	uint64_t board = strtoull(argv[1], NULL, 16); // interpret as hex string
	generate_lut();

	char* default_table_postfix = "%d.tables"; // TODO pull out
	char* table_fmt = format_str("%s%s", tdir, default_table_postfix);

	char *table_dir_str;
	table *t  = malloc_errcheck(sizeof(table));

	do{
		output_board(board);
		if(!canMove(board)){
			log_out("AI died :(", LOG_INFO);
			exit(0);
		}
		printf("\n");
		table_dir_str = format_str(table_fmt, get_sum(board));
		read_table(t, table_dir_str);
		free(table_dir_str);
		struct dirprob tmp = best(board, t);
		movedir(&board, tmp.d);
		if(tmp.prob == 1.0f){
			log_out("AI won!", LOG_INFO);
			exit(0);
		}
		free(t->key.bp);
		free(t->value.bp);
		spawn(&board);
	}
	while (true);
	free(table_fmt);
	free(table_dir_str);
	free(t);
}

void benchmark(void){
	dynamic_arr_info initial_arr = init_darr(0,2);
	time_t curr = time(NULL);
	log_out("Benchmarking multi-threaded generation (LL-256)", LOG_INFO);
	push_back(&initial_arr, 0x1000000021ff12ff);
	push_back(&initial_arr, 0x1000000012ff21ff);
	static_arr_info initial_arr_sarr;
	initial_arr_sarr.bp = initial_arr.bp;
	initial_arr_sarr.size = initial_arr.sp - initial_arr.bp;
	initial_arr_sarr.valid = true;
	set_log_level(LOG_WARN);
	generate(16, 300, ".benchmark/%d.boards", &initial_arr_sarr);
	set_log_level(LOG_INFO);
	printf("Multi-threaded LL-256: %d seconds\n", (unsigned)difftime(time(NULL), curr));
	static_arr_info winstates = init_sarr(0,1);
	winstates.bp[0] = 0x0000000008ff0ff;
	log_out("Benchmarking multi-threaded solving (LL-256)", LOG_INFO); // TODO add profiling for solving
	curr = time(NULL);
	set_log_level(LOG_WARN);
	solve(300, 16, ".benchmark/%d.boards", ".benchmark/%d.tables", &winstates);
	set_log_level(LOG_INFO);
	printf("Multi-threaded LL-256: %d seconds\n", (unsigned)difftime(time(NULL), curr));
}

static void parseGenSolve(){
	parseGenerate();
	parseSolve();
}

static void testThunk(){
	test(); // ignore the bool return value
}

int main(int argc, char **argv){
	set_log_level(LOG_INFO);
	int commandOffset = -1;
	struct {
		char *name;
		bool args;
		union {
			void (*noargs)(void);
			void (*args)(int, char**);
		} fn;
	} commands[] = {
		{"help", false, .fn.noargs = help},
		{"generate", false, .fn.noargs = parseGenerate},
		{"solve", false, .fn.noargs = parseSolve},
		{"generate_solve", false, .fn.noargs = parseGenSolve},
		{"test", false, .fn.noargs = testThunk},
		{"read", true, .fn.args = parseRead},
		{"write", true, .fn.args = parseWrite},
		{"lookup", true, .fn.args = parseLookupMove},
		{"lookup_spawn", true, .fn.args = parseLookupSpawn},
		{"explore", true, .fn.args = parseExplore},
		{"train", true, .fn.args = parseTrain},
		{"play", true, .fn.args = parsePlay},
		{"benchmark", false, .fn.noargs = benchmark},
	};
	for(int i = 0; i < argc; i++){
		for(size_t cm = 0; cm < sizeof(commands)/sizeof(commands[0]); cm++){
			if(!strcmp(commands[cm].name, argv[i])){
				commandOffset = i;
			}
		}
	}
	if(commandOffset == -1){
		log_out("Command required!", LOG_ERROR);
		help();
		exit(EXIT_FAILURE);
	}
	option_t opts[] = {
		{ parseCfg , "-C", "--config", 0, 0 }, // TODO: this last field shouldn't really be exposed to the frontend of parse.h
		{ parseBool, "-f", "--free-formation", &get_settings_min()->free_formation, 0 },
		{ parseBool, "-v", "--ignore-unmergeable", &get_settings_min()->ignore_f, 0 },
		{ parseLL  , "-c", "--cores", &get_settings_min()->cores, 0 },
		{ parseLL  , "-n", "--nox", &get_settings_min()->nox, 0 },
		{ parseStr , "-t", "--tdir", &get_settings_min()->tdir, 0 },
		{ parseStr , "-b", "--bdir", &get_settings_min()->bdir, 0 },
		{ parseStr , "-i", "--initial", &get_settings_min()->initial, 0 },
		{ parseLL  , "-e", "--end-gen", &get_settings_min()->end_gen, 0 },
		{ parseLL  , "-E", "--end-solve", &get_settings_min()->end_solve, 0 },
		{ parseStr , "-w", "--winstate", &get_settings_min()->winstates, 0 },
		{ parseBool, "-d", "--delete", &get_settings_min()->delete_boards, 0 }
	};
	parse(opts, sizeof(opts)/sizeof(opts[0]), commandOffset, argv);
	for(size_t i = 0; i < sizeof(commands)/sizeof(commands[0]); i++){
		if(!strcmp(commands[i].name, argv[commandOffset])){
			if(commands[i].args){
				commands[i].fn.args(argc - commandOffset, argv + commandOffset);
			}
			else{
				commands[i].fn.noargs();
			}
		}
	}
	exit(EXIT_SUCCESS);
}
