#include "settings.h"
#include "cfgpath.h"
#include <stdbool.h>
#include "ini.h"
#define LOG_H_ENUM_PREFIX_
#define LOG_H_NAMESPACE_ 
#include "logging.h"
#include "array.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#ifdef WINDOWS
#include <direct.h>
#define getcwd_ _getcwd
#else
#include <unistd.h>
#define getcwd_ getcwd
#endif
bool settings_read = false;
bool custom_path = false;
static int get_bool_setting(const char *key, bool*);
static int get_str_setting (const char *key, char**);
static int get_int_setting (const char *key, long long*);

static int get_bool_setting_section(const char *key, char *section, bool*);
static int get_str_setting_section (const char *key, char *section, char**);
static int get_int_setting_section (const char *key, char *section, long long*);

min_settings_t *get_settings_min(void){
	return (min_settings_t*)get_settings();
}

settings_t *get_settings(void){
	static settings_t res = { // set sane defaults
		.min.free_formation = 0,
		.min.ignore_f = 0,
		.min.cores = 1,
		.min.nox = 0,
		.mask = 0,
		.compress = false,
		.use_gpu = true,


		.premove = false,
		.min.bdir = "./boards/",
		.min.initial = "./initial",
		.min.end_gen = 1200,
		.stsl = 200,
		.ltc = 5,
		.smallest_large = 6,
		.prune = false,

		.min.tdir = "./tables/",
		.min.winstates = "./winstates",
		.min.end_solve = 0,
		.score = false,
		.delete_boards = false
	};
	if(settings_read)
		return &res;
	settings_read = true; 
	log_out("Reading settings...", LOG_DBG);
	get_bool_setting("free_formation", &res.min.free_formation);
	get_bool_setting("ignore_f", &res.min.ignore_f);
	get_int_setting("cores", &res.min.cores); 
	get_int_setting("nox", &res.min.nox); 
	get_bool_setting("mask", &res.mask); 
	get_bool_setting("compress", &res.compress);
	get_int_setting("max_prealloc", &res.max_prealloc);
	get_bool_setting("use_gpu", &res.use_gpu);
	get_bool_setting_section("premove", "Generate", &res.premove);
	get_str_setting_section("dir", "Generate", &res.min.bdir);
	get_str_setting_section("initial", "Generate", &res.min.initial);
	get_int_setting_section("end", "Generate", &res.min.end_gen); 
	get_int_setting_section("stsl", "Generate", &res.stsl); 
	get_int_setting_section("ltc", "Generate", &res.ltc); 
	get_int_setting_section("smallest_large", "Generate", &res.smallest_large); 
	get_bool_setting_section("prune", "Generate", &res.prune);
	get_str_setting_section("dir", "Solve", &res.min.tdir);
	get_str_setting_section("winstates", "Solve", &res.min.winstates);
	get_int_setting_section("end", "Solve", &res.min.end_solve); 
	get_bool_setting_section("score", "Solve", &res.score);
	get_bool_setting_section("delete_boards", "Solve", &res.delete_boards);
	return &res;
}

char *strlwr_(char *str) {
  unsigned char *p = (unsigned char *)str;
  while (*p) {
     *p = tolower((unsigned char)*p);
      p++;
  }
  return str;
}
char cfgdir[MAX_PATH];
#define MAX_PROP_SIZE 100
ini_t* get_cfg(void){
	if(!custom_path){
		get_user_config_file(cfgdir, sizeof(cfgdir), "cablegen");
	}
	if (cfgdir[0] == 0) {
		log_out("Could not find config directory!", LOG_WARN);
		return NULL;
	}
	log_out("Loading config from: ", LOG_DBG);
	log_out(cfgdir, LOG_DBG);
	return ini_load(cfgdir);
}
void change_config(char *cfg){	
	if(cfg == NULL || strlen(cfg) >= MAX_PATH){
		log_out("Invalid config file location!", LOG_WARN);
		return;
	}
	settings_read = false; // update settings reading
	custom_path = true;
	memcpy(cfgdir, cfg, strlen(cfg) + 1);
	log_out("New config dir: ", LOG_DBG);
	log_out(cfgdir, LOG_DBG);
}
void init_settings(void){
	char buf[MAX_PATH];
	if(!getcwd_(buf, MAX_PATH)){
		log_out("Could not get current working directory!", LOG_WARN);
		return;
	}
	if(strlen(buf) + strlen("cablegen.conf") >= MAX_PATH || errno){
		log_out("Invalid config file location!", LOG_WARN);
		return;
	}
	log_out("Reading settings from conf dir", LOG_TRACE);
	get_settings();
	strcat(buf, "/cablegen.conf");
	log_out("Reading settings from local dir", LOG_TRACE);
	log_out(buf, LOG_TRACE);
	change_config(buf);
	get_settings();
}
static int get_str_setting(const char *key, char **str){
	return get_str_setting_section(key, "Cablegen", str);
}
static int get_str_setting_section (const char *key, char *section, char** str){
	ini_t* cfg = get_cfg();
	if(cfg == NULL){
		return 1;
	}
	char *res = ini_get(cfg, section, key);
	if(res == NULL){
		log_out("Could not find property!", LOG_DBG);
		log_out(key, LOG_DBG);
		ini_free(cfg);
		return 1;
	}
	char *rres = malloc_errcheck(strlen(res) + 1);
	strcpy(rres, res);
	*str = rres;
	ini_free(cfg);
	return 0;
}
static int get_bool_setting(const char *key, bool *data){
	char *str;
	int e = get_str_setting(key, &str);
	if(e){
		return e;
	}
	strlwr_(str);
	*data = !strcmp(str, "true");
	free(str);
	return 0;
}
static int get_int_setting (const char *key, long long *data){
	char *str;
	int e = get_str_setting(key, &str);
	if(e){
		return e;
	}
	*data = atoll(str);
	free(str);
	return 0;
}

static int get_bool_setting_section(const char *key, char *section, bool *data){
	char *str;
	int e = get_str_setting_section(key, section, &str);
	if(e){
		return e;
	}
	strlwr_(str);
	*data = !strcmp(str, "true");
	free(str);
	return 0;
}
static int get_int_setting_section (const char *key, char *section, long long *data){
	char *str;
	int e = get_str_setting_section(key, section, &str);
	if(e){
		return e;
	}
	*data = atoll(str);
	free(str);
	return 0;
}
