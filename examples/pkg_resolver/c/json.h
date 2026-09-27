/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* Copyright (c) 2026 John William Creighton (@s243a) */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum JsonType {
    JSON_NULL,
    JSON_BOOL,
    JSON_INT,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct JsonPair JsonPair;

struct Json {
    JsonType type;
    bool bool_val;
    int64_t int_val;
    char *str_val;
    struct Json *arr_val;
    size_t arr_len;
    size_t arr_cap;
    JsonPair *obj_val;
    size_t obj_len;
    size_t obj_cap;
};

struct JsonPair {
    char *key;
    struct Json value;
};

typedef struct Json Json;

typedef struct JsonParseError {
    const char *message;
} JsonParseError;

Json json_null(void);
Json json_bool(bool b);
Json json_int(int64_t n);
Json json_string(const char *s);
Json json_string_owned(char *s);
Json json_array(void);
Json json_object(void);

void json_free(Json *v);
Json json_clone(const Json *v);

JsonType json_type(const Json *v);
bool json_is_null(const Json *v);
bool json_is_bool(const Json *v);
bool json_is_int(const Json *v);
bool json_is_string(const Json *v);
bool json_is_array(const Json *v);
bool json_is_object(const Json *v);

bool json_as_bool(const Json *v);
int64_t json_as_int(const Json *v);
const char *json_as_string(const Json *v);
const Json *json_array_items(const Json *v, size_t *out_len);
const JsonPair *json_object_items(const Json *v, size_t *out_len);

const Json *json_find(const Json *v, const char *key);
bool json_has(const Json *v, const char *key);
void json_array_push(Json *arr, Json item);
void json_object_set(Json *obj, const char *key, Json value);
size_t json_size(const Json *v);
const Json *json_at(const Json *v, size_t i);

Json json_parse(const char *text, JsonParseError *err);
char *json_dump(const Json *v);
