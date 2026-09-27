/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#include "json.h"
#include "cstr.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Json json_null(void) {
    Json j;
    memset(&j, 0, sizeof j);
    j.type = JSON_NULL;
    return j;
}

Json json_bool(bool b) {
    Json j = json_null();
    j.type = JSON_BOOL;
    j.bool_val = b;
    return j;
}

Json json_int(int64_t n) {
    Json j = json_null();
    j.type = JSON_INT;
    j.int_val = n;
    return j;
}

Json json_string(const char *s) {
    Json j = json_null();
    j.type = JSON_STRING;
    j.str_val = cstr_dup(s);
    return j;
}

Json json_string_owned(char *s) {
    Json j = json_null();
    j.type = JSON_STRING;
    j.str_val = s ? s : cstr_dup("");
    return j;
}

Json json_array(void) {
    Json j = json_null();
    j.type = JSON_ARRAY;
    return j;
}

Json json_object(void) {
    Json j = json_null();
    j.type = JSON_OBJECT;
    return j;
}

static void json_free_inner(Json *v) {
    if (!v) return;
    switch (v->type) {
    case JSON_STRING:
        free(v->str_val);
        v->str_val = NULL;
        break;
    case JSON_ARRAY:
        for (size_t i = 0; i < v->arr_len; i++)
            json_free(&v->arr_val[i]);
        free(v->arr_val);
        v->arr_val = NULL;
        v->arr_len = v->arr_cap = 0;
        break;
    case JSON_OBJECT:
        for (size_t i = 0; i < v->obj_len; i++) {
            free(v->obj_val[i].key);
            json_free(&v->obj_val[i].value);
        }
        free(v->obj_val);
        v->obj_val = NULL;
        v->obj_len = v->obj_cap = 0;
        break;
    default:
        break;
    }
}

void json_free(Json *v) {
    if (!v) return;
    json_free_inner(v);
    *v = json_null();
}

Json json_clone(const Json *v) {
    if (!v) return json_null();
    switch (v->type) {
    case JSON_NULL: return json_null();
    case JSON_BOOL: return json_bool(v->bool_val);
    case JSON_INT: return json_int(v->int_val);
    case JSON_STRING: return json_string(v->str_val ? v->str_val : "");
    case JSON_ARRAY: {
        Json arr = json_array();
        for (size_t i = 0; i < v->arr_len; i++)
            json_array_push(&arr, json_clone(&v->arr_val[i]));
        return arr;
    }
    case JSON_OBJECT: {
        Json obj = json_object();
        for (size_t i = 0; i < v->obj_len; i++)
            json_object_set(&obj, v->obj_val[i].key, json_clone(&v->obj_val[i].value));
        return obj;
    }
    }
    return json_null();
}

JsonType json_type(const Json *v) { return v ? v->type : JSON_NULL; }
bool json_is_null(const Json *v) { return v && v->type == JSON_NULL; }
bool json_is_bool(const Json *v) { return v && v->type == JSON_BOOL; }
bool json_is_int(const Json *v) { return v && v->type == JSON_INT; }
bool json_is_string(const Json *v) { return v && v->type == JSON_STRING; }
bool json_is_array(const Json *v) { return v && v->type == JSON_ARRAY; }
bool json_is_object(const Json *v) { return v && v->type == JSON_OBJECT; }

bool json_as_bool(const Json *v) { return v && v->type == JSON_BOOL ? v->bool_val : false; }
int64_t json_as_int(const Json *v) { return v && v->type == JSON_INT ? v->int_val : 0; }
const char *json_as_string(const Json *v) { return v && v->type == JSON_STRING && v->str_val ? v->str_val : ""; }

const Json *json_array_items(const Json *v, size_t *out_len) {
    if (!v || v->type != JSON_ARRAY) {
        if (out_len) *out_len = 0;
        return NULL;
    }
    if (out_len) *out_len = v->arr_len;
    return v->arr_val;
}

const JsonPair *json_object_items(const Json *v, size_t *out_len) {
    if (!v || v->type != JSON_OBJECT) {
        if (out_len) *out_len = 0;
        return NULL;
    }
    if (out_len) *out_len = v->obj_len;
    return v->obj_val;
}

const Json *json_find(const Json *v, const char *key) {
    if (!v || v->type != JSON_OBJECT || !key) return NULL;
    for (size_t i = 0; i < v->obj_len; i++) {
        if (strcmp(v->obj_val[i].key, key) == 0)
            return &v->obj_val[i].value;
    }
    return NULL;
}

bool json_has(const Json *v, const char *key) { return json_find(v, key) != NULL; }

void json_array_push(Json *arr, Json item) {
    if (!arr || arr->type != JSON_ARRAY) {
        json_free(&item);
        return;
    }
    if (arr->arr_len + 1 > arr->arr_cap) {
        size_t cap = arr->arr_cap ? arr->arr_cap * 2 : 4;
        Json *next = realloc(arr->arr_val, cap * sizeof(Json));
        if (!next) {
            json_free(&item);
            return;
        }
        arr->arr_val = next;
        arr->arr_cap = cap;
    }
    arr->arr_val[arr->arr_len++] = item;
}

void json_object_set(Json *obj, const char *key, Json value) {
    if (!obj || obj->type != JSON_OBJECT || !key) {
        json_free(&value);
        return;
    }
    for (size_t i = 0; i < obj->obj_len; i++) {
        if (strcmp(obj->obj_val[i].key, key) == 0) {
            json_free(&obj->obj_val[i].value);
            obj->obj_val[i].value = value;
            return;
        }
    }
    if (obj->obj_len + 1 > obj->obj_cap) {
        size_t cap = obj->obj_cap ? obj->obj_cap * 2 : 4;
        JsonPair *next = realloc(obj->obj_val, cap * sizeof(JsonPair));
        if (!next) {
            json_free(&value);
            return;
        }
        obj->obj_val = next;
        obj->obj_cap = cap;
    }
    char *key_copy = cstr_dup(key);
    if (!key_copy) {
        json_free(&value);
        return;
    }
    obj->obj_val[obj->obj_len].key = key_copy;
    obj->obj_val[obj->obj_len].value = value;
    obj->obj_len++;
}

size_t json_size(const Json *v) {
    if (!v) return 0;
    if (v->type == JSON_ARRAY) return v->arr_len;
    if (v->type == JSON_OBJECT) return v->obj_len;
    return 0;
}

const Json *json_at(const Json *v, size_t i) {
    if (!v || v->type != JSON_ARRAY || i >= v->arr_len) return NULL;
    return &v->arr_val[i];
}

typedef struct {
    const char *text;
    size_t pos;
    JsonParseError *err;
} Parser;

static void parse_fail(Parser *p, const char *msg) {
    if (p && p->err && !p->err->message)
        p->err->message = msg;
}

static void skip_ws(Parser *p) {
    while (p->text[p->pos] == ' ' || p->text[p->pos] == '\t' ||
           p->text[p->pos] == '\n' || p->text[p->pos] == '\r')
        p->pos++;
}

static char *parse_raw_string(Parser *p);

static Json parse_value(Parser *p);

static Json parse_true(Parser *p) {
    if (strncmp(p->text + p->pos, "true", 4) == 0) {
        p->pos += 4;
        return json_bool(true);
    }
    parse_fail(p, "Expected 'true'");
    return json_null();
}

static Json parse_false(Parser *p) {
    if (strncmp(p->text + p->pos, "false", 5) == 0) {
        p->pos += 5;
        return json_bool(false);
    }
    parse_fail(p, "Expected 'false'");
    return json_null();
}

static Json parse_null_lit(Parser *p) {
    if (strncmp(p->text + p->pos, "null", 4) == 0) {
        p->pos += 4;
        return json_null();
    }
    parse_fail(p, "Expected 'null'");
    return json_null();
}

static Json parse_number(Parser *p) {
    bool neg = false;
    if (p->text[p->pos] == '-') {
        neg = true;
        p->pos++;
        if (!isdigit((unsigned char)p->text[p->pos])) {
            parse_fail(p, "Expected digit after '-'");
            return json_null();
        }
    }
    uint64_t val = 0;
    if (p->text[p->pos] == '0') {
        p->pos++;
        if (isdigit((unsigned char)p->text[p->pos])) {
            parse_fail(p, "Leading zeroes are not permitted in numbers");
            return json_null();
        }
    } else {
        while (isdigit((unsigned char)p->text[p->pos])) {
            uint64_t digit = (uint64_t)(p->text[p->pos] - '0');
            uint64_t limit = (uint64_t)INT64_MAX + (neg ? 1u : 0u);
            if (val > (limit - digit) / 10) {
                parse_fail(p, "Integer outside JSON int64 range");
                return json_null();
            }
            val = val * 10 + digit;
            p->pos++;
        }
    }
    if (p->text[p->pos] == '.' || p->text[p->pos] == 'e' || p->text[p->pos] == 'E') {
        parse_fail(p, "Floating-point numbers not supported");
        return json_null();
    }
    int64_t result = neg
        ? (val == (uint64_t)INT64_MAX + 1u ? INT64_MIN : -(int64_t)val)
        : (int64_t)val;
    return json_int(result);
}

static bool parse_hex4(Parser *p, uint32_t *cp) {
    *cp = 0;
    for (int i = 0; i < 4; i++) {
        unsigned char h = (unsigned char)p->text[p->pos];
        if (!h) { parse_fail(p, "Incomplete Unicode escape"); return false; }
        p->pos++;
        *cp <<= 4;
        if (h >= '0' && h <= '9') *cp |= (uint32_t)(h - '0');
        else if (h >= 'a' && h <= 'f') *cp |= (uint32_t)(h - 'a' + 10);
        else if (h >= 'A' && h <= 'F') *cp |= (uint32_t)(h - 'A' + 10);
        else { parse_fail(p, "Invalid Unicode hex digit"); return false; }
    }
    return true;
}

static char *parse_raw_string(Parser *p) {
    p->pos++;
    size_t cap = 32, len = 0;
    char *s = malloc(cap);
    if (!s) {
        parse_fail(p, "Out of memory");
        return NULL;
    }
    while (p->text[p->pos]) {
        char c = p->text[p->pos++];
        if (c == '"') {
            s[len] = '\0';
            return s;
        }
        if (c == '\\') {
            char esc = p->text[p->pos++];
            if (!esc) {
                parse_fail(p, "Unfinished escape sequence in string");
                free(s);
                return NULL;
            }
            if (len + 5 > cap) {
                cap *= 2;
                char *n = realloc(s, cap);
                if (!n) { free(s); parse_fail(p, "Out of memory"); return NULL; }
                s = n;
            }
            switch (esc) {
            case '"': s[len++] = '"'; break;
            case '\\': s[len++] = '\\'; break;
            case '/': s[len++] = '/'; break;
            case 'b': s[len++] = '\b'; break;
            case 'f': s[len++] = '\f'; break;
            case 'n': s[len++] = '\n'; break;
            case 'r': s[len++] = '\r'; break;
            case 't': s[len++] = '\t'; break;
            case 'u': {
                uint32_t cp;
                if (!parse_hex4(p, &cp)) { free(s); return NULL; }
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (p->text[p->pos] != '\\' || p->text[p->pos + 1] != 'u') {
                        parse_fail(p, "Missing low Unicode surrogate"); free(s); return NULL;
                    }
                    p->pos += 2;
                    uint32_t low;
                    if (!parse_hex4(p, &low)) { free(s); return NULL; }
                    if (low < 0xDC00 || low > 0xDFFF) {
                        parse_fail(p, "Invalid low Unicode surrogate"); free(s); return NULL;
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    parse_fail(p, "Unpaired low Unicode surrogate"); free(s); return NULL;
                }
                if (cp == 0) {
                    parse_fail(p, "NUL is unsupported in WAM C strings"); free(s); return NULL;
                }
                if (cp <= 0x7F) s[len++] = (char)cp;
                else if (cp <= 0x7FF) {
                    s[len++] = (char)(0xC0 | (cp >> 6));
                    s[len++] = (char)(0x80 | (cp & 0x3F));
                } else if (cp <= 0xFFFF) {
                    s[len++] = (char)(0xE0 | (cp >> 12));
                    s[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    s[len++] = (char)(0x80 | (cp & 0x3F));
                } else {
                    s[len++] = (char)(0xF0 | (cp >> 18));
                    s[len++] = (char)(0x80 | ((cp >> 12) & 0x3F));
                    s[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    s[len++] = (char)(0x80 | (cp & 0x3F));
                }
                break;
            }
            default:
                parse_fail(p, "Invalid escape character");
                free(s);
                return NULL;
            }
            continue;
        }
        if ((unsigned char)c < 0x20) {
            parse_fail(p, "Unescaped control character in string");
            free(s);
            return NULL;
        }
        if (len + 1 >= cap) {
            cap *= 2;
            char *n = realloc(s, cap);
            if (!n) { free(s); parse_fail(p, "Out of memory"); return NULL; }
            s = n;
        }
        s[len++] = c;
    }
    parse_fail(p, "Unterminated string");
    free(s);
    return NULL;
}

static Json parse_array(Parser *p) {
    p->pos++;
    Json arr = json_array();
    skip_ws(p);
    if (p->text[p->pos] == ']') { p->pos++; return arr; }
    for (;;) {
        Json elem = parse_value(p);
        if (p->err && p->err->message) { json_free(&elem); json_free(&arr); return json_null(); }
        json_array_push(&arr, elem);
        skip_ws(p);
        if (!p->text[p->pos]) { parse_fail(p, "Unterminated array"); json_free(&arr); return json_null(); }
        if (p->text[p->pos] == ']') { p->pos++; return arr; }
        if (p->text[p->pos] == ',') {
            p->pos++;
            skip_ws(p);
            if (p->text[p->pos] == ']') { parse_fail(p, "Trailing comma in array"); json_free(&arr); return json_null(); }
        } else {
            parse_fail(p, "Expected ',' or ']' in array");
            json_free(&arr);
            return json_null();
        }
    }
}

static Json parse_object(Parser *p) {
    p->pos++;
    Json obj = json_object();
    skip_ws(p);
    if (p->text[p->pos] == '}') { p->pos++; return obj; }
    for (;;) {
        skip_ws(p);
        if (p->text[p->pos] != '"') {
            parse_fail(p, "Expected string key in object");
            json_free(&obj);
            return json_null();
        }
        char *key = parse_raw_string(p);
        if (!key) { json_free(&obj); return json_null(); }
        skip_ws(p);
        if (p->text[p->pos] != ':') {
            parse_fail(p, "Expected ':' after object key");
            free(key);
            json_free(&obj);
            return json_null();
        }
        p->pos++;
        Json val = parse_value(p);
        if (p->err && p->err->message) { json_free(&val); free(key); json_free(&obj); return json_null(); }
        json_object_set(&obj, key, val);
        free(key);
        skip_ws(p);
        if (!p->text[p->pos]) { parse_fail(p, "Unterminated object"); json_free(&obj); return json_null(); }
        if (p->text[p->pos] == '}') { p->pos++; return obj; }
        if (p->text[p->pos] == ',') {
            p->pos++;
            skip_ws(p);
            if (p->text[p->pos] == '}') { parse_fail(p, "Trailing comma in object"); json_free(&obj); return json_null(); }
        } else {
            parse_fail(p, "Expected ',' or '}' in object");
            json_free(&obj);
            return json_null();
        }
    }
}

static Json parse_value(Parser *p) {
    skip_ws(p);
    if (!p->text[p->pos]) { parse_fail(p, "Unexpected end of input"); return json_null(); }
    char c = p->text[p->pos];
    if (c == '{') return parse_object(p);
    if (c == '[') return parse_array(p);
    if (c == '"') return json_string_owned(parse_raw_string(p));
    if (c == 't') return parse_true(p);
    if (c == 'f') return parse_false(p);
    if (c == 'n') return parse_null_lit(p);
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(p);
    parse_fail(p, "Unexpected character in JSON");
    return json_null();
}

Json json_parse(const char *text, JsonParseError *err) {
    JsonParseError local = {0};
    if (!err) err = &local;
    err->message = NULL;
    Parser p = { text ? text : "", 0, err };
    skip_ws(&p);
    if (!p.text[p.pos]) { err->message = "Empty input"; return json_null(); }
    Json val = parse_value(&p);
    if (err->message) { json_free(&val); return json_null(); }
    skip_ws(&p);
    if (p.text[p.pos]) { err->message = "Trailing junk after JSON value"; json_free(&val); return json_null(); }
    return val;
}

static void dump_string(const char *s, char **out, size_t *len, size_t *cap) {
    #define ENSURE(n) do { \
        while (*len + (n) + 1 > *cap) { \
            *cap = *cap ? *cap * 2 : 64; \
            char *nbuf = realloc(*out, *cap); \
            if (!nbuf) return; \
            *out = nbuf; \
        } \
    } while (0)
    ENSURE(1);
    (*out)[(*len)++] = '"';
    for (const char *p = s ? s : ""; *p; p++) {
        char c = *p;
        const char *esc = NULL;
        char ubuf[8];
        switch (c) {
        case '"': esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\b': esc = "\\b"; break;
        case '\f': esc = "\\f"; break;
        case '\n': esc = "\\n"; break;
        case '\r': esc = "\\r"; break;
        case '\t': esc = "\\t"; break;
        default:
            if ((unsigned char)c < 0x20) {
                snprintf(ubuf, sizeof ubuf, "\\u%04x", (unsigned char)c);
                esc = ubuf;
            }
            break;
        }
        if (esc) {
            ENSURE(strlen(esc));
            memcpy(*out + *len, esc, strlen(esc));
            *len += strlen(esc);
        } else {
            ENSURE(1);
            (*out)[(*len)++] = c;
        }
    }
    ENSURE(1);
    (*out)[(*len)++] = '"';
    #undef ENSURE
}

static void dump_value(const Json *v, char **out, size_t *len, size_t *cap);

static void dump_value(const Json *v, char **out, size_t *len, size_t *cap) {
    if (!v) return;
    #define APPEND_STR(s) do { \
        const char *ss = (s); \
        size_t sl = strlen(ss); \
        while (*len + sl + 1 > *cap) { *cap = *cap ? *cap * 2 : 64; char *nb = realloc(*out, *cap); if (!nb) return; *out = nb; } \
        memcpy(*out + *len, ss, sl); *len += sl; \
    } while (0)
    switch (v->type) {
    case JSON_NULL: APPEND_STR("null"); break;
    case JSON_BOOL: APPEND_STR(v->bool_val ? "true" : "false"); break;
    case JSON_INT: {
        char buf[32];
        snprintf(buf, sizeof buf, "%lld", (long long)v->int_val);
        APPEND_STR(buf);
        break;
    }
    case JSON_STRING:
        dump_string(v->str_val, out, len, cap);
        break;
    case JSON_ARRAY: {
        APPEND_STR("[");
        for (size_t i = 0; i < v->arr_len; i++) {
            if (i) APPEND_STR(",");
            dump_value(&v->arr_val[i], out, len, cap);
        }
        APPEND_STR("]");
        break;
    }
    case JSON_OBJECT: {
        APPEND_STR("{");
        for (size_t i = 0; i < v->obj_len; i++) {
            if (i) APPEND_STR(",");
            dump_string(v->obj_val[i].key, out, len, cap);
            APPEND_STR(":");
            dump_value(&v->obj_val[i].value, out, len, cap);
        }
        APPEND_STR("}");
        break;
    }
    }
    #undef APPEND_STR
}

char *json_dump(const Json *v) {
    size_t len = 0, cap = 0;
    char *out = NULL;
    dump_value(v, &out, &len, &cap);
    if (!out) return cstr_dup("");
    if (len + 1 > cap) {
        char *n = realloc(out, len + 1);
        if (n) out = n;
    }
    out[len] = '\0';
    return out;
}
