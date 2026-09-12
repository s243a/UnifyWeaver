/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#pragma once

#include <stdio.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

static inline void check(int cond, const char *what) {
    if (cond) {
        g_pass++;
    } else {
        g_fail++;
        printf("FAIL: %s\n", what);
    }
}

static inline void check_str(const char *got, const char *want, const char *what) {
    if (got && want && strcmp(got, want) == 0) {
        g_pass++;
    } else {
        g_fail++;
        printf("FAIL: %s (got \"%s\", want \"%s\")\n", what, got ? got : "(null)", want ? want : "(null)");
    }
}

static inline int finish_tests(void) {
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
