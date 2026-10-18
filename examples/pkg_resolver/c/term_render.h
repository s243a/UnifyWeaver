/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#pragma once

#include "wam_runtime.h"

/* Debug-only canonical renderer for selftests (not part of shipped API). */
char *term_render(WamState *state, WamValue v);
