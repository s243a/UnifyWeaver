/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#pragma once

#include "json.h"
#include "term_heap.h"

WamValue tb_ver_term(TermHeap *th, const Json *ver);
WamValue tb_pair_term(TermHeap *th, const char *name, const Json *ver);
WamValue tb_constraint_term(TermHeap *th, const Json *c);
WamValue tb_hold_term(TermHeap *th, const Json *row);
WamValue tb_layer_term(TermHeap *th, const Json *row);
WamValue tb_alias_term(TermHeap *th, const Json *row);
WamValue tb_pkg_term(TermHeap *th, const Json *row);
WamValue tb_dep_term(TermHeap *th, const Json *row);
WamValue tb_provide_term(TermHeap *th, const Json *row);
WamValue tb_conf_term(TermHeap *th, const Json *row);
WamValue tb_request_term(TermHeap *th, const Json *req);
WamValue tb_catalog_to_term(TermHeap *th, const Json *catalog);
