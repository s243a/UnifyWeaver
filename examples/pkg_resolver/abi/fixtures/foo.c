/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* libfoo fixture: built twice under the SAME soname (libfoo.so.1) with two
 * version scripts. v1 exports foo@LIB_1; v2 keeps the LIB_1 node (foo_legacy)
 * but moves foo to LIB_2 only -- so a binary linked against v1 needs foo@LIB_1
 * and the loader rejects v2: "undefined symbol: foo, version LIB_1". */
int foo(void) { return 1; }
int foo_legacy(void) { return 0; }
