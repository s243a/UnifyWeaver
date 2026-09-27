/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* Linked against libfoo v1: requires foo@LIB_1 from libfoo.so.1. */
int foo(void);
int main(void) { return foo() == 1 ? 0 : 1; }
