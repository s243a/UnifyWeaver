/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* Unversioned build of libhid.so.1 (hid_fn@Base): usehid is linked against
 * this one, so its reference to hid_fn is UNVERSIONED. */
int other_fn(void) { return 9; }
int hid_fn(void) { return 5; }
