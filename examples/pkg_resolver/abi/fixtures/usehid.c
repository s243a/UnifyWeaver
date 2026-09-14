/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* Unversioned reference to hid_fn from libhid.so.1. */
int hid_fn(void);
int main(void) { return hid_fn() == 5 ? 0 : 1; }
