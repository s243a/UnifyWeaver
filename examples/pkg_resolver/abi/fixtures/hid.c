/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* libhid fixture (Sol P1b): hid_fn is exported ONLY at a NON-DEFAULT version
 * node (`.symver` with a single `@` = hidden, readelf shows hid_fn@HID_1, not
 * @@). Built twice under soname libhid.so.1:
 *   hid_idx3.map: HID_0 first (verdef index 2), HID_1 second (index 3) -- the
 *                 loader refuses an unversioned reference: "undefined symbol: hid_fn"
 *   hid_idx2.map: HID_1 is the only node (verdef index 2) -- the loader binds a
 *                 legacy unversioned reference to the oldest node even though it
 *                 is hidden (glibc dl-lookup: index < 3 is accepted before the
 *                 hidden test). Ingest records index 2 as "default" for that reason. */
int other_fn(void) { return 9; }
int hid_fn_impl(void) { return 5; }
__asm__(".symver hid_fn_impl, hid_fn@HID_1");
