/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* Needs pub_fn@PUBLIC (a non-numeric node) and plain_fn (unversioned). Both
 * are real loader obligations and must survive ingestion. */
int pub_fn(void);
int plain_fn(void);
int main(void) { return pub_fn() + plain_fn() == 7 ? 0 : 1; }
