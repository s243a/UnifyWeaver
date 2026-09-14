/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* Needs alpha_fn@COMMON_1 from libalpha.so.1 AND beta_fn@COMMON_1 from
 * libbeta.so.1: the same version-node NAME in two verneed files. Attribution
 * must go through the per-symbol version INDEX, not the node name. */
int alpha_fn(void);
int beta_fn(void);
int main(void) { return alpha_fn() + beta_fn() == 3 ? 0 : 1; }
