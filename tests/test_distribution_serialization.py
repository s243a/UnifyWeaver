#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Tests for finite distribution binary encodings."""

import unittest

from scripts.distribution_serialization import (
    decode_packed_sparse_histogram,
    decode_quantized_cdf_table,
    encode_packed_sparse_histogram,
    encode_quantized_cdf_table,
    encode_selected_distribution,
    max_cdf_error,
)


class DistributionSerializationTests(unittest.TestCase):
    def test_packed_sparse_histogram_round_trips_probabilities(self):
        payload = encode_packed_sparse_histogram([0.25, 0.0, 0.75], origin=4, total_mass=8)
        decoded, meta = decode_packed_sparse_histogram(payload)

        self.assertEqual(decoded, [0.25, 0.0, 0.75])
        self.assertEqual(meta["origin"], 4)
        self.assertEqual(meta["bin_count"], 3)
        self.assertEqual(meta["total_mass"], 8)
        self.assertEqual(meta["payload_bytes"], len(payload))

    def test_quantized_cdf_table_round_trips_with_small_cdf_error(self):
        exact = [0.2, 0.3, 0.5]
        payload = encode_quantized_cdf_table(exact, origin=2, total_mass=10, bits=16)
        decoded, meta = decode_quantized_cdf_table(payload)

        self.assertEqual(meta["representation"], "quantized_cdf_table")
        self.assertEqual(meta["origin"], 2)
        self.assertEqual(meta["bin_count"], 3)
        self.assertAlmostEqual(sum(decoded), 1.0)
        self.assertLessEqual(max_cdf_error(exact, decoded), meta["quantization_step"])

    def test_selected_prefix_representation_uses_quantized_cdf(self):
        payload, meta = encode_selected_distribution([0.2, 0.3, 0.5], "quantized_cdf_table", cdf_bits=8)

        self.assertEqual(meta["representation"], "quantized_cdf_table")
        self.assertEqual(meta["requested_representation"], "quantized_cdf_table")
        self.assertEqual(meta["payload_bytes"], len(payload))
        self.assertLessEqual(meta["decoded_max_cdf_error"], meta["quantization_step"])

    def test_selected_functional_representation_uses_sparse_histogram(self):
        payload, meta = encode_selected_distribution([0.2, 0.0, 0.8], "packed_sparse_histogram")

        self.assertEqual(meta["representation"], "packed_sparse_histogram")
        self.assertEqual(meta["payload_bytes"], len(payload))
        self.assertEqual(meta["decoded_max_cdf_error"], 0.0)

    def test_unknown_representation_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_selected_distribution([1.0], "parametric:binomial_fit")


if __name__ == "__main__":
    unittest.main()
