#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Binary encodings for finite path-statistic distributions."""

from __future__ import annotations

import math
import struct


MAGIC = b"UWDS"
VERSION = 1
KIND_PACKED_SPARSE = 1
KIND_QUANTIZED_CDF = 2
HEADER = struct.Struct("<4sBBHiId")
SPARSE_COUNT = struct.Struct("<I")
SPARSE_ENTRY = struct.Struct("<Id")


def normalize(probabilities: list[float]) -> list[float]:
    total = sum(probabilities)
    if total <= 0.0:
        return [1.0] if probabilities else []
    return [value / total for value in probabilities]


def pad_to(left: list[float], right: list[float]) -> tuple[list[float], list[float]]:
    size = max(len(left), len(right))
    return left + [0.0] * (size - len(left)), right + [0.0] * (size - len(right))


def max_cdf_error(left: list[float], right: list[float]) -> float:
    left, right = pad_to(left, right)
    left_total = 0.0
    right_total = 0.0
    worst = 0.0
    for left_value, right_value in zip(left, right):
        left_total += left_value
        right_total += right_value
        worst = max(worst, abs(left_total - right_total))
    return worst


def w1_cdf_error(left: list[float], right: list[float]) -> float:
    left, right = pad_to(left, right)
    left_total = 0.0
    right_total = 0.0
    total = 0.0
    for left_value, right_value in zip(left, right):
        left_total += left_value
        right_total += right_value
        total += abs(left_total - right_total)
    return total


def _header(kind: int, bits: int, origin: int, bin_count: int, total_mass: float) -> bytes:
    return HEADER.pack(MAGIC, VERSION, kind, int(bits), int(origin), int(bin_count), float(total_mass))


def _parse_header(payload: bytes) -> tuple[int, int, int, int, float, int]:
    if len(payload) < HEADER.size:
        raise ValueError("payload too short for distribution header")
    magic, version, kind, bits, origin, bin_count, total_mass = HEADER.unpack_from(payload, 0)
    if magic != MAGIC:
        raise ValueError("bad distribution payload magic")
    if version != VERSION:
        raise ValueError("unsupported distribution payload version")
    return kind, bits, origin, bin_count, total_mass, HEADER.size


def encode_packed_sparse_histogram(probabilities: list[float], origin: int = 0, total_mass: float = 1.0) -> bytes:
    probabilities = normalize(probabilities)
    entries = [(index, value) for index, value in enumerate(probabilities) if value != 0.0]
    out = bytearray(_header(KIND_PACKED_SPARSE, 0, origin, len(probabilities), total_mass))
    out.extend(SPARSE_COUNT.pack(len(entries)))
    for index, value in entries:
        out.extend(SPARSE_ENTRY.pack(index, value))
    return bytes(out)


def decode_packed_sparse_histogram(payload: bytes) -> tuple[list[float], dict[str, float | int | str]]:
    kind, bits, origin, bin_count, total_mass, offset = _parse_header(payload)
    if kind != KIND_PACKED_SPARSE:
        raise ValueError("payload is not a packed sparse histogram")
    if len(payload) < offset + SPARSE_COUNT.size:
        raise ValueError("payload too short for sparse entry count")
    (entry_count,) = SPARSE_COUNT.unpack_from(payload, offset)
    offset += SPARSE_COUNT.size
    probabilities = [0.0 for _ in range(bin_count)]
    for _ in range(entry_count):
        if len(payload) < offset + SPARSE_ENTRY.size:
            raise ValueError("payload too short for sparse entry")
        index, value = SPARSE_ENTRY.unpack_from(payload, offset)
        offset += SPARSE_ENTRY.size
        if index >= bin_count:
            raise ValueError("sparse entry index outside support")
        probabilities[index] = value
    return probabilities, {
        "representation": "packed_sparse_histogram",
        "origin": origin,
        "bits": bits,
        "bin_count": bin_count,
        "total_mass": total_mass,
        "payload_bytes": len(payload),
    }


def _integer_format(bits: int) -> tuple[str, int]:
    if bits <= 8:
        return "<B", 8
    if bits <= 16:
        return "<H", 16
    if bits <= 32:
        return "<I", 32
    raise ValueError("quantized CDF bits must be <= 32")


def encode_quantized_cdf_table(probabilities: list[float], origin: int = 0, total_mass: float = 1.0, bits: int = 16) -> bytes:
    probabilities = normalize(probabilities)
    fmt, stored_bits = _integer_format(bits)
    packer = struct.Struct(fmt)
    levels = (1 << stored_bits) - 1
    cumulative = 0.0
    out = bytearray(_header(KIND_QUANTIZED_CDF, stored_bits, origin, len(probabilities), total_mass))
    previous = 0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        quantized = levels if index == len(probabilities) - 1 else round(cumulative * levels)
        quantized = min(levels, max(previous, int(quantized)))
        out.extend(packer.pack(quantized))
        previous = quantized
    return bytes(out)


def decode_quantized_cdf_table(payload: bytes) -> tuple[list[float], dict[str, float | int | str]]:
    kind, bits, origin, bin_count, total_mass, offset = _parse_header(payload)
    if kind != KIND_QUANTIZED_CDF:
        raise ValueError("payload is not a quantized CDF table")
    fmt, stored_bits = _integer_format(bits)
    packer = struct.Struct(fmt)
    levels = (1 << stored_bits) - 1
    probabilities = []
    previous = 0
    for _ in range(bin_count):
        if len(payload) < offset + packer.size:
            raise ValueError("payload too short for quantized CDF point")
        (quantized,) = packer.unpack_from(payload, offset)
        offset += packer.size
        if quantized < previous:
            raise ValueError("quantized CDF is not monotone")
        probabilities.append((quantized - previous) / levels)
        previous = quantized
    return probabilities, {
        "representation": "quantized_cdf_table",
        "origin": origin,
        "bits": stored_bits,
        "bin_count": bin_count,
        "total_mass": total_mass,
        "payload_bytes": len(payload),
        "quantization_step": 1.0 / levels,
    }


def decode_distribution_payload(payload: bytes) -> tuple[list[float], dict[str, float | int | str]]:
    kind, _bits, _origin, _bin_count, _total_mass, _offset = _parse_header(payload)
    if kind == KIND_PACKED_SPARSE:
        return decode_packed_sparse_histogram(payload)
    if kind == KIND_QUANTIZED_CDF:
        return decode_quantized_cdf_table(payload)
    raise ValueError("unsupported distribution payload kind: {}".format(kind))


def encode_selected_distribution(
    probabilities: list[float],
    representation: str,
    origin: int = 0,
    total_mass: float = 1.0,
    cdf_bits: int = 16,
) -> tuple[bytes, dict[str, float | int | str]]:
    if representation == "quantized_cdf_table":
        payload = encode_quantized_cdf_table(probabilities, origin, total_mass, cdf_bits)
        decoded, meta = decode_quantized_cdf_table(payload)
    elif representation in {"packed_sparse_histogram", "exact_histogram", "tail_pruned_histogram"}:
        payload = encode_packed_sparse_histogram(probabilities, origin, total_mass)
        decoded, meta = decode_packed_sparse_histogram(payload)
    else:
        raise ValueError("unsupported serializable representation: {}".format(representation))
    meta = dict(meta)
    meta["requested_representation"] = representation
    meta["decoded_max_cdf_error"] = max_cdf_error(normalize(probabilities), decoded)
    meta["decoded_w1_cdf_error"] = w1_cdf_error(normalize(probabilities), decoded)
    return payload, meta
