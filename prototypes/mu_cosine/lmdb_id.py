#!/usr/bin/env python3
"""Shared MediaWiki category LMDB id encoding helpers.

MediaWiki page IDs are unsigned 32-bit integers. Existing fixtures currently sit below
2**31, so this remains byte-compatible with earlier signed-int helpers while avoiding
latent failures for larger page-id spaces.
"""

import struct

ID32 = struct.Struct("<I")


def enc_id(value):
    return ID32.pack(int(value))


def dec_id(value):
    return ID32.unpack(bytes(value))[0]


def looks_int(text):
    try:
        int(text)
    except (TypeError, ValueError):
        return False
    return True
