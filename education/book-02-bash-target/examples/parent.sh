# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# This file is part of UnifyWeaver.
# Licensed under either MIT or Apache-2.0 at your option.

#!/bin/bash
# parent - fact lookup
declare -A parent_data=(
[abraham:ishmael]=1
    [abraham:isaac]=1
    [sarah:isaac]=1
    [isaac:esau]=1
    [isaac:jacob]=1
    [rebekah:esau]=1
    [rebekah:jacob]=1
    [jacob:reuben]=1
    [jacob:simeon]=1
    [jacob:levi]=1
    [jacob:judah]=1
)
parent() {
  local key="$1:$2"
  [[ -n "$parent_data[$key]}" ]] && echo "$key"
}
parent_stream() {
  for key in "${!parent_data[@]}"; do
    echo "$key"
  done
}
parent_reverse_stream() {
  for key in "${!parent_data[@]}"; do
    IFS=":" read -r a b <<< "$key"
    echo "$b:$a"
  done
}
