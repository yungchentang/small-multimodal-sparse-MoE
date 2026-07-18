#!/usr/bin/env bash

: "${CONDITIONAL_BATCH_SIZE:=16}"
if [[ ! "$CONDITIONAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONDITIONAL_BATCH_SIZE must be a positive integer" >&2
  return 2 2>/dev/null || exit 2
fi
export CONDITIONAL_BATCH_SIZE
