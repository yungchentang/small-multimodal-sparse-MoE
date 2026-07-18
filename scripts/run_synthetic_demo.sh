#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-training/config_top2_1week.yaml}" OUT="${OUT:-outputs/top2_1week}" bash run.sh synthetic-demo
