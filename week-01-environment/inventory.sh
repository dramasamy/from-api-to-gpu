#!/usr/bin/env bash
# Week 1 — DGX Spark machine inventory.
# Runs read-only inventory commands on the Spark over SSH and prints a report.
# Prereq: an SSH alias `spark` that reaches your DGX Spark (see README.md).
set -euo pipefail

SPARK_HOST="${SPARK_HOST:-spark}"

run() {
  echo "=== $1 ==="
  # shellcheck disable=SC2029
  ssh "$SPARK_HOST" "$2" 2>&1 || true
  echo
}

echo "# DGX Spark inventory ($(date))"
echo "# Host alias: $SPARK_HOST"
echo

run "uname"        'uname -a'
run "os-release"   'cat /etc/os-release'
run "lscpu"        'lscpu'
run "memory"       'free -h'
run "storage"      'lsblk'
run "gpu"          'nvidia-smi'
run "cuda-nvcc"    'nvcc --version || /usr/local/cuda/bin/nvcc --version'
run "cuda-dirs"    'ls -d /usr/local/cuda*'
run "python"       'python3 --version'
run "docker"       'docker version'
