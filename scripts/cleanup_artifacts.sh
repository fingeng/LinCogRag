#!/usr/bin/env bash
set -euo pipefail

# 将根目录的运行产物归档到 experiments/ 下（不直接删除）

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p experiments/logs experiments/pids experiments/results

shopt -s nullglob

echo "[cleanup] moving *.log, nohup.out -> experiments/logs/"
for f in *.log nohup.out; do
  [ -e "$f" ] || continue
  mv -f "$f" "experiments/logs/"
done

echo "[cleanup] moving *.pid -> experiments/pids/"
for f in *.pid; do
  [ -e "$f" ] || continue
  mv -f "$f" "experiments/pids/"
done

echo "[cleanup] moving results_*.json -> experiments/results/"
for f in results_*.json results_*.jsonl results_*.csv; do
  [ -e "$f" ] || continue
  mv -f "$f" "experiments/results/"
done

echo "[cleanup] done."


