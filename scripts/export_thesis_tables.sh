#!/usr/bin/env bash
set -euo pipefail

echo "Curated thesis table sources:"
find results/tables -maxdepth 1 -type f -name '*.csv' | sort

echo
echo "Selected thesis figures:"
find results/figures -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.svg' \) | sort

echo
echo "These files are already export-ready CSV/figure artifacts."
echo "Use docs/results_inventory.md for the thesis-section mapping."
