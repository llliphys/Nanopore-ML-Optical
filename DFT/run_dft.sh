#!/usr/bin/env bash
set -e

# Example DFT runner (extend as needed)
# Usage: ./run_dft.sh input.fdf
if [ $# -lt 1 ]; then
    echo "Usage: $0 INPUT_FDF"
    exit 1
fi

INPUT=$1
echo "Running DFT for $INPUT ..."
# Example SIESTA command:
# siesta < "$INPUT" > "${INPUT%.fdf}.out"
