#!/usr/bin/env bash
set -e

# Example NEGF runner
# Usage: ./run_negf.sh transport.fdf
if [ $# -lt 1 ]; then
    echo "Usage: $0 TRANSPORT_FDF"
    exit 1
fi

INPUT=$1
echo "Running NEGF for $INPUT ..."
# Example TranSIESTA command:
# transiesta < "$INPUT" > "${INPUT%.fdf}.TS.out"
