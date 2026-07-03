#!/bin/bash

# Create a run directory and populate it with template files.
# Usage:
#   ./init_run_dir.sh <run_dir>
#   ./init_run_dir.sh <run_dir> --force

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [--force]"
    exit 1
fi

run_dir=$1
force=${2:-}

script_dir=$(cd "$(dirname "$0")" && pwd)
tpl_dir="${script_dir}/templates"

if [ ! -d "${tpl_dir}" ]; then
    echo "Templates directory not found: ${tpl_dir}"
    exit 1
fi

mkdir -p "${run_dir}"

if [ "${force}" = "--force" ]; then
    cp -f "${tpl_dir}"/* "${run_dir}"/
else
    cp -n "${tpl_dir}"/* "${run_dir}"/
fi

echo "Run directory initialized: ${run_dir}"
echo "Template source: ${tpl_dir}"
if [ "${force}" != "--force" ]; then
    echo "Existing files were kept. Use --force to overwrite."
fi
