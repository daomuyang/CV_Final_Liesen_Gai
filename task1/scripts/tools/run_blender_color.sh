#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${BLENDER_PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/pacman/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "bpy python not found: $PYTHON" >&2
  echo "Install bpy in conda env 'pacman', or set BLENDER_PYTHON to that interpreter." >&2
  exit 1
fi

exec "$PYTHON" "$TOOLS_DIR/blender_vertex_color.py" "$@"
