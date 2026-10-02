#!/bin/sh
set -eu
APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$APP_DIR"
if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python --no-deps -e .
if ! command -v exiftool >/dev/null 2>&1 && [ ! -f .tools/exiftool-13.59/exiftool ]; then
    mkdir -p .tools
    curl -fL https://codeload.github.com/exiftool/exiftool/tar.gz/refs/tags/13.59 -o .tools/exiftool-13.59.tar.gz
    printf '%s\n' '87d3317882fdae9cb4dcfe57a96a378d0132ffc02c731315bf128b19ddcf7aac  .tools/exiftool-13.59.tar.gz' | sha256sum -c -
    tar -xzf .tools/exiftool-13.59.tar.gz -C .tools
fi
./photo-denoise download all
mkdir -p "$HOME/.local/bin"
if [ ! -e "$HOME/.local/bin/photo-denoise" ] && [ ! -L "$HOME/.local/bin/photo-denoise" ]; then
    ln -s "$APP_DIR/photo-denoise" "$HOME/.local/bin/photo-denoise"
fi
./photo-denoise doctor
