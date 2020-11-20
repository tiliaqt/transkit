#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "$DIR/.."
ggrep --exclude-dir="backups"\
      --exclude-dir=".ipynb_checkpoints"\
      --exclude-dir="__pycache__"\
      --exclude-dir=".git"\
      --exclude-dir="transkit-pages"\
      --exclude-dir="transkit-fresh"\
      --exclude="*.swp"\
      -n --color=auto -r "$@" .
