#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NOTEBOOKS="notebooks/TransKit.ipynb notebooks/Cycles.ipynb notebooks/Injectables.ipynb"

cd $DIR && poetry run jupyter nbconvert ${NOTEBOOKS} --to html --template lab --output-dir ../transkit-pages
