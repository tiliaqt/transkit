#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NOTEBOOKS="TransKit.ipynb Cycles.ipynb Injectables.ipynb"

cd $DIR && poetry run jupyter nbconvert ${NOTEBOOKS} --to html --template lab --output-dir ../transkit-pages
