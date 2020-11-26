#!/bin/bash

poetry install -D
poetry run pre-commit install
poetry run jupyter contrib nbextension install --sys-prefix --symlink
poetry run jupyter nbextensions_configurator enable --sys-prefix
# Go to http://localhost:8888/nbextensions and enable "Equation Auto Numbering"
