#!/bin/bash

poetry run jupyter nbconvert Estradiol.ipynb Cycles.ipynb Injectables.ipynb --to html --template lab --output-dir ../hormone-pages
