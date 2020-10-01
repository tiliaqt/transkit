# Estradiol

## Setup

All dependencies are handled using [Poetry](https://python-poetry.org/), and are specified in the project's `pyproject.toml` file. All the dependencies you need to work with these notebooks will be installed in a virtualenv specifically for this project using:

```
> poetry install
```

Then, to start Jupyter, run:

```
> poetry run jupyter lab
```

To cut an html release of the notebooks for publishing, run:

```
> ./publish.sh
```

It outputs the html files to the `../hormone-pages` directory, and expects that to exist.
