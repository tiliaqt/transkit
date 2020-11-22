# TransKit

## Setup

All dependencies are handled using [Poetry](https://python-poetry.org/), and are specified in the project's `pyproject.toml` file.
All the dependencies you need to work with these notebooks will be installed in a virtualenv specifically for this project using:

```
> poetry install
```

## Usage

To start Jupyter, run:

```
> poetry run jupyter lab
```

`TransKit.ipynb` is a notebook using sample injections data demonstrating basic usage.

I keep my own hormonal data in a different repository, checked out in a parallel directory to this one.
My directory structure looks like this:

```
hormones/
    my/
        Estradiol.ipynb -- personal notebook
        transition.py   -- my injections data
    transkit/
        TransKit.ipynb
        lab.sh
        publish.sh
        <etc>
    transkit-pages/
        <html files>
```

To start Jupyter so that it can access both directories, you can run the `./lab.sh` script.

To cut an html release of the notebooks for publishing, run the `./publish.sh` script.
It outputs the html files to the `../trankit-pages` directory, and expects that to exist.

## Developing

TransKit uses [Black](https://github.com/psf/black) for code formatting and [Flake8](https://flake8.pycqa.org/en/latest/) for linting. They're set up to run in a pre-commit hook.

After installing dependencies with poetry, run:

```
> poetry run pre-commit install
```

Then, all the required checks will run before you can commit.

If the Black check fails, run:

```
> poetry run black .
```

To automatically reformat Python files as needed.
