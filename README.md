# TransKit

TransKit is a pharmacokinetic simulation library specifically tailored for transgender hormone therapy. It provides a set of low level tools that transgender people can use to understand, manage, and control their hormonal transitions. It is one component of [Tilia](https://github.com/tiliaqt), and is designed as an open source library that can be used by other projects supporting the transgender community. In its current state it should be considered experimental, but already contains a number of tools that can be useful (and have been useful to the author).

Currently the easiest way to access this library is through a Jupyter Notebook, documented below.

## Setup

All dependencies are handled using [Poetry](https://python-poetry.org/), and are specified in the project's `pyproject.toml` file.
All the dependencies you need to work with these notebooks will be installed in a virtualenv specifically for this project using:

```
> poetry install -D
```

If you don't want to follow the steps manually, you can also run the `./tools/install.sh` script.

## Usage

To start Jupyter, run:

```
> poetry run jupyter lab
```

`notebooks/TransKit.ipynb` is a notebook using sample injections data demonstrating basic usage.

I keep my own hormonal data in a different repository, checked out in a parallel directory to this one.
My directory structure looks like this:

```
hormones/
    my/
        Estradiol.ipynb -- personal notebook
        transition.py   -- my injections data
    transkit/
        notebooks/
            TransKit.ipynb
            ...
        tools/
            lab.sh
            publish.sh
            ...
        ...
    transkit-pages/
        <html files>
```

To start Jupyter so that it can access both directories, you can run the `./tools/lab.sh` script.

To cut an html release of the notebooks for publishing, run the `./tools/publish.sh` script.
It outputs the html files to the `../trankit-pages` directory, and expects that to exist.

## Developing

TransKit uses [Pytest](https://docs.pytest.org/en/latest/) for running tests.
You can run all the tests using:

```
> poetry run pytest
```

And you can generate a pretty code coverage report with:

```
> poetry run pytest --cov-report html --cov=transkit
```

We use [Black](https://github.com/psf/black) for code formatting and [Flake8](https://flake8.pycqa.org/en/latest/) for linting.
They're required and are set up to run in a pre-commit hook.

After installing dependencies with poetry, run:

```
> poetry run pre-commit install
```

Then, all the required checks will run before you can commit. You can also manually run them with:

```
> poetry run pre-commit run --all-files
```

We also use [jupytext](https://github.com/mwouts/jupytext) to allow Jupyter notebooks to be represented as plain-text markdown files.
The `notebooks/*.md` files represent the input text of each notebook.
The `notebooks/*.ipynb` notebook files are paired with them and represent the full notebook, including output.
You can open either in Jupyter Lab and the changes will be synced between them.
We commit both files to benefit from the usable diffs of markdown files, and from the accessible viewing of the ipynb files.
The pre-commit hook ensures that they are in sync.
