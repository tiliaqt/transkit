# TransKit

## Setup

All dependencies are handled using [Poetry](https://python-poetry.org/), and are specified in the project's `pyproject.toml` file.
All the dependencies you need to work with these notebooks will be installed in a virtualenv specifically for this project using.

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
