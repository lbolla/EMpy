# EMpy

[![Build Status](https://travis-ci.org/lbolla/EMpy.svg?branch=master)](https://travis-ci.org/lbolla/EMpy)

`EMpy (Electromagnetic Python)` is a suite of algorithms widely knonw and used in electromagnetic problems and optics:
the transfer matrix algorithm, the rigorous coupled wave analysis algorithm and more.

Run the examples in `examples/*` to have an idea how EMpy works.
Visit http://lbolla.github.io/EMpy/ for more information.

## Installation

  $> pip install ElectromagneticPython

## Development

- Download the source code from https://github.com/lbolla/EMpy.
- From inside a virtualenv, install with:

    >>> pip install -r requirements.txt
    >>> python setup.py install

  or

    >>> python setup.py develop
    
- Run tests with:

    >>> python setup.py test
