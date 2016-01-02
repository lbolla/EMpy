EMpy - ElectroMagnetic Python
*****************************

.. image:: https://travis-ci.org/lbolla/EMpy.svg?branch=master
    :target: https://travis-ci.org/lbolla/EMpy

`EMpy - Electromagnetic Python` is a suite of algorithms widely known
and used in electromagnetic problems and optics: the transfer matrix
algorithm, the rigorous coupled wave analysis algorithm and more.

Run the examples in `examples/*` to have an idea how EMpy works.

Visit http://lbolla.github.io/EMpy/ for more information.

Installation
============

.. code-block:: bash

  $> pip install ElectromagneticPython

Optionally, install `bvp`:

.. code-block:: bash

  $> pip install scikits.bvp1lg

Development
===========

First, download the source code from https://github.com/lbolla/EMpy. Then, from inside a `virtualenv`, install with:

.. code-block:: bash

    $> pip install -r requirements.txt
    $> python setup.py install
    
Run tests with:

.. code-block:: bash

    $> python setup.py test

Release process
===============

1. Edit CHANGES
2. Edit `version` in `setup.py`
3. `git tag`
4. Push to PyPi: `python setup.py release`
