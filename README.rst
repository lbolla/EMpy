EMpy - ElectroMagnetic Python
*****************************

.. image:: https://travis-ci.org/lbolla/EMpy.svg?branch=master
    :target: https://travis-ci.org/lbolla/EMpy

`EMpy - Electromagnetic Python` is a suite of algorithms widely knonw
and used in electromagnetic problems and optics: the transfer matrix
algorithm, the rigorous coupled wave analysis algorithm and more.

Run the examples in `examples/*` to have an idea how EMpy works.

Visit http://lbolla.github.io/EMpy/ for more information.

Installation
============

.. code-block:: bash

  $> pip install ElectromagneticPython

Development
===========

First, download the source code from https://github.com/lbolla/EMpy. Then, from inside a `virtualenv`, install with:

.. code-block:: python

    >>> pip install -r requirements.txt
    >>> python setup.py install
    
Run tests with:

.. code-block:: python

    >>> python setup.py test
