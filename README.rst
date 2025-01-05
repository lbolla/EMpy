EMpy - ElectroMagnetic Python
*****************************

.. image:: https://github.com/lbolla/EMpy/actions/workflows/python-app.yml/badge.svg
    :target: https://github.com/lbolla/EMpy/actions/workflows/python-app.yml

.. image:: https://badge.fury.io/py/ElectroMagneticPython.svg
    :target: https://badge.fury.io/py/ElectroMagneticPython

`EMpy - ElectroMagnetic Python` is a suite of algorithms widely known
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

First, download the source code from https://github.com/lbolla/EMpy.

Create a virtualenv with, e.g:

.. code-block:: bash

  $> make venv

Then, from inside a `virtualenv`, install dev environment with:

.. code-block:: bash

  $> make develop

Run tests with:

.. code-block:: bash

Upgrade dependencies with:

.. code-block:: bash

  $> make requirements-upgrade
  $> make requirements-sync


Release process
===============

1. Edit CHANGES
2. `make release PART=major|minor|patch`

Citation
========

If you find EMpy useful in your job, please consider adding a citation.

As reference:

.. code-block::

   Bolla, L. (2017). EMpy [Computer software]. https://github.com/lbolla/EMpy/

As text:

.. code-block::

   We used EMpy (version x.y.z) to complete our work.
