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

EMpy is managed with `uv` (https://docs.astral.sh/uv/):

Run tests with:

.. code-block:: bash

  $> make test

Install dev environment with:

.. code-block:: bash

  $> make develop

Upgrade dependencies with `uv`:

.. code-block:: bash

  # add or update a dependency, then refresh lockfile and sync environment
  $> make upgrade

Release process
===============

Package version is handed by `setuptools_scm`/`uv`:

.. code-block:: bash

  # tag the release (use the desired version)
  git tag vX.Y.Z
  git push --tags

GitHub actions will build and upload the package to PyPI.

Citation
========

If you find EMpy useful in your job, please consider adding a citation.

As reference:

.. code-block::

   Bolla, L. (2017). EMpy [Computer software]. https://github.com/lbolla/EMpy/

As text:

.. code-block::

   We used EMpy (version x.y.z) to complete our work.
