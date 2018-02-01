EMpy - ElectroMagnetic Python
*****************************

.. image:: https://travis-ci.org/lbolla/EMpy.svg?branch=master
    :target: https://travis-ci.org/lbolla/EMpy

.. image:: https://api.codacy.com/project/badge/Grade/25215dbf146d47818023159ee64fc563
    :target: https://www.codacy.com/app/lbolla/EMpy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=lbolla/EMpy&amp;utm_campaign=Badge_Grade

.. image:: https://badge.fury.io/py/ElectromagneticPython.svg
    :target: https://badge.fury.io/py/ElectromagneticPython

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

    $> pip install -r requirements_dev.txt
    $> python setup.py develop
    
Run tests with:

.. code-block:: bash

    $> python setup.py test

Release process
===============

1. Edit CHANGES
2. `bumpversion major|minor|patch`
3. `git push && git push --tags`
4. Push to PyPi: `python setup.py release`

Citation
========

If you find EMpy useful in your job, please consider adding a citation.

As reference:

.. code-block::

   Bolla, L. (2017). EMpy [Computer software]. https://github.com/lbolla/EMpy/

As text:

.. code-block::

   We used EMpy (version x.y.z) to complete our work.
