#!/usr/bin/env python

"""EMpy: Electromagnetic Python.

EMpy is a suite of numerical algorithms used in electromagnetism.

"""

__author__ = 'Lorenzo Bolla'

DOCLINES = __doc__.split('\n')

# ZIP file: python setup.py sdist
# EXE installer: python setup.py bdist_wininst
# see http://docs.python.org/dist/dist.html

import os
import sys

from setuptools import setup, find_packages
from EMpy.version import version
from EMpy.dependencies import dependencies

setup(
		name = 'EMpy',
		version = version,
		maintainer = 'Lorenzo Bolla',
		maintainer_email = 'lbolla@users.sourceforge.net',
		description = DOCLINES[0],
		long_description = '\n'.join(DOCLINES[2:]),
		url = 'http://empy.sourceforge.net',
		download_url = 'http://sourceforge.net/project/showfiles.php?group_id=205871',
		license = 'BSD',
		author = 'Lorenzo Bolla',
		author_email = 'lbolla@users.sourceforge.net',
		platforms = ['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
		packages = find_packages(),
		package_data = {'EMpy' : ['tests/*.py', 'doc/*.txt', '*.txt']},
		# install_requires = ['numpy>=1.1.0', 'scipy>=0.7.0', 'bvp>=0.2.2'],
		requires = dependencies, # requires doesn't seem to work!!! emulate it with dependencies.py
		provides = ['EMpy'],
		classifiers = [
			'Development Status :: 2 - Pre-Alpha',
			'Environment :: Console',
			'Intended Audience :: Science/Research',
			'License :: OSI Approved :: BSD License',
			'Natural Language :: English',
			'Programming Language :: Python',
			'Operating System :: OS Independent',
			'Topic :: Scientific/Engineering :: Physics',
			]
		)
