"""EMpy: Electromagnetic Python.

EMpy is a suite of numerical algorithms used in electromagnetism.

"""

__author__ = 'Lorenzo Bolla'

DOCLINES = __doc__.split('\n')

# ZIP file: python setup.py sdist
# EXE installer: python setup.py bdist_wininst
# see http://docs.python.org/dist/dist.html

from setuptools import setup, find_packages
from EMpy.version import version
from EMpy.dependencies import dependencies

setup(
    name='EMpy',
    version=version,
    maintainer='Lorenzo Bolla',
    maintainer_email='lbolla@gmail.com',
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    url='http://lbolla.github.io/EMpy/',
    download_url='https://github.com/lbolla/EMpy',
    license='BSD',
    author='Lorenzo Bolla',
    author_email='lbolla@gmail.com',
    platforms=['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    packages=find_packages(),
    package_data={'EMpy': ['tests/*.py', 'doc/*.txt', '*.txt']},
    install_requires=dependencies,
    requires=dependencies,
    provides=['EMpy'],
    classifiers=[
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
