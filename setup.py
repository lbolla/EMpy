"""EMpy: Electromagnetic Python.

EMpy is a suite of numerical algorithms used in electromagnetism.
"""

__author__ = 'Lorenzo Bolla'

from setuptools import setup, find_packages

DOCSTRING = __doc__.split('\n')

setup(
    name='EMpy',
    version='0.1.4',
    author='Lorenzo Bolla',
    author_email='lbolla@gmail.com',
    description=DOCSTRING[0],
    long_description='\n'.join(DOCSTRING[2:]),
    url='http://lbolla.github.io/EMpy/',
    download_url='https://github.com/lbolla/EMpy',
    license='BSD',
    platforms=['Windows', 'Linux', 'Mac OS-X'],
    packages=find_packages(),
    package_data={'EMpy': ['tests/*.py', 'doc/*.txt', '*.txt']},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
    provides=['EMpy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
