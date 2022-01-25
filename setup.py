from setuptools import setup, find_packages
try:
    __version__ = open('EMpy/version.py').read().split("'")[1]
except ImportError:
    __version__ = None

__author__ = 'Lorenzo Bolla'

with open('README.rst', 'r') as readme:
    long_description = readme.read()

setup(
    name='ElectroMagneticPython',
    version=__version__,
    author='Lorenzo Bolla',
    author_email='code@lbolla.info',
    description='EMpy - ElectroMagnetic Python',
    long_description=long_description,
    url='http://lbolla.github.io/EMpy/',
    download_url='https://github.com/lbolla/EMpy',
    license='BSD',
    platforms=['Windows', 'Linux', 'Mac OS-X'],
    packages=find_packages(),
    install_requires=[
        'future',
        'numpy>=1.18',
        'scipy>=1.7',
        'matplotlib>=3.1',
    ],
    provides=['EMpy'],
    test_suite='tests',
    tests_require=[
        'nose<2.0dev',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
