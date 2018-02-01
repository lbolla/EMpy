from setuptools import setup, find_packages
try:
    from EMpy import __version__
except ImportError:
    __version__ = None

__author__ = 'Lorenzo Bolla'

with open('README.rst', 'r') as readme:
    long_description = readme.read()

setup(
    name='ElectroMagneticPython',
    version=__version__,
    author='Lorenzo Bolla',
    author_email='lbolla@gmail.com',
    description='EMpy - ElectroMagnetic Python',
    long_description=long_description,
    url='http://lbolla.github.io/EMpy/',
    download_url='https://github.com/lbolla/EMpy',
    license='BSD',
    platforms=['Windows', 'Linux', 'Mac OS-X'],
    packages=find_packages(),
    install_requires=[
        'future<1.0dev',
        'numpy<2.0dev',
        'scipy<1.0dev',
        'matplotlib<2.0dev',
    ],
    provides=['EMpy'],
    test_suite='tests',
    tests_require=[
        'nose<2.0dev',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
