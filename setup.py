__author__ = 'Lorenzo Bolla'

from setuptools import setup, find_packages

with open('README.rst', 'r') as readme:
    long_description = readme.read()

setup(
    name='ElectroMagneticPython',
    version='1.1',
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
        'distribute>=0.6.28',
        'future',
        'numpy',
        'scipy',
        'matplotlib',
    ],
    provides=['EMpy'],
    test_suite='tests',
    tests_require=[
        'nose==1.3.7',
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
