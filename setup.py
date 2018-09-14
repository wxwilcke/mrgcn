#!/usr/bin/env python

from setuptools import find_packages, setup


setup(
    name='mrgcn',
    version='0.1',
    author='Xander Wilcke',
    author_email='w.x.wilcke@vu.nl',
    description='',
    license='GLP3',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        'toml',
        'tensorflow'
    ],
)
