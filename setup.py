#!/usr/bin/env python

from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='mrgcn',
    version='0.1',
    author='Xander Wilcke',
    author_email='w.x.wilcke@vu.nl',
    url='https://gitlab.com/wxwilcke/mrgcn',
    description='Multimodal Relational Graph Convolutional Network (MRGCN)',
    license='GLP3',
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        'torch',
        'numpy',
        'rdflib',
        'scipy',
        'pillow',
        'nltk',
        'toml',
    ],
    packages=['mrgcn'],
)
