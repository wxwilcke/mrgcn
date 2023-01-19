#!/usr/bin/env python

from setuptools import find_namespace_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='mrgcn',
    version='2.1',
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
        'toml',
        'deep_geometry',
        'huggingface_hub',
        'transformers',
        'sentencepiece',
        'sacremoses', 
        'importlib_metadata',
        'tqdm',
        'packaging'
    ],
    packages=find_namespace_packages(include=['mrgcn.*']),
)
