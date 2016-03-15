# -*- coding: utf-8 -*-
import setuptools

setuptools.setup(
    name='embedor',
    version='0.0.0',
    description='A package for creating word (and other) embeddings',
    author='Shayne Miel',
    author_email='miel.shayne@gmail.com',
    url='https://github.com/FragLegs/embedor',
    packages=setuptools.find_packages(exclude=['tests*']),

    install_requires=[
        'tensorflow>=0.7.1'
    ])
