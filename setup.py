# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# Get the long description from the README file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='iRRR',
    version='0.0.1',
    description='Python3 implementation of integrative reduced-rank regression.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Emily Stephen',
    email='emilyps14@gmail.com',
    url='https://https://github.com/emilyps14/iRRR_python',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'PyQt5',
        'matplotlib',
        'scipy',
        'numpy'],
)
