from glob import glob
from os.path import splitext
from setuptools import setup
from setuptools import find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name='nam',
    version='1.0.0',
    description='Neural Additive Models with Interaction',
    author='Komatsu-T',
    url='https://github.com/Komatsu-T/Neural-Additive-Models',
    packages=find_packages(),
    install_requires=_requires_from_file('requirements.txt'),
    classifiers=[
            "Programming Language :: Python :: 3.6",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
)
