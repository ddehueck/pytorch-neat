from setuptools import setup
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pytorch-neat',
    version='0.0.1',
    description='A PyTorch implementation of the NEAT (NeuroEvolution of Augmenting Topologies) method which was originally created by Kenneth O. Stanley as a principled approach to evolving neural networks.',
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',
    url='https://github.com/ddehueck/pytorch-neat',
    author='Devin de Hueck',
    author_email='d.dehueck@gmail.com',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],

    keywords='neat pytorch neuroevolution',
    packages=['neat']
)
