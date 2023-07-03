import os
from setuptools import setup

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name='colab-diffusion',
    version='0.1.0',
    install_requires=required,
    description='An implementation of Diffusion model in Google Colab',
    author='Jeremy de Gail'
)
