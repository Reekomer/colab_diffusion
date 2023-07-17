from glob import glob
from os.path import abspath, basename, dirname, join, splitext

import setuptools

requirements_path = join(dirname(abspath(__file__)), "requirements.txt")

with open(requirements_path) as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="simplediffusion",
    version="0.0.1",
    install_requires=requirements,
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    author='Jeremy de Gail',
    python_requires=">=3.10",
)
