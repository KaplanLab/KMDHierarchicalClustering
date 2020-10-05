import setuptools
import re

from distutils.util import convert_path


with open("README.md", "r") as fh:
    long_description = fh.read()

with open(convert_path('KMDHierarchicalClustering/__init__.py')) as ver_file:
    match = next(re.finditer('__version__ = "(.*)"', ver_file.read(), re.MULTILINE))
    version = match.group(1)


setuptools.setup(
    name="KMDHierarchicalClustering",  # Replace with your own username
    version=version,
    author='Aviv Zelig, Noam Kaplan',
    author_email="noam.kaplan@technion.ac.il",
    description="KMD clustering: Robust generic clustering of biological data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires='>=3.6',
)
