import setuptools
from Cython.Build import cythonize
from setuptools import Extension


extensions = [
    Extension("linearity/*", ["linearity/*.pyx"]),
]

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="kdtrees",
    version="0.0.2",
    author="Maixent Chenebaux",
    author_email="max.chbx@gmail.com",
    description="Linear algebra with minimal overhead",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kerighan/linearity",
    packages=setuptools.find_packages(),
    include_package_data=True,
    ext_modules=cythonize(extensions),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5"
)
