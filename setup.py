import setuptools
from Cython.Build import cythonize
from setuptools import Extension


extensions = [
    Extension("kdforest/*", ["kdforest/*.pyx"]),
]

setuptools.setup(
    name="kdtrees",
    version="0.0.1",
    author="Maixent Chenebaux",
    author_email="max.chbx@gmail.com",
    description="Forest of kd-trees with minimal index size",
    long_description_content_type="text/markdown",
    url="https://github.com/kerighan/kdforest",
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
