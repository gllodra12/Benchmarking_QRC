from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="benchmarking-qrc",
    version="0.0.3",
    description="""Benchmarking QRC measures the ability to store information of 
    different quantum particles""",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    url="https://github.com/gllodra12/Benchmarking_QRC",
    author="Guillem Llodrà",
    author_email="gllodra1225@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=["openfermion~=1.1.0"],
    extras_requires={"dev": ["pytest>=6", "twine>=4"]},
)
