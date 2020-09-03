from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()


setup(
    name="flydf",
    version="0.1",
    packages=["flydf",],
    author="Florian Aymanns",
    author_email="florian.ayamnns@epfl.ch",
    description="Utility function for managing pandas.DataFrames that hold data of multiple flies and trials.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/flydf",
    install_requires=["pytest", "pandas"],
)
