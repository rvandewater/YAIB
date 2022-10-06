#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README_old.md") as readme_file:
    readme = readme_file.read()

setup_requirements = ["pytest-runner"]

setup(
    author="Robin van de Water",
    author_email="robin.vandewater@hpi.de",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    description="This project aim is to build a benchmark for ICU related tasks.",
    entry_points={"console_scripts": ["icu-benchmarks = icu_benchmarks.run:main"]},
    install_requires=[],  # dependencies managed via conda for the moment
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="icu_benchmarks",
    name="icu_benchmarks",
    packages=find_packages(include=["icu_benchmarks"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=[],
    url="https://github.com/rvandewater/YAIB",
    version="0.1.0",
    zip_safe=False,
)
