#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from pathlib import Path
from setuptools import setup, find_packages
import yaml


root_path = Path(__file__).resolve().parent

def parse_environment_yml():
    """Parse the environment.yml file and extract the package names."""
    with open(root_path / "environment.yml") as f:
        environment = yaml.safe_load(f)
    dependencies = []
    for entry in environment["dependencies"]:
        if isinstance(entry, dict):
            for key, value in entry.items():
                if key == "pip":
                    dependencies += value
        else:
            dependencies.append(entry)
    # dependencies = ["==".join(dep.split("=")) for dep in dependencies]
    
    sanitized_dependencies = []
    for dependency in dependencies:
        dependency = "==".join(dependency.split("="))
        if "http://" in dependency or "https://" in dependency:
            package_name = dependency.split("/")[-1].split(".")[0]
            dependency = package_name + "@" + dependency
        sanitized_dependencies.append(dependency)
    print("extraced dependencies:", sanitized_dependencies)
    return sanitized_dependencies

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = ["pytest-runner"]

setup(
    author="Robin van de Water",
    author_email="robin.vandewater@hpi.de",
    classifiers=[
        "Development Status :: Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    description="Yet Another ICU Benchmark is a holistic framework for the automation of clinical prediction models "
    "on ICU data. Users can create custom datasets, cohorts, prediction tasks, endpoints, and models.",
    entry_points={"console_scripts": ["icu-benchmarks = icu_benchmarks.run:main"]},
    # install_requires=["recipys@git+https://github.com/rvandewater/recipys.git"],
    install_requires=parse_environment_yml(),
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="benchmark mimic-iii eicu hirid clinical machine learning",
    name="Yet Another ICU Benchmark",
    packages=find_packages(include=["icu_benchmarks"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=[],
    url="https://github.com/rvandewater/YAIB",
    version="0.1.0",
    zip_safe=False,
)
