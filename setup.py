#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import sys
from pathlib import Path
from setuptools import setup, find_packages


root_path = Path(__file__).resolve().parent

def parse_environment_yml():
    """Parse the environment.yml file and extract the package names."""
    # here we cannot use pyyaml because it is not installed yet
    with open(root_path / "environment.yml") as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines]
    dependencies = []
    inside_dependencies = False
    for entry in lines:
        print("entry:", entry)
        if entry == "dependencies:":
            print("found dependencies")
            inside_dependencies = True
            continue
        if inside_dependencies:
            print("parsing dependencies:", entry)
            if not entry.startswith("-"):
                break
            dependency_name = entry.strip().split(" ")[-1]
            if dependency_name != "pip:":
                dependencies.append(dependency_name)
            print("now dependencies:", dependencies)
    
    sanitized_dependencies = []
    for dependency in dependencies:
        # conda package ignite is named pytorch-ignite on pypi
        if "ignite" in dependency:
            dependency = "pytorch-" + dependency
        if dependency.startswith("pytorch="):
            dependency = dependency.replace("pytorch", "torch")
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
