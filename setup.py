import os
from setuptools import setup, find_packages

setup(
    name="svc_helper",
    version="0.0.0",
    author="effusiveperiscope",
    description = (""),
    license="MIT",
    packages=find_packages(exclude=['tests']),
    package_data={
        'svc_helper': ['svc/rvc/configs/**/*.json']
    }
)