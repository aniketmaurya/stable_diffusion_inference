import os

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

def _load_requirements(path_dir: str, file_name: str = "requirements.txt") -> list:
    reqs = parse_requirements(open(os.path.join(path_dir, file_name)).readlines())
    return list(map(str, reqs))


setup(
    name='stable_diffusion_inference',
    version='0.0.1',
    description='',
    package_dir = {"": "src"},
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    install_requires=_load_requirements(_PATH_ROOT),
)
