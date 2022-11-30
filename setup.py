from setuptools import setup, find_packages

setup(
    name='stable_diffusion_inference',
    version='0.0.1',
    description='',
    package_dir = {"": "src"},
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
)
