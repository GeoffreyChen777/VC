from setuptools import setup

setup(
    name="gvcore",
    version="1.0",
    description="The core library for vision framework.",
    url="",
    author="geo",
    author_email="",
    license="MIT",
    packages=["gvcore"],
    zip_safe=False,
    install_requires=["torch", "torchvision", "numpy", "tabulate",],
)
