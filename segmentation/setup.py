from setuptools import setup

setup(
    name="gvcore",
    version="1.1",
    description="The core library for Geo's vision framework.",
    author="Geoffrey Chen",
    author_email="geoffreychen777@gmail.com",
    license="MIT",
    packages=["gvcore"],
    zip_safe=False,
    install_requires=["torch", "torchvision", "numpy", "tabulate",],
    entry_points={"console_scripts": ["gvrun=gvcore.utils.launcher:gvrun", "gvsubmit=gvcore.utils.launcher:gvsubmit", "gvsweep=gvcore.utils.sweeper:gvsweep"]},
)
