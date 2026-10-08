# A platform wheel (it carries native binaries), retagged py3-none-manylinux_2_28_x86_64 by build_wheel.sh:
# the binaries do not touch Python, so one wheel serves every Python version.
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
