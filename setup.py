"""Setup imageruler."""
import codecs
import os.path
import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string.')


setuptools.setup(
    name='imageruler',
    version=get_version('imageruler/__init__.py'),
    license='MIT',
    author='Wenchao Ma, Ian A. D. Williamson, Martin F. Schubert, Ardavan Oskooi, and Steven G. Johnson',
    author_email='mawc@mit.edu, stevenj@mit.edu',
    install_requires=[
        'numpy',
        'opencv-python',
        'pytest',
        'pytest-xdist',
    ],
    url='https://github.com/NanoComp/imageruler',
    packages=setuptools.find_packages(),
    python_requires='>=3',
    entry_points={"console_scripts": [
        "imageruler = imageruler.cli:main",
    ]})