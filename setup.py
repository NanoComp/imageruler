"""Setup imageruler."""
import setuptools

setuptools.setup(
    name='imageruler',
    version='1.0',
    license='MIT',
    author='',
    author_email='',
    install_requires=[
        'numpy',
        'opencv-python',
        'pytest',
    ],
    url='https://github.com/NanoComp/imageruler',
    packages=setuptools.find_packages(),
    python_requires='>=3',
)