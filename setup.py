from setuptools import setup, find_packages

setup(
    name='surface_enlargement_factor',
    version='0.1.0',
    description='A package for surface enlargement factor calculation with optional bending correction.',
    author='Markus Sekulla',
    author_email='markus.sekulla@mb.tu-chemnitz.de',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'matplotlib>=3.0.0',
        'cmasher>=0.5.0',
        'tabulate>=0.8.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)