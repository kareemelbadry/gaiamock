from setuptools import setup, find_packages

setup(
    name='gaiamock',
    version='1.0.0',
    description='Gaia astrometry simulation package',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'healpy',
        'joblib',
    ],
    python_requires='>=3.6',
)
