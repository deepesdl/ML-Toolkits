from setuptools import setup, find_packages

setup(
    name='ml4xcube',
    version='1.0.1',
    package_dir={'': 'mltools'},
    packages=find_packages(where='mltools'),
    package_data={
        'ml4xcube': ['gapfilling/helper/*'],  # Include all files within the 'helper' directory
    },
    install_requires=[
        'dask >=2023.2.0',
        'numpy >=1.24',
        'pandas >=2.2',
        'scikit-learn >1.3.1',
        'xarray >2023.8.0',
        'zarr >2.11',
        'rechunker >=0.5.1',
    ],
)
