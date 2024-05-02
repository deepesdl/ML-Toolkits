from setuptools import setup, find_packages

setup(
    name='ml4xcube',
    version='0.0.4',
    package_dir={'': 'mltools'},
    packages=find_packages(where='mltools'),
    package_data={
        'ml4xcube': ['gapfilling/helper/*'],  # Include all files within the 'helper' directory
    },
    install_requires=[
        'bokeh >=2.4.3',
        'dask >=2023.2.0',
        'jinja2 ==3.1.3',
        'mypy_extensions ==1.0.0',
        'numpy >=1.24',
        'pandas >=2.2',
        'scikit-learn >1.3.1',
        'xarray >2023.8.0',
        'zarr >2.11',
        'rechunker >=0.5.1',
        'sentinelhub',
        'xcube'
    ],
)
