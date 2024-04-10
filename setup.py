from setuptools import setup, find_packages

setup(
    name='mltools',
    version='0.1.0',
    package_dir={'': 'mltools'},
    packages=find_packages(where='mltools'),
    install_requires=[
        'bokeh >=2.4.3, <3',
        'dask >2024.2',
        'dask-core >2024.2',
        'jinja2 ==3.1.3',
        'mypy_extensions ==1.0.0',
        'numpy >=1.24',
        'pandas >=2.2',
        'scikit-learn >1.3.1',
        'xarray >=2024.0',
        'zarr >2.11',
        'rechunker >=0.5.1',
        'sentinelhub'
    ],
)
