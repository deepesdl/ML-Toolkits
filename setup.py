from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='mltools',
    version='0.1',
    packages=find_packages(),
    description='ML package for data cubes',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Julia Peters',
    author_email='julia.peters@informatik.uni-leipzig.de',
    url='https://github.com/deepesdl/ML-Toolkits/tree/develop/mltools',
    install_requires=required_packages,
    python_requires='>=3.8',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        # 'Development Status :: 3 - Alpha',
        # 'Intended Audience :: Developers',
        # 'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='machine learning, tools, data cube utilities',  # Optional
)
