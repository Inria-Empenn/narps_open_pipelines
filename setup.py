#!/usr/bin/python
# coding: utf-8

"""
Allow the project to be installed as a local package, using :
    python -m pip install .
    or
    python -m pip install -e .
The -e option stands for editable, which allows to change the
source code of the package without having to reinstall it. Using this option
will add the current directory (.) to the system path.
"""

from setuptools import setup, find_packages

requires = [
    'importlib_resources>=5.10.2,<5.11',
    'nipype',
    'pandas'
]
extras_require = {
    'tests': [
        'pylint',
        'pytest',
        'pytest-cov',
        'pytest-helpers-namespace'
        ]
}

setup(
    name = 'narps_open',
    version = '0.1.0',
    description = 'The NARPS open pipelines project :\
    a codebase reproducing the 70 pipelines of the NARPS study (Botvinik-Nezer et al., 2020).',
    long_description = 'file: README.md',
    long_description_content_type = 'text/markdown',
    author = '',
    author_email = '',
    url = 'https://github.com/Inria-Empenn/narps_open_pipelines',

    include_package_data = True,
    python_requires = '>=3.8,<3.12',
    install_requires = requires,
    extras_require = extras_require,
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8'
    ],
    project_urls = {
        'Bug Tracker': 'https://github.com/Inria-Empenn/narps_open_pipelines/issues',
        'Source': 'https://github.com/Inria-Empenn/narps_open_pipelines',
    },
    packages = find_packages(exclude=('tests', 'examples')),
    data_files = [
        ('narps_open/pipelines', ['narps_open/pipelines/analysis_pipelines_derived_configuration.tsv']),
        ('narps_open/pipelines', ['narps_open/pipelines/analysis_pipelines_full_configuration.tsv'])
    ]
)
