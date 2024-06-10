# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from setuptools import find_packages, setup

VERSION = '1.1.0'

test_deps = []

setup(
    name='pandemic_simulator',
    version=VERSION,
    description='Sony AI PandemicSimulator',
    package_dir={'': 'python'},
    packages=find_packages('python', exclude=[]),
    package_data={'': ['VERSION'],
                  'pandemic_simulator': ['py.typed']},
    install_requires=[
        'dataclasses',
        'gym>=0.15.4',
        'istype>=0.2.0',
        'matplotlib',
        'networkx',  # for graph analysis
        'numpy==1.19.5',    # 1.20 numpy requires mypy fixes #TODO
        'scipy',
        'probabilistic-automata>=0.4.0',  # for probabilistic DFA
        'pyrsistent>=0.15.5',  # for frozen classes
        'pytest>=5.2.2',
        'PyYAML>=5.3.1',
        'typing-inspect==0.5.0',  # to handle issubclass changes in python 3.7,

        'orderedset>=2.0.3',
        'cachetools>=4.1.0',
        'h5py>=2.10.0',
        'tqdm>=4.48.0',
        'GPyOpt',
        'pandas',
        'structlog'
    ],
    tests_require=test_deps,
    extras_require={
        'test': test_deps
    },
)
