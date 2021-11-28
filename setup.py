import setuptools

setuptools.setup(
    name='util',
    install_requires=[
        'click>=7.0', 'numpy', 'ase', 'pandas', 'pyyaml', 'tqdm', 'lxml',
        'tabulate', 'seaborn', 'quippy-ase', 'pytest'
    ],
    entry_points="""
    [console_scripts]
    egg=util.cli.cli:cli
    """
)
