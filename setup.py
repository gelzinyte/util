import setuptools

setuptools.setup(
    name='util',
    install_requires=[
        'click>=7.0', 'numpy', 'ase', 'pandas'
    ],
    entry_points="""
    [console_scripts]
    util=util.cli:cli
    """
)