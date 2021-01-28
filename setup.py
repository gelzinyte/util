import setuptools

setuptools.setup(
    name='util',
    entry_points="""
    [console_scripts]
    util=util.cli:cli
    """
)