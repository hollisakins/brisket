from setuptools import find_packages, setup

setup(
    name='brisket',
    packages=find_packages(include=['brisket','brisket.*']),
    version='0.1.0',
    description='BRISKET SED fitter',
    entry_points = {
        "console_scripts": ['brisket-mod = brisket.brisket:mod',
                            'brisket-fit = brisket.brisket:fit',
                            'brisket-plot = brisket.brisket:plot',
                            'brisket-filters = brisket.brisket:filters']
        },
    author='Hollis Akins',
    license='MIT',
    install_requires=['astropy','numpy','matplotlib','scipy','tqdm'],
)
