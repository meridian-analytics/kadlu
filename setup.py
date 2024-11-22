import os
from pathlib import Path
from setuptools import setup, find_packages
#from kadlu.__init__ import __version__

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='kadlu',
    version="2.4.3",  #__version__
    description="Python package for ocean acoustics modelling",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/meridian-analytics/kadlu',
    author='Oliver Kirsebom, Matthew Smith',
    author_email='oliver.kirsebom@gmail.com',
    license='GNU General Public License v3.0',
    packages=find_packages(exclude=('tests', )),
    install_requires=[
        'Pillow',
        'cdsapi',
        'eccodes',
        'gsw',
        'imageio',
        'matplotlib',
        'mpl_scatter_density',
        'netcdf4',
        'numpy<2.0.0',
        'pygrib',  # DEPENDS ON eccodes binaries
        'pyproj',
        #'pyqt5',
        'requests',
        'scipy',
        'tqdm',
        'xarray',
        'copernicusmarine',  #requires numpy<2.0.0
        'getgfs',
    ],
    #setup_requires=[ 'pytest-runner', ],
    #tests_require=['pytest', 'pytest-parallel'],
    tests_require=['pytest'],
    include_package_data=True,
    exclude_package_data={'': ['tests']},
    python_requires='>=3.6',
)
