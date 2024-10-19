import os
from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='kadlu',
    version=os.environ.get('KADLUVERSION', '2.4.1'),
    description="MERIDIAN Python package for ocean ambient noise modelling",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.meridian.cs.dal.ca/public_projects/kadlu',
    author='Oliver Kirsebom, Matthew Smith',
    author_email='oliver.kirsebom@dal.ca, matthew.smith@dal.ca',
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
        'numpy',
        'pygrib',  # DEPENDS ON eccodes binaries
        'pyproj',
        #'pyqt5',
        'requests',
        'scipy',
        'tqdm',
        'xarray',
    ],
    #setup_requires=[ 'pytest-runner', ],
    #tests_require=['pytest', 'pytest-parallel'],
    tests_require=['pytest'],
    include_package_data=True,
    exclude_package_data={'': ['tests']},
    python_requires='>=3.6',
)
