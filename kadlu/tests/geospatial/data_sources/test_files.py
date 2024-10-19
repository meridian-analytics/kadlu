import os
import subprocess

import kadlu


class test_files():
    """ start a debugging session for reading .nc and .tif file formats.

    dynamically generates tests for a given set of input data files.
    pytest tests will be written to a script using files in
    ``kadlu_data/testfiles/`` as inputs.
    this is then run in a pytest subprocess, the test script will be cleaned
    upon exit.

    pytest config can be passed using the ``DEBUGOPTS`` environment variable:
    by default the script will drop into an interactive pdb debugging session.
    to set a breakpoint, add ``breakpoint()`` directly to the source code.
    otherwise, pdb will break on exceptions.

    alternatively, run tests in parallel using the pytest-parallel package

    Usage:

    .. code-block:: python

        from importlib import reload
        import os, kadlu

        # place some files in the testfiles directory
        kadlu.ifremer().fetch_ifremer_netcdf_hs2013()

        # reload kadlu and run tests
        reload(kadlu); kadlu.test_files()

        # optionally pass args to pytest with DEBUGOPTS
        os.environ.setdefault('DEBUGOPTS', '--pdb')  # default behaviour
        reload(kadlu); kadlu.test_files()

    From the terminal:

    .. code-block:: bash

        # set pytest options
        export DEBUGOPTS='--pdb --tb=short -s'
        python3 test_files.py

        # run tests in parallel and log the results
        pip install pytest-parallel
        export DEBUGOPTS='--workers=auto --tb=line --durations=0'
        python3 -B test_files.py | tee testresults.log

    see the ``DEBUGOPTS`` env var and 'man pytest' for further usage information
    """

    def __init__(self):
        with self as tests:
            tests.run()

    def __enter__(self):
        IMPORTS = 'import os, kadlu'
        PATH, _, FILES = list(os.walk(kadlu.storage_cfg() + 'testfiles'))[0]
        TESTS = lambda F, PATH=PATH: F'''

    def test_loadfile_{F.replace('.','').replace('-','').replace(' ','')}():
        kadlu.load_file(os.path.join('{PATH}','{F}')), 'error: {F}' '''
        with open('scriptoutput.py', 'w') as OUTPUT:
            OUTPUT.write(IMPORTS + '\nif 21392>>4:' +
                         ''.join(map(TESTS, sorted(FILES))))
        return self

    def run(self):
        subprocess.run(
            f'python -B -m pytest {os.environ.get("DEBUGOPTS", "--pdb --tb=native -s")} scriptoutput.py'
            .split())

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove('scriptoutput.py')


if __name__ == '__main__':
    test_files()
