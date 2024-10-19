How to contribute
=================

We welcome contributions!

You can help by:

* Suggesting new features
* Reporting/fixing bugs
* Adding features to the codebase
* Expanding the testing suit
* Improving the documentation


How to Report Issues
---------------------

When reporting issues, please include as many of the the following details as possible:

* Which version of Kadlu you are using
* The source code that generated the problem (if applicable)
* Which platform you are using (Operating system and version)
* A minimum example that reproduces the issue
* What result you got
* What you were expecting

We use GitLab as a repository, so the best way to submit issues is through the `issues system <https://gitlab.meridian.cs.dal.ca/public_projects/kadlu/issues>`_.


Merge Requests
--------------

We welcome merge requests and additional features to the repository. 
The preferred style guide for new contributions is `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_. 
When creating a request, please include a single feature per merge, on a branch forked from master.
For small updates, a description of the changes in the merge request is sufficient. 
For larger contributions please open a discussion thread in the issue tracker before merging.
Please include tests for new features in the ``kadlu/tests`` directory - if a test generates or downloads new data, include cleanup code here as well. 
Finally, please run the tests before merging and ensure tests complete locally before submitting a merge request.

Thank you for your contributions!


Running the tests
-----------------

Kadlu includes its own test suite. They are included in the /kadlu/tests/  directory.

To run all tests, run: ::

    pytest --doctest-modules kadlu

You can also specify a module: ::

    pytest kadlu/tests/geospatial/test_interpolation.py


File testing with the python debugger (PDB)
------------------------------------------------------
    
.. autofunction:: kadlu.test_files()

