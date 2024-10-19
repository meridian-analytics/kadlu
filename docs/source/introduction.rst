Introduction
============

Kadlu is a software package for modeling underwater noise, first released in March 2020.
It was developed for the purpose of modelling noise due to waves 
and rain in shallow coastal waters, but contains tools useful for 
many other ocean acoustics modeling tasks.

Kadlu is written in Python and utilizes a number of powerful software packages 
including NumPy, HDF5, NetCDF-4, SQLite, and GDAL.
It is licensed under the `GNU GPLv3 license <https://www.gnu.org/licenses/>`_ and hence freely available for anyone to use and modify.
The project is hosted on GitLab at 
`https://gitlab.meridian.cs.dal.ca/public_projects/kadlu <https://gitlab.meridian.cs.dal.ca/public_projects/kadlu>`_ . 
Kadlu was developed by the `MERIDIAN <http://meridian.cs.dal.ca/>`_ Data Analytics Team at the 
`Institute for Big Data Analytics <https://bigdata.cs.dal.ca/>`_ at Dalhousie University with the 
support and assistance of David Barclay and Calder Robinson, both from the Department of Oceanography 
at Dalhousie University.

Kadlu provides functionalities that automate the process of fetching and interpolating 
environmental data necessary to model ocean ambient noise levels (bathymetry, water temperature 
and salinity, wave height, wind speed, etc.). It also includes various routines that allow 
accurate estimates of noise source levels and transmission losses in realistic ocean environments.
You can find more information about the technical aspects of how sound propagation is modelled in 
Kadlu in :download:`this note <./_static/kadlu_sound_propagation_note.pdf>`.

The intended users of Kadlu are researchers and students in underwater acoustics working with ambient noise modeling. 
While Kadlu comes with complete documentation and comprehensive step-by-step tutorials, some familiarity with Python and 
especially the NumPy package would be beneficial. A basic understanding of 
the physical principles of underwater sound propagation would also be an advantage.

To get started with Kadlu, follow the :ref:`installation_instructions` instructions and then proceed to the :ref:`tutorials` section.

In Inuit mythology, Kadlu is the name of a goddess that creates thundery weather, for example, by jumping on hollow ice. 
Thus, the name Kadlu was chosen to highlight the software package's main intended application, modeling of noise due to 
environmental forcings.

