###########################################################
pandalone: process data-trees with relocatable-paths
###########################################################
|pypi-ver| |travis-status| |appveyor-status| |cover-status| |docs-status| \
|dependencies| |downloads-count| |github-issues| |python-ver| |proj-license|

.. image:: doc/_static/pandalone_logo.png
   :width: 300 px
   :align: center

**pandalone** is a collection of utilities for working with *hierarchical-data*
using *relocatable-paths*.

:Release:       0.2.4
:Date:          2017-03-21 18:44:54
:Documentation: https://pandalone.readthedocs.org/
:Source:        https://github.com/pandalone/pandalone
:PyPI repo:     https://pypi.python.org/pypi/pandalone
:Keywords:      calculation, data, dependencies, engineering, excel, library,
                numpy, pandas, processing, python, resolution, scientific,
                simulink, tree, utility
:Copyright:     2015 European Commission (`JRC-IET
                <https://ec.europa.eu/jrc/en/institutes/iet>`_)
:License:       `EUPL 1.1+ <https://joinup.ec.europa.eu/software/page/eupl>`_

Currently only 2 portions of the envisioned functionality are ready for use:

- :mod:`pandalone.xleash`: A mini-language for "throwing the rope" around rectangular areas
  of Excel-sheets.
- :mod:`pandalone.mappings`: Hierarchical string-like objects that may be used for
  indexing, facilitating renaming keys and column-names at a later stage.


Our goal is to facilitate the composition of *engineering-models* from
loosely-coupled *components*.
Initially envisioned as an *indirection-framework* around *pandas* coupled
with a *dependency-resolver*, every such model should auto-adapt and process
only values available, and allow *remapping* of the paths accessing them,
to run on renamed/relocated *value-trees* without component-code modifications.

It is an open source library written for *python-3.4* but tested under both
*python-2.7* and *python-3.3+*, for *Windows* and *Linux*.

.. Note::
    The project, as of May-2015, is considered at an alpha-stage,
    without any released version in *pypi* yet.


.. _end-opening:
.. contents:: Table of Contents
  :backlinks: top
.. _begin-intro:

Introduction
============

Overview
--------

At the most fundamental level, an "execution" or a "run" of any data-processing
can be thought like that::

          .--------------.     _____________        .-------------.
         ;  DataTree    ;    |             |      ;   DataTree   ;
        ;--------------; ==> |  <cfunc_1>  | ==> ;--------------;
       ; /some/data   ;      |  <cfunc_2>  |    ; /some/data   ;
      ;  /some/other ;       |     ...     |   ;  /some/other ;
     ;   /foo/bar   ;        |_____________|  ;   /foo/bar   ;
    '--------------'                         '--------------.


- The *data-tree* might come from *json*, *hdf5*, *excel-workbooks*, or
  plain dictionaries and lists.
  Its values are strings and numbers, *numpy-lists*, *pandas* or
  *xray-datasets*, etc.

- The *component-functions* must abide to the following simple signature::

    cfunc_do_something(pandelone, datatree)

  and must not return any value, just read and write into the data-tree.

- Here is a simple component-function:

  .. code-block:: python

    def cfunc_standardize(pandelone, datatree):
        pin, pon = pandelone.paths(),
        df = datatree.get(pin.A)
        df[pon.A.B_std] = df[pin.A.B] / df[pin.A.B].std()

- Notice the use of the *relocatable-paths* marked specifically as input or
  output.

- TODO: continue rough example in tutorial...



Quick-start
-----------

.. Note::
    The program runs on **Python-2.7+** and **Python-3.3+** (preferred) and requires
    **numpy/scipy**, **pandas** and **win32** libraries along with their *native backends* to be installed.
    If you do not have such an environment already installed, please read :doc:`install` section below for
    suitable distributions such as |anaconda|_ or |winpython|_.

Assuming that you have a working python-environment, open a *command-shell*,
(in *Windows* use :program:`cmd.exe` BUT ensure :program:`python.exe` is in its :envvar:`PATH`),
try the following commands:

.. Tip::
    The commands beginning with ``$``, below, imply a *Unix* like operating system with a *POSIX* shell
    (*Linux*, *OS X*). Although the commands are simple and easy to translate in its *Windows* ``cmd.exe`` counterpart,
    it would be worthwile to install `Cygwin <https://www.cygwin.com/>`_ to get the same environment on *Windows*.
    If you choose to do that, include also the following packages in the *Cygwin*'s installation wizard::

        * git, git-completion
        * make, zip, unzip, bzip2, dos2unix
        * openssh, curl, wget

    But do not install/rely on cygwin's outdated python environment.

:Install:
    .. code-block:: bash

        $ pip install pandalone                 ## Use `--pre` if version-string has a build-suffix.

    Or in case you need the very latest from `master` branch :

    .. code-block:: bash

        $ pip install git+https://github.com/pandalone/pandalone.git

    See: :doc:`install`

:Run:
    .. code-block:: bash

        $ pandalone --version



.. _install:

Install
=======
Current version(|version|) runs on **Python-2.7+** and **Python-3.3+** and requires
**numpy/scipy**, **pandas** and **win32** libraries along with their *native backends* to be installed.

It has been tested under *Windows* and *Linux* and *Python-3.3+* is the preferred interpreter,
i.e, the *Excel* interface and desktop-UI runs only with it.

It is distributed on `Wheels <https://pypi.python.org/pypi/wheel>`_.


Python installation
-------------------

.. Warning::
    On *Windows* it is strongly suggested **NOT to install the standard CPython distribution**,
    unless:

    a) you have *administrative priviledges*,
    b) you are an experienced python programmer, so that
    c) you know how to hunt dependencies from *PyPi* repository and/or
       the `Unofficial Windows Binaries for Python Extension Packages <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.

As explained above, this project depends on packages with *native-backends* that require the use
of *C* and *Fortran* compilers to build from sources.
To avoid this hassle, you should choose one of the user-friendly distributions suggested below.

Below is a matrix of the two suggested self-wrapped python distributions for running this program
(we excluded here default *python* included in *linux*). Both distributions:

- are free (as of freedom),
- do not require *admin-rights* for installation in *Windows*, and
- have been tested to run successfully this program (also tested on default *linux* distros).

+-----------------+-------------------------------------------+-------------------------------------------+
| *Distributions* | |winpython|_                              | |anaconda|_                               |
|                 |                                           |                                           |
+=================+===========================================+===========================================+
| *Platform*      | **Windows**                               | **Windows**, **Mac OS**, **Linux**        |
+-----------------+-------------------------------------------+-------------------------------------------+
| *Ease of*       | Fair                                      | - *Anaconda:* Easy                        |
|                 |                                           | - *MiniConda:* Moderate                   |
|                 | (requires fiddling with the               |                                           |
|                 | :envvar:`PATH`                            |                                           |
| *Installation*  |                                           |                                           |
|                 | and the Registry after install)           |                                           |
|                 |                                           |                                           |
+-----------------+-------------------------------------------+-------------------------------------------+
| *Ease of Use*   | Easy                                      | Moderate                                  |
|                 |                                           |                                           |
|                 |                                           | (should use :command:`conda` and/or       |
|                 |                                           | :command:`pip`                            |
|                 |                                           |                                           |
|                 |                                           | depending on whether a package            |
|                 |                                           |                                           |
|                 |                                           | contains native libraries                 |
|                 |                                           |                                           |
+-----------------+-------------------------------------------+-------------------------------------------+
| *# of Packages* | Only what's included                      | Many 3rd-party packages                   |
|                 |                                           |                                           |
|                 | in the downloaded-archive                 | uploaded by users                         |
|                 |                                           |                                           |
+-----------------+-------------------------------------------+-------------------------------------------+
| *Notes*         | After installation, see :ref:`faq` for:   | - Check also the lighter `miniconda       |
|                 |                                           |   <http://conda.pydata.org/               |
|                 | - Registering WinPython installation      |   miniconda.html>`_.                      |
|                 | - Adding your installation in             | - For installing native-dependencies      |
|                 |   :envvar:`PATH`                          |                                           |
|                 |                                           |   with :command:`conda` see files:        |
|                 |                                           |                                           |
|                 |                                           |   - :file:`requirements/miniconda.conda`  |
|                 |                                           |   - :file:`.travis.yaml`                  |
|                 |                                           |                                           |
+-----------------+-------------------------------------------+-------------------------------------------+
|                 | Check also installation instructions from `the  pandas site                           |
|                 | <http://pandas.pydata.org/pandas-docs/stable/install.html>`_.                         |
|                 |                                                                                       |
+-----------------+-------------------------------------------+-------------------------------------------+



Package installation
--------------------

Before installing it, make sure that there are no older versions left over
on the python installation you are using.
To cleanly uninstall it, run this command until you cannot find any project installed:

.. code-block:: bash

    $ pip uninstall pandalone                   ## Use `pip3` if both python-2 & 3 are in PATH.


You can install the project directly from the |pypi|_ the "standard" way,
by typing the :command:`pip` in the console:

  .. code-block:: bash

      $ pip install pandalone

- If you want to install a *pre-release* version (the version-string is not plain numbers, but
  ends with ``alpha``, ``beta.2`` or something else), use additionally option ``--pre``.

.. code-block:: bash

    $ pip install pandalone

- Also you can install the very latest version straight from the sources:

  .. code-block:: bash

      $ pip install git+git://github.com/pandalone/pandalone.git  --pre

- If you want to upgrade an existing installation along with all its dependencies,
  add also option ``--upgrade`` (or option ``-U`` equivalently), but then the build might take some
  considerable time to finish.  Also there is the possibility the upgraded libraries might break
  existing programs(!) so use it with caution, or from within a |virtualenv|_.

- To install it for different Python environments, repeat the procedure using
  the appropriate :program:`python.exe` interpreter for each environment.

- .. Tip::
    To debug installation problems, you can export a non-empty :envvar:`DISTUTILS_DEBUG`
    and *distutils* will print detailed information about what it is doing and/or
    print the whole command line when an external program (like a C compiler) fails.


After installation, it is important that you check which version is visible in your :envvar:`PATH`:

.. code-block:: bash

    $ pndlcmd --version
    0.2.4


To install for different Python versions, repeat the procedure for every required version.



Older versions
--------------
To install an older released version issue the console command:

.. code-block:: bash

    $ pip install pandalone=0.0.1                   ## Use `--pre` if version-string has a build-suffix.

or alternatively straight from the sources:

  .. code-block:: bash

      $ pip install git+https://github.com/pandalone/pandalone.git@v0.0.9-alpha.3.1  --pre

Of course you can substitute `v0.0.9-alpha.3.1` with any slug from "commits", "branches" or "releases"
that you will find on project's `github-repo <https://github.com/pandalone/pandalone>`_).

.. Note::
    If you have another version already installed, you have to use option ``--ignore-installed`` (or option ``-I``).
    For using the specific version, check this (untested)
    `stackoverflow question
    <http://stackoverflow.com/questions/6445167/force-python-to-use-an-older-version-of-module-than-what-i-have-installed-now>`_.

    You can install each version in a separate |virtualenv|_ and shy away from all this.
    Check


Installing sources
-----------------------
If you download the sources you have more options for installation.
There are various methods to get hold of them:

* Download the *source* distribution from |pypi|_.
* Download a `release-snapshot from github <https://github.com/pandalone/pandalone/releases>`_
* Clone the *git-repository* at *github*.

  Assuming you have a working installation of `git <http://git-scm.com/>`_
  you can fetch and install the latest version of the project with the following series of commands:

  .. code-block:: bash

      $ git clone "https://github.com/pandalone/pandalone.git" pandalone.git
      $ cd pandalone.git
      $ python setup.py install                                 ## Use `python3` if both python-2 & 3 installed.


When working with sources, you need to have installed all libraries that the project depends on:

.. code-block:: bash

    $ pip install -r requirements/execution.pip .


The previous command installs a "snapshot" of the project as it is found in the sources.
If you wish to link the project's sources with your python environment, install the project
in `development mode <http://pythonhosted.org/setuptools/setuptools.html#development-mode>`_:

.. code-block:: bash

    $ python setup.py develop


.. Note:: This last command installs any missing dependencies inside the project-folder.



Project files and folders
-------------------------
The files and folders of the project are listed below::

    +--pandalone/       ## (package) Python-code
    +--tests/           ## (package) Test-cases
    +--doc/             ## Documentation folder
    +--setup.py         ## (script) The entry point for `setuptools`, installing, testing, etc
    +--requirements/    ## (txt-files) Various pip and conda dependencies.
    +--README.rst
    +--CHANGES.rst
    +--AUTHORS.rst
    +--CONTRIBUTING.rst
    +--LICENSE.txt



.. _usage:

Usage
=====
Currently 2 portions of this library are ready for use: :mod:`pandalone.xleash` and
:mod:`pandalone.mappings`

.. _cmd-line-usage:

Cmd-line usage
--------------
.. Warning:: Not implemented in yet.

The command-line usage below requires the Python environment to be installed, and provides for
executing an experiment directly from the OS's shell (i.e. :program:`cmd` in windows or :program:`bash` in POSIX),
and in a *single* command.

[TBD]


GUI usage
---------
.. Attention:: Desktop UI requires Python 3!

For a quick-'n-dirty method to explore the structure of the data-tree and run an experiment,
just run:

.. code-block:: bash

    $ pandalone gui



.. _excel-usage:

Excel usage
-----------
.. Attention:: Excel-integration requires Python-3 and *Windows* or *OS X*!

In *Windows* and *OS X* you may utilize the excellent `xlwings <http://xlwings.org/quickstart/>`_ library
to use Excel files for providing input and output to the experiment.

To create the necessary template-files in your current-directory you should enter:

.. code-block:: console

     $ pandalone excel


You could type instead :samp:`pandalone excel {file_path}` to specify a different destination path.

[TBD]



.. _python-usage:

Python usage
------------
Example python :abbr:`REPL (Read-Eval-Print Loop)` example-commands  are given below
that setup and run an *experiment*.

First run :command:`python` or :command:`ipython` and try to import the project to check its version:

.. doctest::

    >>> import pandalone

    >>> pandalone.__version__           ## Check version once more.
    '0.2.4'

    >>> pandalone.__file__              ## To check where it was installed.         # doctest: +SKIP
    /usr/local/lib/site-package/pandalone-...


.. Tip:
    The use :command:`ipython` is preffered over :command:`python` since it offers various user-friendly
    facilities, such as pressing :kbd:`Tab` for completions, or allowing you to suffix commands with `?` or `??`
    to get help and read their source-code.

    Additionally you can <b>copy any python commands starting with ``>>>`` and ``...``</b> and copy paste them directly
    into the ipython interpreter; it will remove these prefixes.
    But in :command:`python` you have to remove it youself.

If everything works, create the :term:`data-tree` to hold the input-data (strings and numbers).
You assemble data-tree by the use of:

* sequences,
* dictionaries,
* :class:`pandas.DataFrame`,
* :class:`pandas.Series`, and
* URI-references to other data-trees.


[TBD]



.. _contribute:

Getting Involved
================
This project is hosted in **github**.
To provide feedback about bugs and errors or questions and requests for enhancements,
use `github's Issue-tracker <https://github.com/pandalone/pandalone/issues>`_.



Sources & Dependencies
----------------------
To get involved with development, you need a POSIX environment to fully build it
(*Linux*, *OSX* or *Cygwin* on *Windows*).

First you need to download the latest sources:

.. code-block:: console

    $ git clone https://github.com/pandalone/pandalone.git pandalone.git
    $ cd pandalone.git


.. Admonition:: Virtualenv
    :class: note

    You may choose to work in a |virtualenv|_,
    to install dependency libraries isolated from system's ones, and/or without *admin-rights*
    (this is recommended for *Linux*/*Mac OS*).

    .. Attention::
        If you decide to reuse stystem-installed packages using  option ``--system-site-packages``
        with ``virtualenv <= 1.11.6``
        (to avoid, for instance, having to reinstall *numpy* and *pandas* that require native-libraries)
        you may be bitten by `bug #461 <https://github.com/pypa/virtualenv/issues/461>`_ which
        prevents you from upgrading any of the pre-installed packages with :command:`pip`.

.. Admonition:: Liclipse IDE
    :class: note

    Within the sources there are two sample files for the comprehensive
    `LiClipse IDE <http://www.liclipse.com/>`_:

    * :file:`eclipse.project`
    * :file:`eclipse.pydevproject`

    Remove the `eclipse` prefix, (but leave the dot(`.`)) and import it as "existing project" from
    Eclipse's `File` menu.

    Another issue is caused due to the fact that LiClipse contains its own implementation of *Git*, *EGit*,
    which badly interacts with unix *symbolic-links*, such as the :file:`docs/docs`, and it detects
    working-directory changes even after a fresh checkout.  To workaround this, Right-click on the above file
    :menuselection:`Properties --> Team --> Advanced --> Assume Unchanged`


Then you can install all project's dependencies in *`development mode* using the :file:`setup.py` script:

.. code-block:: console

    $ python setup.py --help                           ## Get help for this script.
    Common commands: (see '--help-commands' for more)

      setup.py build      will build the package underneath 'build/'
      setup.py install    will install the package

    Global options:
      --verbose (-v)      run verbosely (default)
      --quiet (-q)        run quietly (turns verbosity off)
      --dry-run (-n)      don't actually do anything
    ...

    $ python setup.py develop                           ## Also installs dependencies into project's folder.
    $ python setup.py build                             ## Check that the project indeed builds ok.


You should now run the test-cases to check
that the sources are in good shape:

.. code-block:: console

   $ python setup.py test


.. Note:: The above commands installed the dependencies inside the project folder and
    for the *virtual-environment*.  That is why all build and testing actions have to go through
    :samp:`python setup.py {some_cmd}`.

    If you are dealing with installation problems and/or you want to permantly install dependant packages,
    you have to *deactivate* the virtual-environment and start installing them into your *base*
    python environment:

    .. code-block:: console

       $ deactivate
       $ python setup.py develop

    or even try the more *permanent* installation-mode:

    .. code-block:: console

       $ python setup.py install                # May require admin-rights



Design
------
See `architecture live-document
<https://docs.google.com/document/d/1P73jgcAEzR_Vw491DQR0zogdunJOj3qh0h_lvphdaHk>`_.



.. _faq:

FAQ
===

Why another XXX?  What about YYY?
---------------------------------
These are the knowingly related python projects:

- `OpenMDAO <http://openmdao.org/>`_:
  It has influenced pandalone's design.
  It is planned to interoperate by converting to and from it's data-types.
  But it works on python-2 only and its architecture needs attending from
  programmers (no `setup.py`, no official test-cases).

- `PyDSTool <http://www2.gsu.edu/~matrhc/PyDSTool.htm>`_:
  It does not overlap, since it does not cover IO and dependencies of data.
  Also planned to interoperate with it (as soon as we have
  a better grasp of it :-).
  It has some issues with the documentation, but they are working on it.

- `xray <http://xray.readthedocs.org/en/stable/faq.html>`_:
  Pandas for higher dimensions; data-trees should in principle work
  with "xray".

- `Blaze <http://blaze.pydata.org>`_:
  NumPy and Pandas interface to Big Data; data-trees should in principle work
  with "blaze".

- `netCDF4 <http://unidata.github.io/netcdf4-python/>`_:
  Hierarchical file-data-format similar to `hdf5`; a data-tree may derive
  in principle from "netCDF4 ".

- `hdf5 <http://www.h5py.org/>`_:
  Hierarchical file-data-format, `supported natively by pandas
  <http://pandas.pydata.org/pandas-docs/version/0.15.2/io.html#io-hdf5>`_;
  a data-tree may derive in principle from "netCDF4 ".

Which other projects/ideas have you reviewed when building this library?
------------------------------------------------------------------------
- `bubbles ETL <http://bubbles.databrewery.org/documentation.html>`_:
  Processing-pipelines for (mostly) categorical data.

- `Data-protocols <http://dataprotocols.org/>`_:

  - `JTSKit <https://github.com/okfn/jtskit-py>`_, A utility library for
    working with `JSON Table Schema <http://dataprotocols.org/json-table-schema/>`_
    in Python.
  - `Data Packages <http://dataprotocols.org/data-packages/>`_

- `Celery <http://www.celeryproject.org/>`_:
  Execute distributed asynchronous tasks using message passing on a single or
  more worker servers using multiprocessing, Eventlet, or gevent.

- `Fuzzywuzzy <https://github.com/seatgeek/fuzzywuzzy>`_ and
  `Jellyfish <https://github.com/sunlightlabs/jellyfish>`_:
  Fuzzy string matching in python.  Use it for writting code that can read
  coarsely-known column-names.

- `"Other's people's messy data (and how not to hate it)"
  <https://youtu.be/_eQ_8U5kruQ>`_,
  PyCon 2015(Canada) presentation by Mali Akmanalp.


.. _glossary:

Glossary
========
.. glossary::

    data-tree
        The *container* of data consumed and produced by a :term`model`, which
        may contain also the model.
        Its values are accessed using :term:`path` s.
        It is implemented by :class:`pandalone.pandata.Pandel` as
        a mergeable stack of :term:`JSON-schema` abiding trees of strings and
        numbers, formed with:

            - sequences,
            - dictionaries,
            - :mod:`pandas` instances, and
            - URI-references.

    value-tree
        That part of the :term:`data-tree`  that relates only to the I/O data
        processed.

    model
        A collection of :term:`component` s and accompanying :term:`mappings`.

    component
        Encapsulates a data-transformation function, using :term:`path`
        to refer to its inputs/outputs within the :term:`value-tree`.

    path
        A `/file/like` string functioning as the *id* of data-values
        in the :term:`data-tree`.
        It is composed of :term:`step`, and it follows the syntax of
        the :term:`JSON-pointer`.

    step
    pstep
    path-step
        The parts between between two conjecutive slashes(`/`) within
        a :term:`path`.  The :class:`Pstep` facilitates their manipulation.

    pmod
    pmods
    pmods-hierarchy
    mapping
    mappings
        Specifies a transformation of an "origin" path to
        a "destination" one (also called as "from" and "to" paths).
        The mapping always transforms the *final* path-step, and it can
        either *rename* or *relocate* that step, like that::

            ORIGIN          DESTINATION   RESULT_PATH
            ------          -----------   -----------
            /rename/path    foo       --> /rename/foo        ## renaming
            /relocate/path  foo/bar   --> /relocate/foo/bar  ## relocation
            /root           a/b/c     --> /a/b/c             ## Relocates all /root sub-paths.

        The hierarchy is formed by :class:`Pmod` instances,
        which are build when parsing the :term:`mappings` list, above.

    JSON-schema
        The `JSON schema <http://json-schema.org/>`_ is an `IETF draft
        <http://tools.ietf.org/html/draft-zyp-json-schema-03>`_
        that provides a *contract* for what JSON-data is required for
        a given application and how to interact with it.
        JSON Schema is intended to define validation, documentation,
        hyperlink navigation, and interaction control of JSON data.
        You can learn more about it from this `excellent guide
        <http://spacetelescope.github.io/understanding-json-schema/>`_,
        and experiment with this `on-line validator <http://www.jsonschema.net/>`_.

    JSON-pointer
        JSON Pointer(:rfc:`6901`) defines a string syntax for identifying
        a specific value within a JavaScript Object Notation (JSON) document.
        It aims to serve the same purpose as *XPath* from the XML world,
        but it is much simpler.



.. _begin-replacements:

.. |virtualenv| replace::  *virtualenv* (isolated Python environment)
.. _virtualenv: http://docs.python-guide.org/en/latest/dev/virtualenvs/

.. |pypi| replace:: *PyPi* repo
.. _pypi: https://pypi.python.org/pypi/pandalone

.. |winpython| replace:: *WinPython*
.. _winpython: http://winpython.github.io/

.. |anaconda| replace:: *Anaconda*
.. _anaconda: http://docs.continuum.io/anaconda/

.. |travis-status| image:: https://travis-ci.org/pandalone/pandalone.svg
    :alt: Travis build status
    :scale: 100%
    :target: https://travis-ci.org/pandalone/pandalone

.. |appveyor-status| image:: https://ci.appveyor.com/api/projects/status/jayah84y3ae7ddfc?svg=true
    :alt: Apveyor build status
    :scale: 100%
    :target: https://ci.appveyor.com/project/ankostis/pandalone

.. |cover-status| image:: https://coveralls.io/repos/pandalone/pandalone/badge.svg
    :target: https://coveralls.io/r/pandalone/pandalone

.. |docs-status| image:: https://readthedocs.org/projects/pandalone/badge/
    :alt: Documentation status
    :scale: 100%
    :target: https://readthedocs.org/builds/pandalone/

.. |pypi-ver| image::  https://img.shields.io/pypi/v/pandalone.svg
    :target: https://pypi.python.org/pypi/pandalone/
    :alt: Latest Version in PyPI

.. |python-ver| image:: https://img.shields.io/pypi/pyversions/pandalone.svg
    :target: https://pypi.python.org/pypi/pandalone/
    :alt: Supported Python versions

.. |downloads-count| image:: https://img.shields.io/pypi/dm/pandalone.svg?period=month
    :target: https://pypi.python.org/pypi/pandalone/
    :alt: Downloads

.. |github-issues| image:: https://img.shields.io/github/issues/pandalone/pandalone.svg
    :target: https://github.com/pandalone/pandalone/issues
    :alt: Issues count

.. |proj-license| image:: https://img.shields.io/badge/license-EUPL%201.1%2B-blue.svg
    :target: https://raw.githubusercontent.com/pandalone/pandalone/master/LICENSE.txt
    :alt: Project License

.. |dependencies| image:: https://img.shields.io/requires/github/pandalone/pandalone.svg
    :target: https://requires.io/github/pandalone/pandalone/requirements/
    :alt: Dependencies up-to-date?
