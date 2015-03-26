#! python
#-*- coding: utf-8 -*-
#
# Copyright 2013-2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
# git: $Id$
''''Installation script for *pandalon*.


Install:
========
To install it, assuming you have download the sources,
do the usual::

    python setup.py install

Or get it directly from the PIP repository::

    pip install pandalon

Or get it directly from the github repository::

    pip install git+https://github.com/pandalone/pandalone.git
'''
## Got ideas for project-setup from many places, among others:
#    http://www.jeffknupp.com/blog/2013/08/16/open-sourcing-a-python-project-the-right-way/
#    http://python-packaging-user-guide.readthedocs.org/en/latest/current.html

import os, sys, io
import re

from setuptools import setup


## Fail early on ancient python-versions
#
py_ver = sys.version
if py_ver < "2.7":
    exit("Sorry, Python2 >= 2.7 is supported!")
if py_ver.startswith('3') and py_ver < "3.3":
    exit("Sorry, Python3 >= 3.3 is supported!")
if sys.argv[-1] == 'setup.py':
    exit("To install, run `python setup.py install`")
    
proj_name = 'pandalone'
mydir = os.path.dirname(__file__)



## Version-trick to have version-info in a single place,
## taken from: http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
##
def read_project_version():
    fglobals = {}
    with io.open(os.path.join(mydir, proj_name, '_version.py')) as fd:
        exec(fd.read(), fglobals)  # To read __version__
    return fglobals['__version__']

def read_text_lines(fname):
    with io.open(os.path.join(mydir, fname)) as fd:
        return fd.readlines()

def yield_sphinx_only_markup(lines):
    """
    :param file_inp:     a `filename` or ``sys.stdin``?
    :param file_out:     a `filename` or ``sys.stdout`?`

    """
    substs = [
        ## Selected Sphinx-only Roles.
        #
        (r':abbr:`([^`]+)`',        r'\1'),
        (r':ref:`([^`]+)`',         r'`\1`_'),
        (r':term:`([^`]+)`',        r'**\1**'),
        (r':dfn:`([^`]+)`',         r'**\1**'),
        (r':(samp|guilabel|menuselection):`([^`]+)`',        r'``\2``'),


        ## Sphinx-only roles:
        #        :foo:`bar`   --> foo(``bar``)
        #        :a:foo:`bar` XXX afoo(``bar``)
        #
        #(r'(:(\w+))?:(\w+):`([^`]*)`', r'\2\3(``\4``)'),
        (r':(\w+):`([^`]*)`', r'\1(``\2``)'),


        ## Sphinx-only Directives.
        #
        (r'\.\. doctest',           r'code-block'),
        (r'\.\. plot::',            r'.. '),
        (r'\.\. seealso',           r'info'),
        (r'\.\. glossary',          r'rubric'),
        (r'\.\. figure::',          r'.. '),
        (r'\.\. image::',          r'.. '),


        ## Other
        #
        (r'\|version\|',              r'x.x.x'),
    ]

    regex_subs = [ (re.compile(regex, re.IGNORECASE), sub) for (regex, sub) in substs ]

    def clean_line(line):
        try:
            for (regex, sub) in regex_subs:
                line = regex.sub(sub, line)
        except Exception as ex:
            print("ERROR: %s, (line(%s)"%(regex, sub))
            raise ex

        return line

    for line in lines:
        yield clean_line(line)



proj_ver = read_project_version()


readme_lines = read_text_lines('README.rst')
description = readme_lines[1]
long_desc = ''.join(yield_sphinx_only_markup(readme_lines))
## Trick from: http://peterdowns.com/posts/first-time-with-pypi.html
download_url = 'https://github.com/pandalone/%s/tarball/v%s' %(proj_name, proj_ver)

setup(
    name = proj_name,
    version=proj_ver,
    description=description,
    long_description=long_desc,
    author="Kostis Anagnostopoulos at European Commission (JRC)",
    author_email="ankostis@gmail.com",
    url = "https://github.com/pandalone/pandalone",
    download_url=download_url,
    keywords = [
        "python", "utility", "library", "data", "tree", "processing",
        "calculation", "dependencies", "resolution", "scientific", "engineering",
    ],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: Implementation :: CPython",
        "Development Status :: 1 - Planning",
        'Natural Language :: English',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: European Union Public Licence 1.1 (EUPL 1.1)",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    packages = ['pandalone'],
    #include_package_data = True,
    #package_data= {
    #    'pandalone': ['*.vba', '*.ico'],
    #},
    install_requires = [
        'six',
        'jsonschema>=2.4',
        'numpy',
        'pandas', #'openpyxl', 'xlrd',
        'Pillow',       ## For UI About boxes
        'xlwings',      ## For Excel integration
        'doit',
    ],
    setup_requires = [
        'setuptools',
        'setuptools-git >= 0.3', ## Gather package-data from all files in git.
        'sphinx >= 1.2', # >=1.3
        'sphinx_rtd_theme',
        'jsonschema >= 2.4',
        'coveralls',
        'wheel',
    ],
    tests_require = [
        'nose',
        'coverage',
    ],
    test_suite='nose.collector',
    entry_points={
        'console_scripts': [
            'pandalon = pandalon.__main__:main',
        ],
    }, 
    zip_safe=True,
    options={
        'build_sphinx' :{
            'build_dir': 'docs/_build',
        },
        'bdist_wheel' :{
            'universal': True,
        },
    }
)



