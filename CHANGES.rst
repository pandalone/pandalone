..    include:: <isonum.txt>

#######
Changes
#######

.. contents::


Known deficiencies
==================

.. _todos-list:

TODOs
-----
- XLeash
    - Core:
        - Syntax:
            - [ ] Notation for specifying the "last-sheet".
            - [ ] Extend RC-coords: ^-1, _-6, .-4
                - [ ] Cell becomes 4-tuple.
            - [ ] Expand meander `@`?
        - filters:
            - [ ] Slices and Index args on 'numpy' and 'df' filters.
            - [x] Impl  new `eval` filter with `asteval` lib
        - [ ] Xlrd-read with slices.
    - Struct:
        - [ ] Plugins for backends (& syntax?)
    - TCs
        - [ ] More TCs.
    - Backends:
        - [ ] xlwings
        - [ ] Clipboard
        - [ ] openpyxl
    - [ ] Split own project
        - [ ] README
    - [ ] Check TODOs in code

Rejected TODOs:
---------------
- xleash
    - Support cubic areas; pandas create dict-of-dfs from multiple sheets.
    - Use *ast* library for filters; cannot extract safely opts.
    - Build Lasso-structs trees on `recursive` filter for debugging; carefully
      crafted exception-messages is enough.


Changelog
=========

- v0.1.3 (Sep-2015):
    - Allow for perl-quoting xlrefs to avoid being treated as python-comments.

- v0.1.1 (Sep-2015): **1st working release**
    - FIX missing `xleash` package from wheel.
    - Renamed package `xlasso`--> `xleash` and  splitted `_filters` packages.
    - Added `py-eval` filter.
    - Accept xl-refs quoted by any char.

- v0.1.0 (Sep-2015): **XLasso BROKEN!**
    - 1st release in *pypi* but broken.
    - The `mappings` and `xlasso` packages are considered ready to be used.
- v0.0.11 (XX-May-2015)
- v0.0.1.dev1 (01-March-2015)

