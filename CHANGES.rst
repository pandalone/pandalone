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

- v0.1.6 (Sep-2015): 
    - xleash: Minor forgotten regression from previous fix context-sheet.
    - pstep: Make steps work as pandas-indexes.
    
- v0.1.5 (Sep-2015): properly fix context-sheet on `Ranger` anf `SheetsFactory`.

- v0.1.4 (Sep-2015): xleash fixes
    - xleash: Temporarily-Hacked for not closing sibling-sheets.
    - xleash: handle gracefully targeting-failures as *empty* captures.

- v0.1.3 (Sep-2015):
    - xleash: perl-quoting xlrefs to avoid being treated as python-comments.

- v0.1.1 (Sep-2015): **1st working release**
    - xleash:
        - FIX missing `xleash` package from wheel.
        - Renamed package `xlasso`--> `xleash` and  factor-out `_filters`
          module.
        - Added `py-eval` filter.
        - Accept xl-refs quoted by any char.

- v0.1.0 (Sep-2015): **XLasso BROKEN!**
    - Release in *pypi* broken, missing xlasso.
    - The `mappings` and `xlasso` packages are considered ready to be used.

- v0.0.11 (XX-May-2015)
- v0.0.1.dev1 (01-March-2015)

