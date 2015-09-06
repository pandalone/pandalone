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
- XLasso
    - Core:
        - Syntax:
            - [ ] Notation for specifying the "last-sheet".
            - [ ] Extend RC-coords: ^-1, _-6, .-4
                - [ ] Cell becomes 4-tuple.
            - [ ] Expand meander `@`?
        - filters:
            - [ ] Slices and Index args on 'numpy' and 'df' filters.
        - [ ] Build XLref recursive trees.
        - [ ] Xlrd-read with slices.
    - Struct:
        - [ ] Plugins for backends (& syntax?)
    - TCs
        - [ ] More TCs.
    - Backends:
        - [ ] xlwings
        - [ ] Clipboard
        - [ ] openpyxl
    - [ ] Check TODOs in code
    - [ ] README
    - [ ] Split own project

Rejected TODOs:
---------------
- xlef: Support cubic areas; pandas create dict-of-dfs from multiple sheets.
- Use *ast* library for filters; cannot extract safely opts.


Changelog
=========

- v0.1.0 (Sep-2015)
    - The `mappings` and `xlasso` packages are considered ready to be used.
- v0.0.11 (XX-May-2015)
- v0.0.1.dev1 (01-March-2015)

