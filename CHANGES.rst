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
            - [x] Edge +/?(instead of -)
        - filters:
            - [ ] Slices and Index args on 'numpy' and 'df' filters.
            - [x] Recursive filter
        - [ ] Build XLref recursive trees.  
        - [ ] Xlrd-read with slices.
        - [X] Respect base-cell also for 1st edge filter
        - [x] Keep only single Lasso-stage in Ranger.
    - Struct:
        - [ ] Plugins for backends (& syntax?)
        - [x] Rename XLasso.
        - [x] Separate modules in _parse, _capture, _lasso .
    - TCs
        - [ ] More TCs.
        - [x] Test XlWings in Windows indeed OK.
        - [x] Recursive with real-excel file.
    - Backends:
        - [ ] xlwings
        - [ ] Clipboard
        - [ ] openpyxl
    - [x] Check TODOs in code
    - [ ] README
    - [ ] Split own project

Rejected TODOs:
---------------
- xlef: Support cubic areas; pandas create dict-of-dfs from multiple sheets.
- Use *ast* library for filters; cannot extract safely opts.


Changelog
=========

- v0.1.0 (XX-Sep-2015) 
- v0.0.11 (XX-May-2015) 
- v0.0.1.dev1 (01-March-2015) 

