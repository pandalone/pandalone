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
        - filters:
            - [ ] Slices and Index args on 'numpy' and 'df' filters.
            - [x] Recursive filter
        - Syntax:
            - [ ] Edge +/?(instead of -)
            - [ ] Notation for specifying the "last-sheet".
            - [ ] Extend RC-coords: ^-1, _-6, .-4 
            - [ ] Expand meander `@`?
        - [ ] Split Laso struct --> Xlref+Lasso(context) 
            - [ ] Build XLref recursive trees.  
        - [ ] Xlrd-read with slices.
        - [X] Respect base-cell also for 1st edge filter
        - [x] Keep only single Lasso-stage in Ranger.
    - Struct:
        - [x] Rename XLasso.
        - [x] Separate modules in _parse, _capture, _lasso .
        - [ ] plugin backends (& syntax?)
    - TCs
        - [x] Recursive with real-excel file.
        - [ ] More
        - [ ] Test XlWings in Windows indeed OK.
    - Backends:
        - [ ] Clipboard
        - [ ] xlwings
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

- v0.1.0 (XX-Aug-2015) 
- v0.0.11 (XX-May-2015) 
- v0.0.1.dev1 (01-March-2015) 

