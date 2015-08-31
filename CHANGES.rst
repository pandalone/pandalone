..    include:: <isonum.txt>

#######
Changes
#######

.. contents::


Known deficiencies
==================

.. _todos-list:

TODOs
=====
- XLasso
    - Core:
        - filters:
            - [x] Recursive filter
            - [ ] Slices and Index args on 'numpy' and 'df' filters.
        - Syntax:
            - [ ] Edge +/?(instead of -)
            - [ ] Use *ast* library for filters
            - [ ] Coord: RC, ^-1,_-6
            - [ ] Expand 5+/5?
        - [ ] Xlrd-read with slices.
        - [ ] OK: Keep only single Lasso-stage in Ranger.
    - Struct:
        - [ ] split lassos from xlref.
        - [ ] plugin backends (& syntax?)
    - TCs
        - [ ] More
        - [ ] test XlWings in Windows indeed OK.
    - Backends:
        - [ ] Clipboard
        - [ ] xlwings
        - [ ] openpyxl
    - [x] Check TODOs in code
    - [ ] README
    - [ ] Split own project
        - Rename XLasso

Changelog
=========

- v0.1.0 (XX-Aug-2015) 
- v0.0.11 (XX-May-2015) 
- v0.0.1.dev1 (01-March-2015) 

