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
            - [ ] Extend RC-coords: ^-1, _[-6], .-4
                - [ ] Cell becomes 4-tuple.
            - [ ] Expand meander `@`?
        - filters:
            - [ ] Slices and Index args on 'numpy' and 'df' filters.
        - [ ] Xlrd-read with slices.
    - Struct:
        - [ ] Plugins for backends (& syntax?)
    - TCs
        - [ ] More TCs.
    - Backends:
        - [ ] xlwings
        - [ ] Clipboard
        - [ ] openpyxl
        - [ ] google-drive sheets
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
    - Add API for returning sheet-names: ``SheetsFactory.list_sheetnames()``
      and ``Sheet.list_sheetnames()``.


Changelog
=========

- v0.1.12 (July-2016): Finikounta release
    - xleash:
      - Make ``_parse_xlref_fragment()`` public (remove ``'_'`` from prefix).
      - #7: FIX ``"df"`` filter to read multi-index excel-tables
        that is supported since ``pandas-v0.16.x```.

    - Mark as "beta" in python trove-classifiers - del non-release opening note.


- v0.1.11 (Apr-2016):
    - Fix regression on install-dependencies.


- v0.1.10 (Apr-2016):
    - xleash:
      - #6: Gracefully handle absolute-paths & file-URLs.
      - #8: Accept xlrefs with only sheets (without rect edges)s.
      - #9: always return [] on empty sheets.
      - **Known issues:** TCs related to asteval and pandas-multi-indexing fail
        and need investigation/fixing respectively.

    - pandata: Add ``resolve_path()`` supporting also relative-paths.
    - TravisCI: Run TCs also on *py3.5*, stop testing on *py3.3*.
    - **Known issues:** Dev-dependencies required for installation (regression).


- v0.1.9 (Dec-2015):
    - pstep: Add ``pstep_from_df()`` utility.


- v0.1.8 (Sep-2015):
    - deps: Do not require flake8.

- v0.1.7 (Sep-2015):
    - deps: Do not enforce pandas/numpy version.

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

