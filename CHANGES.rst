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
        - [x] Add API for returning sheet-names.
        - [ ] Use weak-refs for SheetsFactory (thanks Vinz)
    - Struct:
        - [x] Plugins for backends (& syntax?)
        - [x] Plugins for filters.
        - [ ] Plugins for syntax?
    - TCs
        - [ ] More TCs.
    - Backends:
        - [ ] Invert wrapping of impl-sheets --> attach attribute, to reuse them.
        - [ ] xlwings
        - [ ] Clipboard
        - [ ] openpyxl
        - [ ] google-drive sheets
    - [ ] Split own project
        - [ ] README
    - [ ] Check TODOs in code

Rejected TODOs:
---------------
- xleash:
  - Support cubic areas; pandas create dict-of-dfs from multiple sheets.
  - Use *ast* library for filters; cannot extract safely opts.
  - Build Lasso-structs trees on `recursive` filter for debugging; carefully
    crafted exception-messages is enough.


Changelog
=========

v0.2.2 (7-Feb-2017): "Telos" release
---------------------------------------
- pandas filter updates to `0.19.1`.
- `utils.ensure_file_ext()` accepts multiple extensions/regexes.


v0.2.1 (2-Dec-2016): "Stop" release
---------------------------------------
- remove unused features: doit, tkUI.
- travis-test on CONDA-->"standard" py; test also unde4r PY36-dev.


v0.2.0 (2-Nov-2016): "Really?" release
---------------------------------------
- xleash:
  - Plugins for backends and filters.
  - Packaging now supports 3 extras:
    - ``xlrd`` for the typical backend plugin,
    - ``xlwings`` for the new backend, excel-utils & tests,
    - ``pandas`` for filters plugins.

  - FIX & rename pandas-filter ``series --> sr`` - did not return a ``Lasso``.
  - Always convert xl-ref paths a "local" or "remote" urls to facilitate
    backends & use their `url.params` instead of filter `opts`.
  - Rename ``io._sheets --> io.backend``.

- xlutils, deps: Upgraded to ``xlwings-0.9.x`` released Aug/2/2016
  (see `migration guide <http://docs.xlwings.org/en/stable/migrate_to_0.9.html>`_)
  - Dropped util-functions (provided by now `xlwings`) or renamed:
    - ``xlutils.get_active_workbook()``
    - ``xlutils.get_workbook()``
    - ``tests._tutils.xw_Workbook() --> tests._tutils.xw_no_save_Workbook()``

- utils: Add more file/str functions from co2mpas-sampling
  (which/where, convpath, convert to/from came_case, publicize norm/abs paths)
- Unfortunately, Travis were down during the release (actually logs did not work),
  so TCs fail :-(


v0.1.13 (1-Nov-2016):
---------------------
- chore(deps): unpin OpenPyXL==1.8.6, openpyxl-444 & pandas-10125 have been fixed.
- fix(pandas): FIX pandas-v0.19+ dataframe-filter reading code


v0.1.12 (July-2016): "Stegh" release
-----------------------------------------
- xleash:
  - Make ``_parse_xlref_fragment()`` public (remove ``'_'`` from prefix).
  - #7: FIX ``"df"`` filter to read multi-index excel-tables
    that is supported since ``pandas-v0.16.x``.
- Add API for returning sheet-names: ``SheetsFactory.list_sheetnames()``
  and ``Sheet.list_sheetnames()``.
- Mark as "beta" in python trove-classifiers - del non-release opening note.


v0.1.11 (Apr-2016):
----------------------------------------
  - Fix regression on install-dependencies.


v0.1.10 (Apr-2016):
----------------------------------------
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

