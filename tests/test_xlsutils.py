#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import os
import sys
import unittest

from numpy import testing as npt
from pandas.core.generic import NDFrame

import numpy as np
import pandas as pd
from tests._tutils import (
    _init_logging, TemporaryDirectory, check_excell_installed, xw_Workbook,
    xw_close_workbook)


log = _init_logging(__name__)
is_excel_installed = check_excell_installed()


def from_my_path(*parts):
    return os.path.join(os.path.dirname(__file__), *parts)


def _make_sample_sheet(wb, sheetname, addr, value):
    import xlwings as xw

    xw.Sheet(1).name = sheetname
    xw.Range(addr).value = value


@unittest.skipIf(not is_excel_installed, "Cannot test xlwings without MS Excel.")
class TestExcel(unittest.TestCase):

    def test_build_excel(self):
        from pandalone import xlsutils

        with TemporaryDirectory() as tmpdir:
            wb_inp_fname = from_my_path('..', 'excel', 'ExcelRunner.xlsm')
            wb_out_fname = from_my_path(tmpdir, 'ExcelRunner.xlsm')
            vba_wildcard = from_my_path('..', 'excel', '*.vba')
            try:
                wb = xlsutils.import_files_into_excel_workbook(
                    vba_wildcard, wb_inp_fname, wb_out_fname)
            finally:
                if 'wb' in locals():
                    xw_close_workbook(wb)

    def test_xlwings_smoketest(self):
        import xlwings as xw
        sheetname = 'shitt'
        addr = 'f6'
        table = pd.DataFrame([[1, 2], [True, 'off']], columns=list('ab'))
        with xw_Workbook() as wb:  # FIXME: Ugrade xlwings, Leaves exel open.
            xw.Sheet(1).name = sheetname
            xw.Range(addr).value = table


if __name__ == "__main__":
    unittest.main()
