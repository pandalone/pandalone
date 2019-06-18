#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import os
from tempfile import TemporaryDirectory
from pandalone import xlsutils
from tests._tutils import init_logging, xw_no_save_Workbook, xw_close_workbook
import unittest

import pandas as pd


log = init_logging(__name__)
is_excel_installed = xlsutils.check_excell_installed()


def from_my_path(*parts):
    return os.path.join(os.path.dirname(__file__), *parts)


@unittest.skipIf(not is_excel_installed, "Cannot test xlwings without MS Excel.")
class TestExcel(unittest.TestCase):
    def test_build_excel(self):
        with TemporaryDirectory() as tmpdir:
            wb_inp_fname = from_my_path("empty.xlsx")
            wb_out_fname = from_my_path(tmpdir, "ExcelRunner.xlsm")
            vba_wildcard = from_my_path("..", "pandalone", "excel", "*.vba")
            try:
                wb = xlsutils.import_vba_into_excel_workbook(
                    vba_wildcard, wb_inp_fname, wb_out_fname
                )
            finally:
                if "wb" in locals():
                    xw_close_workbook(wb)

    def test_xlwings_smoketest(self):
        import xlwings as xw

        sheetname = "shitt"
        addr = "f6"
        table = pd.DataFrame([[1, 2], [True, "off"]], columns=list("ab"))
        with xw_no_save_Workbook() as wb:
            xw.Sheet(1).name = sheetname
            xw.Range(addr).value = table


if __name__ == "__main__":
    unittest.main()
