#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import glob
import logging
import operator
import os
import re
from textwrap import dedent

from jsonschema._utils import URIDict
from jsonschema.compat import urlsplit, urljoin
import six

import pandas as pd


__commit__ = ""

log = logging.getLogger(__name__)

# Probably Windows-only:
#    TODO: Should become a class-method on xw.Workbook
# when/if it works reliably, see github/xlwings #30.


def get_active_workbook():
    from win32com.client import dynamic
    import xlwings as xw

    com_app = dynamic.Dispatch('Excel.Application')
    com_wb = com_app.ActiveWorkbook
    wb = xw.Workbook(xl_workbook=com_wb)

    return wb


def get_Workbook(wrkb_fname):
    """
    :param str wrkb_fname: if missing return the active excel-workbook 
    """
    import xlwings as xw

    if wrkb_fname:
        wb = xw.Workbook(wrkb_fname)
    else:
        wb = get_active_workbook()
    return wb


def _get_xl_vb_project(xl_wb):
    """
    To allow updating Excel's VBA, follow instructions at:
      https://social.msdn.microsoft.com/Forums/office/en-US/c175d0d7-7341-4b31-9699-712e5449cf78/access-to-visual-basic-project-is-not-trusted

      1. Go to Excel's Options.
      2. Click Trust Center.
      3. Click Trust Center Settings.
      4. Click Macro Settings.
      5. Click to select the `Trust access to the VBA project object model` check box.
      6. Click OK to close the Excel Options dialog box.

    or use the Excel-2010 shortcuts
      [ALT] + [t][m][s] 
    and then click the above check box. 
    """
    from win32com.universal import com_error
    import easygui

    def show_unlock_msg(msg):
        text = dedent(_get_xl_vb_project.__doc__)
#        try:
#            from tkinter import messagebox as msgbox
#        except ImportError:
#            import tkMessageBox as msgbox
#        msgbox.showinfo(title="Excel Permission Denied!", message=msg)
        easygui.textbox(title="Excel Permission Denied!", msg=msg, text=text)
        return msg

    try:
        xl_vbp = xl_wb.VBProject
        if xl_vbp.Protection == 1:
            msg = "The VBA-code in Excel is protected!"
            msg = show_unlock_msg(msg)
            raise Exception(msg)
        return xl_vbp
    except com_error as ex:  # @UndefinedVariable
        if ex.hresult == -2147352567:
            msg = "Excel complained: \n  %s" % ex.excepinfo[2]
            msg = show_unlock_msg(msg)
            raise Exception(msg)
        raise


def _gather_files(fpaths_wildcard):
    """
    :param str fpaths_wildcard: 'some/foo*bar.py'
    :return: a map {ext-less_basename --> full_path}.
    """
    def basename(fname):
        b, _ = os.path.splitext(os.path.basename(fname))
        return b

    return {basename(f): f for f in glob.glob(fpaths_wildcard)}


#############################
# Update VBA
#############################

XL_TYPE_MODULE = 1
XL_TYPE_SHEET = 100


def _remove_vba_modules(xl_vbcs, *mod_names_to_del):
    """
    :param VBProject.VBComponents xl_vbcs:
    :param str-list mod_names_to_del: which modules to remove, assumed *all* if missing
    """

    # Comparisons are case-insensitive.
    if mod_names_to_del:
        mod_names_to_del = map(str.lower, mod_names_to_del)

    # Gather all workbook vba-modules.
    xl_mods = [xl_vbc for xl_vbc in xl_vbcs if xl_vbc.Type == XL_TYPE_MODULE]

    # Remove those matching.
    #
    for m in xl_mods:
        if not mod_names_to_del or m.Name.lower() in mod_names_to_del:
            log.debug('Removing vba_module(%s)...', m.Name)
            xl_vbcs.Remove(m)


def _import_vba_files(xl_vbcs, vba_file_map):
    """
    :param VBProject.VBComponents xl_vbcs:
    :param dict vba_file_map: a map {module_name --> full_path}.

    """
    from win32com.universal import com_error

    cwd = os.getcwd()
    for vba_modname, vba_fpath in vba_file_map.items():
        try:
            log.debug('Removing vba_module(%s)...', vba_modname)
            old_xl_mod = xl_vbcs.Item(vba_modname)
            xl_vbcs.Remove(old_xl_mod)
            log.info('Removed vba_module(%s).', vba_modname)
        except com_error as ex:
            log.debug(
                'Probably vba_module(%s) did not exist, because: \n  %s', vba_modname, ex)
        log.debug(
            'Importing vba_module(%s) from file(%s)...', vba_modname, vba_fpath)
        xl_vbc = xl_vbcs.Import(os.path.join(cwd, vba_fpath))
        log.info('Imported %i LoC for vba_module(%s) from file(%s).',
                 xl_vbc.CodeModule.CountOfLines, vba_modname, vba_fpath)
        xl_vbc.Name = vba_modname


def _save_workbook(xl_workbook, path):
    # TODO: Remove when xlwings updated to latest.
    # From
    # http://stackoverflow.com/questions/21306275/pywin32-saving-as-xlsm-file-instead-of-xlsx
    xlOpenXMLWorkbookMacroEnabled = 52

    saved_path = xl_workbook.Path
    if (saved_path != '') and (path is None):
        # Previously saved: Save under existing name
        xl_workbook.Save()
    elif (saved_path == '') and (path is None):
        # Previously unsaved: Save under current name in current working
        # directory
        path = os.path.join(os.getcwd(), xl_workbook.Name)
        xl_workbook.Application.DisplayAlerts = False
        xl_workbook.SaveAs(path, FileFormat=xlOpenXMLWorkbookMacroEnabled)
        xl_workbook.Application.DisplayAlerts = True
    elif path:
        # Save under new name/location
        xl_workbook.Application.DisplayAlerts = False
        xl_workbook.SaveAs(path, FileFormat=xlOpenXMLWorkbookMacroEnabled)
        xl_workbook.Application.DisplayAlerts = True


def _save_excel_as_macro_enabled(xl_wb, new_fname=None):
    DEFAULT_XLS_MACRO_FORMAT = '.xlsm'

    if not new_fname:
        _, e = os.path.splitext(xl_wb.FullName)
        if e.lower()[-1:] != 'm':
            new_fname = xl_wb.FullName + DEFAULT_XLS_MACRO_FORMAT
            log.info('Cloning as MACRO-enabled the Input-workbook(%s) --> Output-workbook(%s)',
                     xl_wb.FullName, new_fname)
    _save_workbook(xl_wb, new_fname)

    return new_fname


def import_files_into_excel_workbook(infiles_wildcard, wrkb_fname=None, new_fname=None):
    """
    Add or update *xmlwings* VBA-code of a Workbook.

    :param str infiles_wildcard: 'some/foo*bar.vba'
    :param str wrkb_fname: filepath to update, active-workbook assumed if missing
    :param str new_fname: a macro-enabled (`.xlsm`, `.xltm`) filepath for updated workbook, 
                            `.xlsm` gets appended if current not a macro-enabled excel-format
    """

    wb = get_Workbook(wrkb_fname=None)
    xl_wb = wb.get_xl_workbook(wb)
    xl_vbp = _get_xl_vb_project(xl_wb)
    xl_vbcs = xl_vbp.VBComponents

    infiles_map = _gather_files(infiles_wildcard)
    log.info('Modules to import into Workbook(%s): %s', wb, infiles_map)

    if infiles_map:
        # TODO: _remove_vba_modules(xl_vbcs)

        # if xl_wb.FullName.lower()[-1:] != 'm':
        #    new_fname = _save_excel_as_macro_enabled(xl_wb, new_fname=new_fname)

        _import_vba_files(xl_vbcs, infiles_map)

        _save_excel_as_macro_enabled(xl_wb, new_fname=new_fname)

    return wb


#############################
# Excel-refs
#############################

# TODO: Convert Excel-ref RC-notation to A1
_excel_ref_specifier_regex = re.compile(r'''^\s*
            @
            (?:(?P<sheet>.+)!)?             # Sheet-name optional-group
            (?P<ref>                        # start Cell-ref 
                (?:[A-Z]+\d+ | R\d+C\d+ | \(\d+,\d+\))        # FROM-ref, RC/A1 notation, or tuple
                (?:                             # start TO-Cell-ref optional-group
                    :
                    (?:[A-Z]+\d+ | R\d+C\d+ | \(\d+,\d+\))        # RC/A1 notation, or tuple
                )?                              # end TO-Cell-ref optional-group
            )                               # end Cell-ref
            (?:                             # start Shape-specifier optional-group
                \.
                (?P<shape>table|vertical|horizontal)     # See respective xw.Range attributes
            )?                              # end Shape-specifier
            (?:\((?P<range_kws>                   # start RANGE-kws expression
                [^)]*
            )\))?                            # end RANGE-kws
            (?:{(?P<pandas_kws>                   # start PANDAS-kws expression
                [^)]*
            )})?                            # end PANDAS-kws
            \s*$''', re.X + re.IGNORECASE)
_undefined = object()


def _parse_kws(kws_str):
    if kws_str:
        local_vars = {}
        exec('kws = dict(%s)' % kws_str, None, local_vars)
        return local_vars['kws']
    return {}


class _AbsURIDict(URIDict):

    """Takes the absolute-path of local-urls (when schema is `file`)."""

    def normalize(self, uri):
        parts = urlsplit(uri)
        if 'file' == parts.scheme:
            parts = parts._replace(path=os.path.abspath(parts.path))
        return parts.geturl()


class _ExcelCntxt:

    """
    Needed to resolve `excel_urls` for detecting ref-cycles. 

    :ivar str     _wb_url:      the current workbook
    :ivar set     _visited:     (wb_abs_urls, excel_ref)
    :ivar URIDict _wb_cache:    wb_fpaths --> wb_abs_urls
    """

    def __init__(self, my_wb_url, wb_cache=None):
        """
        :param dict wb_cache:   optimization to avoid rebuilding cache when multiple resolutions
        """
        self._wb_cache = wb_cache or _AbsURIDict()
        self._wb_url = self._wb_id(my_wb_url)
        self._visited = set()

    def _wb_id(self, wb_url):
        wb_id = self._wb_cache.get(wb_url, None)
        if not wb_id:
            wb_id = os.path
            self._wb_cache[wb_url] = wb_id
        return wb_id

    def add_ref(self, excel_ref, wb_fpath=None):
        self._visited.append((wb_fpath, excel_ref))


def resolve_excel_url(url_str, default=_undefined, _cntxtx=None):
    """
    Parses and fetches the contents of an `excel_url`.

    Excel-url examples::

        file://wbook.xlsx#<excel_ref>
"""
    pass


def resolve_excel_ref(ref_str, default=_undefined, _cntxtx=None):
    """
    Parses and fetches the contents of an `excel_ref` (the hash-part of the `excel_url`).
    if `ref_str` is an *excel-ref*, it returns the referred-contents as DataFrame or a scalar, `None` otherwise.

    Excel-ref examples::

        #a1
        #E5.column
        #some sheet_name!R1C5.TABLE
        #1!a1:c5.table(header=False)
        #3!a1:C5.horizontal(strict=True; atleast_2d=True)
        #sheet-1!A1.table(asarray=True){columns=['a','b']}
        #any%sheet^&name!A1:A6.vertical(header=True)        ## Setting Range's `header` kw and 
                                                            #      DataFrame will parse 1st row as header

    The *excel-ref* syntax is case-insensitive apart from the key-value pairs, 
    which it is given in BNF-notation:

    .. productionlist::
            excel_ref   : "#" 
                        : [sheet "!"] 
                        : cells 
                        : ["." shape] 
                        : ["(" range_kws ")"] 
                        : ["{" df_kws "}"]
            sheet       : sheet_name | sheet_index
            sheet_name  : <any character>
            sheet_index : `integer`
            cells       : cell_ref [":" cell_ref]
            cell_ref    : A1_ref | RC_ref | tuple_ref
            A1_ref      : <ie: "A1" or "BY34">
            RC_ref      : <ie: "R1C1" or "R24C34">
            tuple_ref   : <ie: "(1,1)" or "(24,1)", the 1st is the row>
            shape       : "." ("table" | "vertical" | "horizontal")
            range_kws   : kv_pairs                    # keywords for xlwings.Range(**kws)
            df_kws      : kv_pairs                    # keywords for pandas.DataFrafe(**kws)
            kv_pairs    : <python code for **keywords ie: "a=3.14, f = 'gg'">


    Note that the "RC-notation" is not converted, so Excel may not support it (unless overridden in its options).
    """
    import xlwings as xw

    matcher = _excel_ref_specifier_regex.match(ref_str)
    if matcher:
        ref = matcher.groupdict()
        log.info("Parsed string(%s) as Excel-ref: %s", ref_str, ref)

        sheet = ref.get('sheet') or ''
        try:
            sheet = int(sheet)
        except ValueError:
            pass
        cells = ref['ref']
        range_kws = _parse_kws(ref.get('range_kws'))
        ref_range = xw.Range(sheet, cells, **range_kws)
        range_shape = ref.get('shape')
        if range_shape:
            ref_range = operator.attrgetter(range_shape.lower())(ref_range)

        v = ref_range.value

        if ref_range.row1 != ref_range.row2 or ref_range.col1 != ref_range.col2:
            # Parse as DataFrame when more than one cell.
            #
            pandas_kws = _parse_kws(ref.get('pandas_kws'))
            if 'header' in range_kws and not 'columns' in pandas_kws:
                # Do not ignore explicit-columns.
                v = pd.DataFrame(v[1:], columns=v[0], **pandas_kws)
            else:
                v = pd.DataFrame(v, **pandas_kws)

        log.debug("Excel-ref(%s) value: %s", ref, v)

        return v
    else:
        if default is _undefined:
            raise ValueError("Invalid excel-ref(%s)!" % ref_str)
        else:
            return default


def main(*argv):
    """
    Updates the vba code on excel workbooks.

    Usage::

        {cmd}  *.vbas                       # Add modules into active excel-workbook.
        {cmd}  *.vbas  in_file.xlsm         # Specify input workbook to add/update.
        {cmd}  *.vbas  foo.xlsx  bar.xlsm   # Specify input & output workbooks.
        {cmd}  *.vbas  foo.xls              # A `foo.xls.xlsm` will be created. 
        {cmd}  *.vbas  foo.xls   bar        # A `bar.xlm` will be created. 
        {cmd}  --pandalone inp.xls [out]    # Add 'pandalone-xlwings' modules.
    """

    cmd = os.path.basename(argv[0])
    if len(argv) < 2:
        exit('Too few arguments! \n%s' % dedent(main.__doc__.format(cmd=cmd)))
    else:
        if argv[1] == '--pandalone':
            mydir = os.path.dirname(__file__)
            argv = list(argv)
            argv[1] = os.path.join(mydir, '*.vba')
        elif argv[1].startswith('--'):
            exit('Unknown option(%s)! \n%s' %
                 (argv[1], dedent(main.__doc__.format(cmd=cmd))))

        import_files_into_excel_workbook(*argv[1:])


if __name__ == '__main__':
    import sys
    main(*sys.argv)
