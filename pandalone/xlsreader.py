from string import ascii_uppercase
import re, numpy as np
from collections import namedtuple
from json import loads

Cell = namedtuple('Cell', ['col', 'row'])


def col2num(upper_col_str):
    """
    Converts the Excel 'str' column ref in a 'int' column ref.

    :param upper_col_str: uppercase excel column ref

    :return:  excel column number [0, ...]
    :rtype: int

    Example::
    
        >>> col2num('D')
        3
    """

    num = 0
    for c in upper_col_str:
        num = num * 26 + ascii_uppercase.rindex(c) + 1

    return num - 1


def fetch_cell(cell, cell_col, cell_row):
    """
    Fetch a cell reference string.

    :param cell:
        whole cell reference string

    :param cell_col:
        col reference string

    :param cell_row:
        row reference string

    :return:
        a formatted cell
    :rtype: Cell

    Example::
        >>> fetch_cell('A1', 'A', '1')
        Cell(col=0, row=0)
        >>> fetch_cell('_1', '_', '1')
        Cell(col=None, row=0)
        >>> fetch_cell('A_', 'A', '_')
        Cell(col=0, row=None)
        >>> fetch_cell(':', None, None)
        Cell(col=None, row=None)
        >>> fetch_cell(None, None, None)

    """
    if cell is None:
        return None

    if cell_row != '0':
        row = int(cell_row) - 1 if cell_row and cell_row != '_' else None
        col = col2num(cell_col) if cell_col and cell_col != '_' else None
        return Cell(col=col, row=row)

    raise ValueError('unsupported row format %s' % cell_row)


def get_no_empty_cells(sheet, cell_up, cell_down=None):
    """
    Discovers a non-empty tabular-shaped region in the xl-sheet from a range.

    :param sheet:     a xlrd Sheet object
    :param cell_up:   a Cell object
    :param cell_down: a Cell object
    :return: matrix or vector
    :rtype: dict

    Example::

        >>> import os, tempfile, xlrd, pandas as pd
        >>> os.chdir(tempfile.mkdtemp())
        >>> df = pd.DataFrame([[None, None, None], [5, 6, 7]])
        >>> tmp = 'sample.xlsx'
        >>> writer = pd.ExcelWriter(tmp)
        >>> df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
        >>> writer.save()

        >>> sheet = xlrd.open_workbook(tmp).sheet_by_name('Sheet1')

        # minimum matrix in the sheet
        >>> get_no_empty_cells(sheet,  Cell(None, None), Cell(None, None))
        {0: {1: 0.0, 2: 1.0, 3: 2.0},
         1: {0: 0.0},
         2: {0: 1.0, 1: 5.0, 2: 6.0, 3: 7.0}}

        # up-left delimited minimum matrix
        >>> get_no_empty_cells(sheet, Cell(0, 0), Cell(None, None))
        {5: {4: 0.0, 5: 1.0, 6: 2.0},
         6: {3: 0.0},
         7: {3: 1.0, 4: 5.0, 5: 6.0, 6: 7.0}}
        >>> get_no_empty_cells(sheet, Cell(3, 6)) # get single value
        0.0
        >>> get_no_empty_cells(sheet, Cell(3, None)) # get column vector
        {6: 0.0,
         7: 1.0}
        >>> get_no_empty_cells(sheet, Cell(None, 5)) # get row vector
        {4: 0.0, 5: 1.0, 6: 2.0}

        # up-left delimited minimum matrix
        >>> get_no_empty_cells(sheet, Cell(4, None), Cell(None, None))
        {0: {0: 0.0, 1: 1.0, 2: 2.0},
         2: {0: 5.0, 1: 6.0, 2: 7.0}}

        # delimited matrix
        >>> get_no_empty_cells(sheet, Cell(3, 5), Cell(5, 7))
        {0: {1: 0.0, 2: 1.0},
         1: {0: 0.0},
         2: {0: 1.0, 1: 5.0, 2: 6.0}}

        # down-right delimited minimum matrix
        >>> get_no_empty_cells(sheet, Cell(None, None), Cell(5, 7))
        {0: {1: 0.0, 2: 1.0},
         1: {0: 0.0},
         2: {0: 1.0, 1: 5.0, 2: 6.0}}

        # up-down-right delimited minimum matrix
        >>> get_no_empty_cells(sheet, Cell(None, 6), Cell(5, 7))
        {0: {0: 0.0},
         1: {0: 1.0, 1: 5.0, 2: 6.0}}

        # down delimited minimum vector (i.e., column)
        >>> get_no_empty_cells(sheet, Cell(5, None), Cell(5, 7))
        {0: 1.0, 2: 6.0}

        # right delimited minimum vector (i.e., row)
        >>> get_no_empty_cells(sheet, Cell(2, 5), Cell(None, 5))
        {2: 0.0, 3: 1.0, 4: 2.0}
    """
    def fetch_cell(cell):
        if cell.ctype in (2,3):
            return float(cell.value)
        elif cell.ctype in (1,6):
            return int(cell.value)
        return None

    if cell_down is None:
        if cell_up.col is None:  # return row
            return list(map(fetch_cell, sheet.row(cell_up.row)))
        elif cell_up.row is None:  # return column
            return list(map(fetch_cell, sheet.col(cell_up.col)))
        else:  # return cell
            if cell_up.row < sheet.nrows and cell_up.col < sheet.ncols:
                return fetch_cell(sheet.cell(1,1))
            return None
    else:  # return table
        up = [0 if i is None else i for i in cell_up]

        if up[1] >= sheet.nrows or up[0] >= sheet.ncols:
            return None

        dn_row = cell_down.row + 1 if cell_down.row is not None else sheet.nrows

        table = [list(map(fetch_cell, sheet.row_slice(r, up[0], cell_down.col))) for r in range(dn_row)]

        def shift_k(vect, k0, *ki):
            return [shift_k(v, *ki) if ki else v for v in vect[k0:]]

        ki = [next((r for r, v in enumerate(t) if not all(v)), 0)
              for t,c in [(table, cell_up.row is None),
                          (table[0], cell_up.col is None)] if c]
        ki = [v for v in ki if v > 0]

        if ki:
            table = shift_k(table, *ki)

        if table[0]:
            if cell_down.col is not None and cell_down.col == cell_up.col:
                table = [v[0] for v in table]
            if cell_down.row is not None and cell_down.row == cell_up.row:
                table = table[0]
        else:
            return None

        return table


_re_url_fragment_parser = re.compile(
    r'^\s*(?:(?P<xl_st>[^!]+)!{1})?'            # xl sheet name
    r'(?P<cl_up>'                               # cell up [opt]
        r'(?P<up_col>[A-Z]+|_)'                 # up col
        r'(?P<up_row>\d+|_)'                    # up row
    r')?'
    r'(?:(?P<cl_dn>:(?:'                        # cell down [opt]
        r'(?P<dn_col>[A-Z]+|_)'                 # down col
        r'(?P<dn_row>\d+|_)'                    # down col
    r')?))?'
    r'(?P<js_ar>\[.*\])?'                       # json args [opt]
    r'(?P<js_kw>\{.*\})?'                       # json kwargs [opt]
    r'\s*$')


def url_fragment_parser(url_fragment):
    """
    Parses and fetches the contents of excel url_fragment.

    :param url_fragment:
        a string with the following format:
        <xl_sheet_name>!<cell_up>:<cell_down><json_args><json_kwargs>

    :return:
        dictionary containing the following parameters:
        - xl_sheet_name
        - cell_up
        - cell_down
        - json_args
        - json_kwargs
    :rtype: dict

    Example::

        >>> url_frg = 'Sheet1!:[1,2]{"1":4,"2":"ciao"}'
        >>> res = url_fragment_parser(url_frg)

        >>> res['xl_sheet_name']
        'Sheet1'
        >>> res['cell_up']
        Cell(col=None, row=None)
        >>> res['cell_down']
        Cell(col=None, row=None)
        >>> res['json_args']
        [1, 2]
        >>> res['json_kwargs'] == {'2': 'ciao', '1': 4}
        True
    """

    try:
        r = _re_url_fragment_parser.match(url_fragment).groupdict(None)

        res = {'xl_sheet_name': r['xl_st']}

        # resolve json_args
        res['json_args'] = loads(r['js_ar']) if r['js_ar'] else None

        # resolve json_kwargs
        res['json_kwargs'] = loads(r['js_kw']) if r['js_kw'] else None

        # fetch cell_down
        res['cell_down'] = fetch_cell(r['cl_dn'], r['dn_col'], r['dn_row'])

        # fetch cell_up
        if r['cl_up'] is None:
            if r['cl_dn']:
                res['cell_up'] = Cell(None, None)
            else:
                res['cell_up'], res['cell_down'] = (Cell(0, 0), Cell(None, None))
            return res

        res['cell_up'] = fetch_cell(r['cl_up'], r['up_col'], r['up_row'])

        # check range
        if res['cell_down'] is not None:

            def check_crossing(up, down):
                return down is not None and up is not None and down < up

            if check_crossing(res['cell_up'].row, res['cell_down'].row) or \
                    check_crossing(res['cell_up'].col, res['cell_down'].col):
                raise ValueError('%s < %s' % (r['cl_dn'], r['cl_up']))

        return res
    except:
        raise ValueError("Invalid excel-url(%s)!" % url_fragment)
