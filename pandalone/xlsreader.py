from string import ascii_uppercase
import re, numpy as np
from collections import namedtuple
from json import loads

Cell = namedtuple('Cell', ['col', 'row'])


def col2num(upper_col_str):
    """
    Converts the Excel 'str' column ref in a 'int' column ref.

    :param upper_col_str: uppercase excel column ref

    :return:  excel column number [-1, ..]
    :rtype: int

    Example::
    
        >>> col2num('D')
        3
        >>> col2num('AAA')
        702
        >>> col2num('')
        -1
    """
    if not isinstance(upper_col_str, str):
        raise TypeError("expected a 'str' object")

    num = 0
    try:
        for c in upper_col_str:
            num = num * 26 + ascii_uppercase.rindex(c) + 1
    except:
        raise ValueError("unsupported column name format '%s'" % upper_col_str)

    return num - 1


_re_cell_parser = re.compile(
    r'^\s*(?P<col>[^\d]+)?'  # column [opt]
    r'(?P<row>[\d]+)?'  # row [opt]
    r'\s*$')


def cell_parser(cell):
    """
    Parses a cell reference string.

    :param cell:
        a cell reference string

    :return:
        a formatted cell
    :rtype: Cell

    Example::
        >>> cell_parser('A1')
        Cell(col=0, row=0)
        >>> cell_parser('1')
        Cell(col=None, row=0)
        >>> cell_parser('A')
        Cell(col=0, row=None)
        >>> cell_parser('')
        Cell(col=None, row=None)
    """

    cell = cell.replace('_', '')

    res = _re_cell_parser.match(cell)
    if res:
        if res.string == '':
            return Cell(col=None, row=None)

        res = res.groupdict(None)

        # parse col
        if res['col']:
            res['col'] = col2num(res['col'])

        # parse row
        if res['row'] is not None:
            if res['row'] == '0':
                raise ValueError('unsupported row format %s' % res['row'])
            res['row'] = int(res['row']) - 1

        return Cell(**res)

    raise ValueError("unsupported cell format '%s'" % cell)


def check_range(cell_up, cell_down):
    """
    Checks if the range is valid.

    :param cell_up:
        a Cell object

    :param cell_down:
        a Cell object

    :return:
        it raise if the range does not pas the check
    :rtype: nothing

    Example::
    
        >>> check_range(Cell(1, 2), Cell(None, None))

        >>> check_range(Cell(None, None), None)
        Traceback (most recent call last):
        ...
        ValueError: unsupported range format 'Cell(col=None, row=None), None'
        >>> check_range(Cell(1, 0), Cell(0, 1))
        Traceback (most recent call last):
        ...
        ValueError: Cell(col=0, row=1) < Cell(col=1, row=0)
    """
    if cell_down is not None:

        def check_crossing(up, down):
            return down is not None and up is not None and down < up

        if check_crossing(cell_up.row, cell_down.row) or \
                check_crossing(cell_up.col, cell_down.col):
            raise ValueError('%s < %s' % (str(cell_down), str(cell_up)))
    elif cell_up.row is None and cell_up.col is None:
        raise ValueError("unsupported range format '%s, None'" % str(cell_up))


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
    r'^\s*(?P<xl_sheet_name>[^#!]+)!'  # xl sheet name
    r'(?P<cell_up>(([A-Z]+|_)(\d+|_)|:\s*$|))'  # cell up
    r'(?::(?P<cell_down>(([A-Z]+|_)(\d+|_)|)))?'  # cell down [opt]
    r'(?P<json_args>\[.*\])?'  # json args [opt]
    r'(?P<json_kwargs>\{.*\})?'  # json kwargs [opt]
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

        >>> url = 'Sheet1!:[1,2]{"1":4,"2":"ciao"}'
        >>> res = url_fragment_parser(url)

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

    res = _re_url_fragment_parser.match(url_fragment)
    try:
        res = res.groupdict(None)

        # parse cell_up
        res['cell_up'] = cell_parser(res['cell_up'])

        # parse cell_down
        if res['cell_down'] is not None:
            res['cell_down'] = cell_parser(res['cell_down'])

        # check range
        check_range(res['cell_up'], res['cell_down'])

        # resolve json_args
        if res['json_args'] is not None:
            res['json_args'] = loads(res['json_args'])
        else:
            res.pop('json_args')

        # resolve json_kwargs
        if res['json_kwargs'] is not None:
            res['json_kwargs'] = loads(res['json_kwargs'])
        else:
            res.pop('json_kwargs')

        return res
    except:
        raise ValueError("Invalid excel-url(%s)!" % url_fragment)
