from string import ascii_uppercase
from re import compile
from xlrd import open_workbook
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


_re_cell_parser = compile(
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
    
        >>> from tempfile import gettempdir
        >>> from pandas import DataFrame, ExcelWriter
        >>> df = DataFrame([[None, None, None], [5, 6, 7]])
        >>> tmp = '/'.join([gettempdir(),'sample.xlsx'])
        >>> writer = ExcelWriter(tmp)
        >>> df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
        >>> writer.save()
    
        >>> import os, pandas as pd, tempfile
        >>> os.chdir(tempfile.mkdtemp())

        >>> url = '%s#%s!A1:C2[1,2]{"1":4,"2":"ciao"}'%(tmp, 'Sheet1')
        >>> res = url_parser(url)

        >>> sheet = res['xl_sheet']

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

    def no_emp_vec(fix_i, it_ind, inverse_fix=False, dbl_res=False):
        if inverse_fix:
            set_cell = lambda i_s: (i_s, fix_i)
        else:
            set_cell = lambda i_s: (fix_i, i_s)

        vect = {k: sheet.cell_value(*c)
                for k, c in ((i, set_cell(i_s)) for i, i_s in it_ind)
                if sheet.cell_type(*c) > 0}

        if dbl_res:
            return vect, min(vect) if vect else None
        else:
            return vect

    if cell_down is None:
        if cell_up.col is None:  # return row
            return no_emp_vec(cell_up.row, enumerate(range(sheet.ncols)))
        elif cell_up.row is None:  # return column
            return no_emp_vec(cell_up.col, enumerate(range(sheet.nrows)), True)
        else:  # return cell
            if cell_up.row < sheet.nrows and cell_up.col < sheet.ncols \
                    and sheet.cell_type(cell_up.row, cell_up.col) > 0:
                return sheet.cell_value(cell_up.row, cell_up.col)
            return None
    else:  # return table
        c1 = [-1 if i is None else i for i in cell_up]
        c2 = [-1 if i is None else i for i in cell_down]

        def set_lower_limits(j, max_i):
            if c2[j] < 0:
                if max_i <= c1[j]:
                    return True
                c2[j] = max_i
            else:
                c2[j] = min(c2[j] + 1, max_i)
            return False

        if set_lower_limits(0, sheet.ncols) or set_lower_limits(1, sheet.nrows):
            return {}

        it_col = list(enumerate(range(max(0, c1[0]), c2[0])))

        it_row = enumerate(range(max(0, c1[1]), c2[1]))

        rows = ((r, no_emp_vec(r_s, it_col, dbl_res=True)) for r, r_s in it_row)

        start = {'col': float('inf')}

        if c1[1] < 0 and c1[0] < 0:
            def set_min(min_c, r):
                start['col'] = min(min_c, start['col'])
                if 'row' not in start:
                    start['row'] = r
                return True
        elif c1[1] < 0:
            def set_min(min_c, r):
                if 'row' not in start:
                    start['row'] = r
                return True
        elif c1[0] < 0:
            def set_min(min_c, r):
                start['col'] = min(min_c, start['col'])
                return True
        else:
            set_min = lambda min_c, r: True

        table = {r: t_r for r, (t_r, m_c) in rows if t_r and set_min(m_c, r)}

        def shift_k(vect, k0, *ki):
            return {k - k0: shift_k(v, *ki) if ki else v for k, v in
                    vect.items()}

        if c1[0] < 0 and isinstance(start['col'], int):
            table = shift_k(table, start.get('row', 0), start['col'])
        elif start.get('row', 0) > 0:
            table = shift_k(table, start['row'])

        if table:
            if cell_down.col is not None and cell_down.col == cell_up.col:
                table = {k: v[0] for k, v in table.items()}
            if cell_down.row is not None and cell_down.row == cell_up.row:
                table = table[0]

        return table


_re_url_parser = compile(
    r'^\s*(?:(?P<xl_file_path>[^#\'\[\]?@!]+)\#{1})'  # xl file path
    r'(?:(?P<sheet_name>[^#!]+)\!){1}'  # xl sheet name
    r'(?P<cell_up>[A-Z]*\d*)'  # cell up
    r'(?:\:(?P<cell_down>[A-Z]*\d*))?'  # cell down [opt]
    r'(?P<json_args>\[.*\])?'  # json args [opt]
    r'(?P<json_kwargs>\{.*\})?'  # json kwargs [opt]
    r'\s*$')


def url_parser(url, xl_workbook=None):
    """
    Parses and fetches the contents of an 'excel_url'.

    :param url:
        a string with the following format:
        <xl_file_path>#<xl_sheet>!<cell_up>:<cell_down><json_args><json_kwargs>

    :param xl_workbook:
        a new xl workbook is opened if this is None
        type xlrd.book.Book

    :return:
        dictionary containing the following parameters:
        - xl_workbook
        - xl_sheet
        - cell_up
        - cell_down
        - json_args
        - json_kwargs
    :rtype: dict

    Example::
    
        >>> from tempfile import gettempdir
        >>> from pandas import DataFrame, ExcelWriter
        >>> df = DataFrame([[None, None, None], [5, 6, 7]])
        >>> tmp = '/'.join([gettempdir(), 'sample.xlsx'])
        >>> writer = ExcelWriter(tmp)
        >>> df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
        >>> writer.save()

        >>> url = '%s#%s!A1:C2[1,2]{"1":4,"2":"ciao"}' %(tmp, 'Sheet1')
        >>> res = url_parser(url)

        >>> res['xl_workbook']
        <xlrd.book.Book object at 0x...>
        >>> res['xl_sheet']
        <xlrd.sheet.Sheet object at 0x...>
        >>> res['cell_up']
        Cell(col=0, row=0)
        >>> res['cell_down']
        Cell(col=2, row=1)
        >>> res['json_args']
        [1, 2]
        >>> res['json_kwargs'] == {'2': 'ciao', '1': 4}
        True
    """

    res = _re_url_parser.match(url)
    try:
        res = res.groupdict(None)

        # parse cell_up
        res['cell_up'] = cell_parser(res['cell_up'])

        # parse cell_down
        if res['cell_down'] is not None:
            res['cell_down'] = cell_parser(res['cell_down'])

        # check range
        check_range(res['cell_up'], res['cell_down'])

        # open workbook
        if xl_workbook is None:
            res['xl_workbook'] = open_workbook(res.pop('xl_file_path'))
        else:
            res['xl_workbook'] = xl_workbook

        # open sheet
        res['xl_sheet'] = res['xl_workbook'].sheet_by_name(
            res.pop('sheet_name'))

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
        raise ValueError("Invalid excel-url(%s)!" % url)
