from string import ascii_uppercase
from re import split
from xlrd import open_workbook
from xlrd.book import Book
from xlrd.sheet import Sheet
from collections import namedtuple, Sequence

Cell = namedtuple('Cell', ['col', 'row'])


def col2num(upper_col_str):
    """
    Example:
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


def str2cells_range(xls_range):
    """
    Example:
    >>> str2cells_range('a1')
    [(0, 0), None]
    >>> str2cells_range('a1:')
    [(0, 0), (None, None)]
    >>> str2cells_range(':b2')
    [(None, None), (1, 1)]
    >>> str2cells_range('a1:b2')
    [(0, 0), (1, 1)]
    >>> str2cells_range('1:b2')
    [(None, 0), (1, 1)]
    >>> str2cells_range('a:b2')
    [(0, None), (1, 1)]
    >>> str2cells_range(':')
    [(None, None), (None, None)]
    """
    if not isinstance(xls_range, str):
        raise TypeError("expected a 'str' object")

    cells = xls_range.upper().split(':')

    n_cells = len(cells)

    if n_cells > 2 or xls_range == '':
        raise ValueError("unsupported range format '%s'" % xls_range)

    for i, cl in enumerate(split('(\d+)', n) for n in cells):
        l = len(cl)
        if l == 1:
            cells[i] = cl + ['0']
        elif l == 3:
            if cl[-1] != '':
                raise ValueError("unsupported range format '%s'" % xls_range)
            cells[i] = cl[:-1]
        else:
            raise ValueError("unsupported range format '%s'" % xls_range)

    rng = [(col2num(c), int(r) - 1) for c, r in cells]

    rng = [(c if c >= 0 else None, r if r >= 0 else None) for c, r in rng]

    if n_cells == 1:
        return [rng[0], None]
    elif not any((rng[1][i] is not None and
            (rng[0][i] is not None and rng[1][i] < rng[0][i])) for i in [0, 1]):
        return rng
    raise ValueError('%s >= %s' % (cells[1], cells[0]))


def check_cell(cell_tuple):
    """
    Example:
    >>> check_cell((1, 2))
    True
    >>> check_cell((1, None))
    True
    >>> check_cell((1, -1))
    False
    >>> check_cell((None, -3))
    False
    >>> check_cell((1, 1, 2))
    False
    >>> check_cell(('1', '2'))
    False
    """
    try:
        return len(cell_tuple) == 2 and all(
            ((isinstance(c, int) and c >=0) or c is None) for c in cell_tuple)
    except:
        raise ValueError("unsupported cell format '%s'" % str(cell_tuple))


def cells_parser(*args):
    """
    Example:
    >>> cells_parser('A1', 'B2')
    [Cell(col=0, row=0), Cell(col=1, row=1)]
    >>> cells_parser('a1', 'B2')
    [Cell(col=0, row=0), Cell(col=1, row=1)]
    >>> cells_parser('a1')
    [Cell(col=0, row=0), None]
    >>> cells_parser((1, None))
    [Cell(col=1, row=None), None]
    >>> cells_parser('b', (2, 4))
    [Cell(col=1, row=None), Cell(col=2, row=4)]
    >>> cells_parser('A1:D2')
    [Cell(col=0, row=0), Cell(col=3, row=1)]
    >>> cells_parser('1:D2')
    [Cell(col=None, row=0), Cell(col=3, row=1)]
    >>> cells_parser('A:2')
    [Cell(col=0, row=None), Cell(col=None, row=1)]
    """
    n_args = len(args)
    if n_args > 2:
        raise TypeError(
            'get_cells() takes at most 2 argument (%d given)' % n_args)
    elif n_args == 0:
        raise TypeError('get_cells expected 1 arguments, got 0')

    def get_cell(arg, none=False):
        if isinstance(arg, str):
            c = str2cells_range(arg)
            if n_args == 1:
                return c
            elif c[1] is None:
                return c[0]
            else:
                raise ValueError("unsupported cell format '%s'" % str(args[0]))
        elif check_cell(arg) or (none and arg is None):
            return arg
        raise ValueError("unsupported cell format '%s'" % str(arg))

    res = [get_cell(a, bool(j)) for j, a in enumerate(args)]

    if len(res) < 2:
        res = res[0] if isinstance(res[0], list) else res + [None]
    if res[1] is not None:
        if any((res[1][i] is not None and
            (res[0][i] is not None and res[1][i] < res[0][i])) for i in [0, 1]):
            raise ValueError('%s >= %s' % (res[1], res[0]))
    else:
        if res[0][0] is None and res[0][1] is None:
            raise ValueError("unsupported cell format '%s'" % str(args[0]))

    return [Cell(*v) if v else None for v in res]


def sheet_parser(*args):
    """
    Example:
    >>> from pandas import DataFrame, ExcelWriter
    >>> df = DataFrame([[None, None, None], [5, 6, 7]])
    >>> writer = ExcelWriter('sample.xlsx')
    >>> df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
    >>> writer.save()
    >>> sheet_parser('sample.xlsx', 'Sheet1') #doctest: +ELLIPSIS
    <xlrd.sheet.Sheet object at 0x...>
    >>> from xlrd import open_workbook
    >>> wb = open_workbook('sample.xlsx')
    >>> sheet_parser(wb, 'Sheet1') #doctest: +ELLIPSIS
    <xlrd.sheet.Sheet object at 0x...>
    >>> sheet_parser(wb, 0) #doctest: +ELLIPSIS
    <xlrd.sheet.Sheet object at 0x...>
    >>> st = wb.sheet_by_name('Sheet1')
    >>> sheet_parser(st) #doctest: +ELLIPSIS
    <xlrd.sheet.Sheet object at 0x...>
    """
    if len(args) == 1 and isinstance(args[0], Sheet):
        return args[0]
    elif len(args) == 2:
        if isinstance(args[0], Book):
            wb = args[0]
        elif isinstance(args[0], str):
            wb = open_workbook(args[0])
        else:
            raise ValueError(
                "unsupported workbook format '%s'" % str(args[0]))

        if isinstance(args[1], int):
            return wb.sheet_by_index(args[1])
        elif isinstance(args[1], str):
            return wb.sheet_by_name(args[1])
        else:
            raise ValueError("unsupported sheet format '%s'" % str(args[1]))
    else:
        raise ValueError("unsupported sheet format '%s'" % str(args))


def args_parser(sheet, *args):
    """
    Example:
    >>> from pandas import DataFrame, ExcelWriter
    >>> df = DataFrame([[None, None, None], [5, 6, 7]])
    >>> writer = ExcelWriter('sample.xlsx')
    >>> df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
    >>> writer.save()
    >>> args_parser(('sample.xlsx', 'Sheet1'), ':') #doctest: +ELLIPSIS
    (<xlrd.sheet.Sheet object at 0x...>, Cell(col=None, row=None), \
Cell(col=None, row=None))
    >>> from xlrd import open_workbook
    >>> wb = open_workbook('sample.xlsx')
    >>> args_parser((wb, 'Sheet1'), (None, None), (None, None)) #doctest: +ELLIPSIS
    (<xlrd.sheet.Sheet object at 0x...>, Cell(col=None, row=None), \
Cell(col=None, row=None))
    >>> args_parser((wb, 0), ':') #doctest: +ELLIPSIS
    (<xlrd.sheet.Sheet object at 0x...>, Cell(col=None, row=None), \
Cell(col=None, row=None))
    >>> st = wb.sheet_by_name('Sheet1')
    >>> args_parser(st, ':') #doctest: +ELLIPSIS
    (<xlrd.sheet.Sheet object at 0x...>, Cell(col=None, row=None), \
Cell(col=None, row=None))
    """
    if isinstance(sheet, str) or not isinstance(sheet, Sequence):
        sheet = sheet_parser(sheet)
    else:
        sheet = sheet_parser(*sheet)

    cell_up, cell_down = cells_parser(*args)

    return sheet, cell_up, cell_down


def get_no_empty_cells(sheet, cell_up, cell_down=None):
    """
    Example::

    >>> from pandas import DataFrame, ExcelWriter
    >>> df = DataFrame([[None, None, None], [5, 6, 7]])
    >>> writer = ExcelWriter('sample.xlsx')
    >>> df.to_excel(writer, 'Sheet1', startrow=5, startcol=3)
    >>> writer.save()

    >>> sheet, cell_up, cell_down = args_parser(('sample.xlsx', 'Sheet1'), ':')
    >>> get_no_empty_cells(sheet, cell_up, cell_down)
    {0: {1: 0.0, 2: 1.0, 3: 2.0}, 1: {0: 0.0}, 2: {0: 1.0, 1: 5.0, 2: 6.0, 3: \
7.0}}
    >>> sheet, cell_up, cell_down = args_parser(('sample.xlsx', 'Sheet1'), 'D7')
    >>> get_no_empty_cells(sheet,  Cell(None, None), Cell(None, None))
    {0: {1: 0.0, 2: 1.0, 3: 2.0}, 1: {0: 0.0}, 2: {0: 1.0, 1: 5.0, 2: 6.0, 3: \
7.0}}
    >>> get_no_empty_cells(sheet, Cell(3, 6)) # get single value
    0.0
    >>> get_no_empty_cells(sheet, Cell(0, 0))

    >>> get_no_empty_cells(sheet, Cell(3, None)) # get column
    {6: 0.0, 7: 1.0}
    >>> get_no_empty_cells(sheet, Cell(None, 5)) # get row
    {4: 0.0, 5: 1.0, 6: 2.0}
    >>> get_no_empty_cells(sheet, Cell(4, None), Cell(None, None))
    {0: {0: 0.0, 1: 1.0, 2: 2.0}, 2: {0: 5.0, 1: 6.0, 2: 7.0}}
    >>> get_no_empty_cells(sheet, Cell(4, 6), Cell(None, None))
    {1: {0: 5.0, 1: 6.0, 2: 7.0}}
    >>> get_no_empty_cells(sheet, Cell(3, 5), Cell(5, 7))
    {0: {1: 0.0, 2: 1.0}, 1: {0: 0.0}, 2: {0: 1.0, 1: 5.0, 2: 6.0}}
    >>> get_no_empty_cells(sheet, Cell(None, None), Cell(5, 7))
    {0: {1: 0.0, 2: 1.0}, 1: {0: 0.0}, 2: {0: 1.0, 1: 5.0, 2: 6.0}}
    >>> get_no_empty_cells(sheet, Cell(None, 6), Cell(5, 7))
    {0: {0: 0.0}, 1: {0: 1.0, 1: 5.0, 2: 6.0}}
    >>> get_no_empty_cells(sheet, Cell(4, None), Cell(5, 7))
    {0: {0: 0.0, 1: 1.0}, 2: {0: 5.0, 1: 6.0}}
    >>> get_no_empty_cells(sheet, Cell(2, 5), Cell(None, 5))
    {2: 0.0, 3: 1.0, 4: 2.0}
    >>> get_no_empty_cells(sheet, Cell(0, 0), Cell(5, 7))
    {5: {4: 0.0, 5: 1.0}, 6: {3: 0.0}, 7: {3: 1.0, 4: 5.0, 5: 6.0}}
    >>> get_no_empty_cells(sheet, Cell(6, 8), Cell(None, None))
    {}
    >>> get_no_empty_cells(sheet, Cell(5, 8), Cell(None, None))
    {}
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
            if cell_up.row < sheet.nrows and \
               cell_up.col < sheet.ncols and \
               sheet.cell_type(cell_up.row,cell_up.col) > 0:
                return sheet.cell_value(cell_up.row,cell_up.col)
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
            return shift_k(table, start.get('row', 0), start['col'])
        elif start.get('row', 0) > 0:
            return shift_k(table, start['row'])

        if table:
            if cell_down.col is not None and cell_down.col == cell_up.col:
                table = {k: v[0] for k, v in table}
            if cell_down.row is not None and cell_down.row == cell_up.row:
                table = table[0]

        return table
