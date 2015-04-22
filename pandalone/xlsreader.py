from string import ascii_uppercase
from re import split
from xlrd import open_workbook
from xlrd.book import Book
from xlrd.sheet import Sheet


def col2num(upper_col_str):
    """
    Example:
    >>> col2num('D')
    3
    >>> col2num('AAA')
    702
    >>> col2num('')
    -1
    >>> col2num('a')
    Traceback (most recent call last):
        ...
    ValueError: unsupported column name format 'a'
    >>> col2num(1)
    Traceback (most recent call last):
        ...
    TypeError: expected a 'str' object
    >>> col2num('@_')
    Traceback (most recent call last):
        ...
    ValueError: unsupported column name format '@_'
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
    [(0, 0), (-1, -1)]
    >>> str2cells_range(':b2')
    [(-1, -1), (1, 1)]
    >>> str2cells_range('a1:b2')
    [(0, 0), (1, 1)]
    >>> str2cells_range('1:b2')
    [(-1, 0), (1, 1)]
    >>> str2cells_range('a:b2')
    [(0, -1), (1, 1)]
    >>> str2cells_range(':')
    [(-1, -1), (-1, -1)]
    >>> str2cells_range(1)
    Traceback (most recent call last):
        ...
    TypeError: expected a 'str' object
    >>> str2cells_range('')
    Traceback (most recent call last):
        ...
    ValueError: unsupported range format ''
    >>> str2cells_range('b:a')
    Traceback (most recent call last):
        ...
    ValueError: ['A', '0'] >= ['B', '0']
    >>> str2cells_range('a:b2:')
    Traceback (most recent call last):
        ...
    ValueError: unsupported range format 'a:b2:'
    >>> str2cells_range('ab2a2')
    Traceback (most recent call last):
        ...
    ValueError: unsupported range format 'ab2a2'
    >>> str2cells_range('a2a')
    Traceback (most recent call last):
        ...
    ValueError: unsupported range format 'a2a'
    >>> str2cells_range('@2')
    Traceback (most recent call last):
        ...
    ValueError: unsupported column name format '@'
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

    if n_cells == 1:
        return [rng[0], None]
    elif not any((rng[1][i] != -1 and rng[1][i] < rng[0][i]) for i in [0, 1]):
        return rng
    raise ValueError('%s >= %s' % (cells[1], cells[0]))


def check_cell(cell):
    """
    Example:
    >>> check_cell((1, 2))
    True
    >>> check_cell((1, -1))
    True
    >>> check_cell((1, -3))
    False
    >>> check_cell((None, -3))
    False
    >>> check_cell((1, 1, 2))
    False
    >>> check_cell(('1', '2'))
    False
    >>> check_cell(1)
    Traceback (most recent call last):
        ...
    ValueError: unsupported cell format '1'
    """
    try:
        return len(cell) == 2 and all(
            (isinstance(c, int) and c > -2) for c in cell)
    except:
        raise ValueError("unsupported cell format '%s'" % str(cell))


def get_cells(*args):
    """
    Example:
    >>> get_cells('A1', 'B2')
    [(0, 0), (1, 1)]
    >>> get_cells('a1', 'B2')
    [(0, 0), (1, 1)]
    >>> get_cells('a1')
    [(0, 0), None]
    >>> get_cells((1, -1))
    [(1, -1), None]
    >>> get_cells('b', (2, 4))
    [(1, -1), (2, 4)]
    >>> get_cells('b1', 'a2')
    Traceback (most recent call last):
        ...
    ValueError: (0, 1) >= (1, 0)
    >>> get_cells('b1', 'a2', (-1, -1))
    Traceback (most recent call last):
        ...
    TypeError: get_cells() takes at most 2 argument (3 given)
    >>> get_cells('b1:', 'a2')
    Traceback (most recent call last):
        ...
    ValueError: unsupported cell format 'b1:'
    >>> get_cells((1, -3))
    Traceback (most recent call last):
        ...
    ValueError: unsupported cell format '(1, -3)'
    >>> get_cells((-1, -1))
    Traceback (most recent call last):
        ...
    ValueError: unsupported cell format '(-1, -1)'
    >>> get_cells((1.0, -1))
    Traceback (most recent call last):
        ...
    ValueError: unsupported cell format '(1.0, -1)'
    >>> get_cells()
    Traceback (most recent call last):
        ...
    TypeError: get_cells expected 1 arguments, got 0
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
        if any((res[1][i] != -1 and res[1][i] < res[0][i]) for i in [0, 1]):
            raise ValueError('%s >= %s' % (res[1], res[0]))
    else:
        if res[0][0] < 0 and res[0][1] < 0:
            raise ValueError("unsupported cell format '%s'" % str(args[0]))

    return res


def get_no_empty_cells(*args):
    """
    Example:
    >>> from pandas import DataFrame, ExcelWriter
    >>> df = DataFrame([[None, None, None], [5, 6, 7]])
    >>> writer = ExcelWriter('sample.xlsx')
    >>> df.to_excel(writer, 'Sheet1', startrow = 5, startcol = 3)
    >>> writer.save()
    >>> get_no_empty_cells(('sample.xlsx', 'Sheet1'), ':') # get table
    {0: {1: 0.0, 2: 1.0, 3: 2.0}, 1: {0: 0.0}, 2: {0: 1.0, 1: 5.0, 2: 6.0, 3: 7.0}}
    >>> get_no_empty_cells('sample.xlsx', 'Sheet1', ':')
    Traceback (most recent call last):
        ...
    ValueError: unsupported input format '('sample.xlsx', 'Sheet1', ':')'
    >>> get_no_empty_cells(('sample.xlsx', ('Sheet1', )), ':')
    Traceback (most recent call last):
        ...
    ValueError: unsupported sheet format '('Sheet1',)'
    >>> get_no_empty_cells((('sample.xlsx',), 'Sheet1'), ':')
    Traceback (most recent call last):
        ...
    ValueError: unsupported workbook format '('sample.xlsx',)'
    >>> from xlrd import open_workbook
    >>> wb = open_workbook('sample.xlsx')
    >>> get_no_empty_cells((wb, 'Sheet1'), (-1, -1), (-1, -1))
    {0: {1: 0.0, 2: 1.0, 3: 2.0}, 1: {0: 0.0}, 2: {0: 1.0, 1: 5.0, 2: 6.0, 3: 7.0}}
    >>> get_no_empty_cells((wb, 0), ':')
    {0: {1: 0.0, 2: 1.0, 3: 2.0}, 1: {0: 0.0}, 2: {0: 1.0, 1: 5.0, 2: 6.0, 3: 7.0}}
    >>> st = wb.sheet_by_name('Sheet1')
    >>> get_no_empty_cells(st, ':')
    {0: {1: 0.0, 2: 1.0, 3: 2.0}, 1: {0: 0.0}, 2: {0: 1.0, 1: 5.0, 2: 6.0, 3: 7.0}}
    >>> get_no_empty_cells(st, 'D7') # get single value
    0.0
    >>> get_no_empty_cells(st, (0, 0))

    >>> get_no_empty_cells(st, 'D') # get column
    {6: 0.0, 7: 1.0}
    >>> get_no_empty_cells(st, '6') # get row
    {4: 0.0, 5: 1.0, 6: 2.0}
    >>> get_no_empty_cells(st, 'E:')
    {0: {0: 0.0, 1: 1.0, 2: 2.0}, 2: {0: 5.0, 1: 6.0, 2: 7.0}}
    >>> get_no_empty_cells(st, 'E7:')
    {1: {0: 5.0, 1: 6.0, 2: 7.0}}
    >>> get_no_empty_cells(st, 'D6:F8')
    {0: {1: 0.0, 2: 1.0}, 1: {0: 0.0}, 2: {0: 1.0, 1: 5.0, 2: 6.0}}
    >>> get_no_empty_cells(st, ':F8')
    {0: {1: 0.0, 2: 1.0}, 1: {0: 0.0}, 2: {0: 1.0, 1: 5.0, 2: 6.0}}
    >>> get_no_empty_cells(st, '7:F8')
    {0: {0: 0.0}, 1: {0: 1.0, 1: 5.0, 2: 6.0}}
    >>> get_no_empty_cells(st, 'E:F8')
    {0: {0: 0.0, 1: 1.0}, 2: {0: 5.0, 1: 6.0}}
    >>> get_no_empty_cells(st, 'A1:F8')
    {5: {4: 0.0, 5: 1.0}, 6: {3: 0.0}, 7: {3: 1.0, 4: 5.0, 5: 6.0}}
    >>> get_no_empty_cells(st, 'H9:')
    {}
    >>> get_no_empty_cells(st, 'F9:')
    {}
    """

    if isinstance(args[0], Sheet):
        sheet = args[0]
    elif isinstance(args[0], tuple) and len(args[0]) == 2:
        if isinstance(args[0][0], Book):
            wb = args[0][0]
        elif isinstance(args[0][0], str):
            wb = open_workbook(args[0][0])
        else:
            raise ValueError(
                "unsupported workbook format '%s'" % str(args[0][0]))

        if isinstance(args[0][1], int):
            sheet = wb.sheet_by_index(args[0][1])
        elif isinstance(args[0][1], str):
            sheet = wb.sheet_by_name(args[0][1])
        else:
            raise ValueError("unsupported sheet format '%s'" % str(args[0][1]))
        del wb
    else:
        raise ValueError("unsupported input format '%s'" % str(args))
    c1, c2 = get_cells(*args[1:])

    def no_emp_vec(fix_i, it_ind, reverse = False, dbl_res = False):
        if reverse:
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

    if c2 is None:
        if c1[0] < 0:  # return row
            return no_emp_vec(c1[1], enumerate(range(sheet.ncols)))
        elif c1[1] < 0:  # return column
            return no_emp_vec(c1[0], enumerate(range(sheet.nrows)), True)
        else:  # return cell
            if sheet.cell_type(c1[1], c1[0]) > 0:
                return sheet.cell_value(c1[1], c1[0])
            return None
    else:  # return table

        c2 = list(c2)  # to update

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

        return table
