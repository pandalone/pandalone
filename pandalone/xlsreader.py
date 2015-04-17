import xlrd
from string import ascii_letters
from re import split


def col2num(col):
    if not isinstance(col,str):
        raise TypeError("expected a 'str' object")
    num = 0
    ord_A = ord('A')
    for c in col:
        if c in ascii_letters or c == '':
            num = num * 26 + (ord(c) - ord_A) + 1
        else:
            raise ValueError("unsupported column name format '%s'" %col)
    return num - 1

def str2cells_range(range):
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
    >>> str2cells_range('a2a')
    Traceback (most recent call last):
        ...
    ValueError: unsupported range format 'a2a'
    >>> str2cells_range('@2')
    Traceback (most recent call last):
        ...
    ValueError: unsupported column name format '@'
    """
    if not isinstance(range,str):
        raise TypeError("expected a 'str' object")

    cells = range.upper().split(':')

    n_cells = len(cells)

    if n_cells > 2 or range == '':
        raise ValueError("unsupported range format '%s'" %range)

    for i,cl in enumerate(split('(\d+)',n) for n in cells):
        l= len(cl)
        if l == 1:
            cells[i] =cl+['0']
        elif l == 3:
            if cl[-1] != '':
                raise ValueError("unsupported range format '%s'" %range)
            cells[i] = cl[:-1]
        else:
            raise ValueError("unsupported range format '%s'" %range)

    rng = [(col2num(c),int(r)-1) for c,r in cells]

    if n_cells==1:
        return rng + [None]
    elif not any((rng[1][i]!=-1 and rng[1][i]<rng[0][i]) for i in [0,1]):
        return rng
    raise ValueError('%s >= %s' %(cells[1],cells[0]))

def check_cell(cell):
    try:
        if len(cell)==2 and all(isinstance(c,int) for c in cell):
            return True
    except:
        pass
    raise ValueError("unsupported cell format '%s'" %cell)

def get_cells(*args):
    n_args = len(args)
    if n_args > 2:
        raise TypeError('get_cells() takes at most 2 argument (%d given)'%n_args)

    def get_cell(arg, none = False):
        if isinstance(arg,str):
            c = str2cells_range(arg)
            if n_args==1:
                return c
            elif c[1] is None:
                return c[0]
            else:
                raise ValueError("unsupported cell format '%s'" %str(args[0]))
        elif check_cell(arg) or (none and arg is None):
            return arg
        raise ValueError("unsupported cell format '%s'" %arg)

    res = [get_cell(a,j) for j,a in enumerate(args)]

    if len(res)<2:
        res = res[0] if isinstance(res[0],list) else res + [None]
    if res[1] is not None:
        if any((res[1][i]!=-1 and res[1][i]<res[0][i]) for i in [0,1]):
            raise ValueError('%s >= %s' %(res[1],res[0]))
    else:
        if res[0][0]<0 and res[0][1]<0:
            raise ValueError("unsupported cell format '%s'" %str(args[0]))

    return res

def get_table(sheet, *args):

    c1, c2 = get_cells(*args)
    max_row = sheet.nrows
    max_col = sheet.ncols
    if c2 is None:
        if c1[0] < 0:
            t={}
            for c,c_s in enumerate(range(max_col)):
                if sheet.cell_type(c1[1],c_s)>0:
                    t[c]=sheet.cell_value(c1[1],c_s)
            if t: t={0:t}
            return t
        elif c1[1] < 0:
            t={}
            for r,r_s in enumerate(range(max_row)):
                if sheet.cell_type(r_s,c1[0])>0:
                    t[r]=sheet.cell_value(r_s,c1[0])
            if t: t={k:{0:v} for k,v in t.items()}
            return t

        else:
            if sheet.cell_type(c1[1],c1[0])>0:
                return {0:{0:sheet.cell_value(c1[1],c1[0])}}
            return {}
    else:
        c2=list(c2)
        if c2[0]<0:
            if max_col<=c1[0]:
                return {}
            c2[0] = max_col
        else:
            c2[0] = min(c2[0]+1,max_col)

        if c2[1]<0:
            if max_row<=c1[1]:
                return {}
            c2[1] = max_row
        else:
            c2[1] = min(c2[1]+1,max_row)

        if max_row<=c1[1] or max_col<=c1[0]:
            return {}

        t = {}

        it_col = list(enumerate(range(max(0,c1[0]),c2[0])))

        for r,r_s in enumerate(range(max(0,c1[1]),c2[1])):
            t[r]={}
            t_r = t[r]
            for c,c_s in it_col:
                if sheet.cell_type(r_s,c_s)>0:
                    t_r[c]=sheet.cell_value(r_s,c_s)
            if not t_r: t.pop(r)

        return t

wb = xlrd.open_workbook('Book1.xlsx')
sheet = wb.sheet_by_index(0)
print(get_table(sheet,':F3'))