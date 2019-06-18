#! python
# -*- coding: utf-8 -*-
#
# Copyright 2013-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import os
import re
import sys

from collections.abc import Sequence

import os.path as osp


__commit__ = ""

# Python-2 compatibility
#
try:  # pragma: no cover
    FileNotFoundError
except NameError:  # pragma: no cover
    FileNotFoundError = IOError  # @ReservedAssignment
else:  # pragma: no cover
    FileNotFoundError = OSError  # @ReservedAssignment


##############
#  Utilities
#


def str2bool(v):
    """
    Utility for parsing cmd-line args.

    :param str v: any of (case insensitive): yes/no, true/false, on/off

    Example::

        >>> str2bool('ON') == str2bool('on') == str2bool('12') == True
        True

        >>> str2bool('') == str2bool('  ') == str2bool('0') == False
        True
        >>> str2bool('no') == str2bool('off') == str2bool('off') == False
        True

        >>> str2bool(0)
        Traceback (most recent call last):
        ValueError: Invalid str-boolean(0) due to: 'int' object has no attribute 'strip'
        >>> str2bool(None)
        Traceback (most recent call last):
        ValueError: Invalid str-boolean(None) due to: 'NoneType' object has no attribute 'strip'

    """
    try:
        vv = v.strip().lower()
        if vv in ("yes", "true", "on"):
            return True
        if vv in ("no", "false", "off", "0"):
            return False
        return bool(vv)
    except Exception as ex:
        msg = "Invalid str-boolean(%s) due to: %s"
        raise ValueError(msg % (v, ex))


def obj2bool(v):
    """
    Utility for parsing anything to bool.

    :param v:
            any of (case insensitive): yes/no, true/false, on/off, `None`,
            or object, optionally with :meth:``__bool__``.

    Example::

        >>> obj2bool(None) == obj2bool('') == obj2bool(0) == False
        True
    """
    return bool(v) and str2bool(v)


def is_travis():  # pragma: no cover
    return "TRAVIS" in os.environ


def as_list(o):
    if isinstance(o, Sequence) and not isinstance(o, str):
        o = list(o)
    else:
        o = [o]
    return o


_camel_to_snake_regex = re.compile("(?<=[a-z0-9])([A-Z]+)")  # ('(?!^)([A-Z]+)')


def camel_to_snake_case(s):
    """Turns `'CO2DiceApp' --> 'co2_dice_app'. """
    return _camel_to_snake_regex.sub(r"_\1", s).lower()


def camel_to_cmd_name(s):
    """Turns `'CO2DiceApp' --> 'co2-dice-app'. """
    return camel_to_snake_case(s).replace("_", "-")


# def format_pairs(items: Sequence[Tuple[Text, Any]], indent=16):
def format_pairs(items, indent=16):
    def format_item(k, v):
        nk = len(k)
        ntabs = max(1, int(nk / indent) + bool(nk % indent))
        key_width = ntabs * indent
        item_pattern = "%%-%is = %%s" % key_width
        return item_pattern % (k, v)

    dic = [format_item(*i) for i in items]

    return "\n".join(dic)


def first_line(doc):
    for l in doc.split("\n"):
        if l.strip():
            return l.strip()


_file_drive_regex = re.compile(r"^([a-z]):(/)?(.*)$", re.I)
_is_dir_regex = re.compile(r"[^/\\][/\\]$")
_unc_prefix = "\\\\?\\"


def normpath(path):
    """Like :func:`osp.normpath()`, but preserving last slash."""
    p = osp.normpath(path)
    if _is_dir_regex.search(path) and p[-1] != os.sep:
        p = p + osp.sep
    return p


def abspath(path):
    """Like :func:`osp.abspath()`, but preserving last slash."""
    p = osp.abspath(path)
    if _is_dir_regex.search(path) and p[-1] != os.sep:
        p = p + osp.sep
    return p


def convpath(fpath, abs_path=True, exp_user=True, exp_vars=True):
    """Without any flags, just pass through :func:`osp.normpath`. """
    if exp_user:
        fpath = osp.expanduser(fpath)
    if exp_vars:
        # Mask UNC '\\server\share$\path` from expansion.
        fpath = fpath.replace("$\\", "_UNC_PATH_HERE_")
        fpath = osp.expandvars(fpath)
        fpath = fpath.replace("_UNC_PATH_HERE_", "$\\")
    fpath = abspath(fpath) if abs_path else normpath(fpath)
    return fpath


def path2urlpath(path):
    r"""Like :func:`ur.path2urlpath()`, but eliminiating UNC(\\?\) and preserving last slash."""
    import urllib.request as ur

    if path.startswith(_unc_prefix):
        path = path[3:]
    u = ur.pathname2url(path)
    if _is_dir_regex.search(path) and u[-1] != "/":
        u = u + "/"
    return u


def urlpath2path(url):
    """Like :func:`ur.url2pathname()`, but prefixing with UNC(\\\\?\\) long paths and preserving last slash."""
    import urllib.request as ur

    p = ur.url2pathname(url)
    if _is_dir_regex.search(url) and p[-1] != os.sep:
        p = p + osp.sep
    if len(p) > 200:
        p += _unc_prefix
    return p


def path2url(path, expandvars=False, expanduser=False):
    """
    Converts path to local('file:') URL, while remote (http:) URLs pass through.

    Windows cases handled are:

      - foo/bar/                     --> file:///D:/CWD/foo/bar/  ## (prefix CWD)
      - D:foo/bar                    --> file:///D:/foo/bar
      - /foo/bar                     --> file:///D:/foo/bar       ## (drive from CWD)
      - ABS WITH drive-letter        --> LOCAL ABS
      - remote REL/ABS WITH/WITHOUT drive-letter pass through.
      - local/remote ABS UNC-paths   --> LOCAL/REMOTE ABS

    :param str path: anything descrbed above

    Complexity because Bill Gates copied the methods of Methodios and Kyrilos.
    """
    import urllib.parse as up
    import urllib.request as ur

    if path:
        if expandvars:
            path = osp.expandvars(path)

        # Trim UNCs, urljoin() makes nonsense, path2urlpath() just fails.
        if path.startswith(_unc_prefix):
            path = path[3:]

        # UNIXize *safely* and join with base-URL,
        # UNLESS it start with drive-letter (not to assume it as schema).
        #
        path = path.replace("\\", "/")
        m = _file_drive_regex.match(path)
        if m:
            # A pesky Drive-relative path...assume it absolute!
            #
            if not m.group(2):
                path = "%s:/%s" % (m.group(1), m.group(3))
            path = "file:///%s" % path
        else:
            # Use CWD as  URL-base to make it absolute.
            #
            cwd = ur.pathname2url("%s/" % os.getcwd())
            baseurl = up.urljoin("file:", cwd)
            path = up.urljoin(baseurl, path)

        # Expand vars, conditionally on remote or local URL.
        #
        parts = up.urlsplit(path)
        p = parts.path
        if parts.scheme == "file" and expanduser:
            p = osp.expanduser(p)
        p = normpath(p).replace("\\", "/")
        path = up.urlunsplit(parts._replace(path=p))

    return path


def generate_filenames(filename):
    f, e = os.path.splitext(filename)
    yield filename
    i = 1
    while True:
        yield "%s%i%s" % (f, i, e)
        i += 1


def make_unique_filename(fname, filegen=generate_filenames):
    fname_genor = generate_filenames(fname)
    fname = next(fname_genor)
    while os.path.exists(fname):
        fname = next(fname_genor)
    return fname


def ensure_file_ext(fname, ext, *exts, is_regex=False):
    r"""
    Ensure that the filepath ends with the extension(s) specified.

    :param str ext:
        The 1st extension (with/without dot `'.'`) that will append if none matches,
        so must not be a regex.
    :param str exts:
        Other extensions. They may be regexes,
        depending on `is_regex`; a `'$'` is added the end.
    :param bool is_regex:
        When true, the rest `exts` are parsed as case-insensitive regexes.

    Example::

        >>> ensure_file_ext('foo', '.bar')
        'foo.bar'
        >>> ensure_file_ext('foo.', '.bar')
        'foo.bar'
        >>> ensure_file_ext('foo', 'bar')
        'foo.bar'
        >>> ensure_file_ext('foo.', 'bar')
        'foo.bar'

        >>> ensure_file_ext('foo.bar', 'bar')
        'foo.bar'


        >>> ensure_file_ext('foo.BAR', '.bar')
        'foo.BAR'
        >>> ensure_file_ext('foo.DDD', '.bar')
        'foo.DDD.bar'
        

    When more are given::

        >>> ensure_file_ext('foo.xlt', '.xlsx', '.XLT')
        'foo.xlt'
        >>> ensure_file_ext('foo.xlt', '.xlsx', '.xltx')
        'foo.xlt.xlsx'

    And when regexes::

        >>> ensure_file_ext('foo.xlt', '.xlsx',  r'\.xl\w{1,2}', is_regex=True)
        'foo.xlt'
        >>> ensure_file_ext('foo.xl^', '.xls',  r'\.xl\w{1,2}', is_regex=True)
        'foo.xl^.xls'

    """
    _, file_ext = os.path.splitext(fname)

    if is_regex:
        ends_with_ext = any(
            re.match(e + "$", file_ext, re.IGNORECASE) for e in (re.escape(ext),) + exts
        )
    else:
        file_ext = file_ext.lower()
        ends_with_ext = any(file_ext.endswith(e.lower()) for e in (ext,) + exts)

    if not ends_with_ext:
        if fname[-1] == ".":
            fname = fname[:-1]
        if ext[0] == ".":
            ext = ext[1:]
        return "%s.%s" % (fname, ext)

    return fname


def ensure_dir_exists(path, mode=0o755):
    """ensure that a directory exists

    If it doesn't exist, try to create it and protect against a race condition
    if another process is doing the same.

    The default permissions are 755, which differ from os.makedirs default of 777.
    """
    import errno

    if not os.path.exists(path):
        try:
            os.makedirs(path, mode=mode)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    elif not os.path.isdir(path):
        raise IOError("%r exists but is not a directory" % path)


def py_where(program, path=None):
    # From: http://stackoverflow.com/a/377028/548792
    winprog_exts = (".bat", "com", ".exe")

    def is_exec(fpath):
        return (
            osp.isfile(fpath)
            and os.access(fpath, os.X_OK)
            and (os.name != "nt" or fpath.lower()[-4:] in winprog_exts)
        )

    progs = []
    if not path:
        path = os.environ["PATH"]
    for folder in path.split(osp.pathsep):
        folder = folder.strip('"')
        if folder:
            exe_path = osp.join(folder, program)
            for f in [exe_path] + ["%s%s" % (exe_path, e) for e in winprog_exts]:
                if is_exec(f):
                    progs.append(f)
    return progs


def where(program):
    import subprocess

    try:
        res = subprocess.check_output('where "%s"' % program, universal_newlines=True)
        return res and [s.strip() for s in res.split("\n") if s.strip()]
    except subprocess.CalledProcessError:
        return []
    except:
        return py_where(program)


def which(program):
    res = where(program)
    return res[0] if res else None


def open_file_with_os(fpath):  # pragma: no cover
    # From http://stackoverflow.com/questions/434597/open-document-with-default-application-in-python
    #     and http://www.dwheeler.com/essays/open-files-urls.html
    import subprocess

    try:
        os.startfile(fpath)  # @UndefinedVariable
    except AttributeError:
        if sys.platform.startswith("darwin"):
            subprocess.call(("open", fpath))
        elif os.name == "posix":
            subprocess.call(("xdg-open", fpath))
    return


class LoggerWriter:
    """From http://plumberjack.blogspot.gr/2009/09/how-to-treat-logger-like-output-stream.html"""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, msg):
        if msg:
            line_endings = ["\r\n", "\n\r", "\n"]
            for le in line_endings:
                if msg.endswith(le):
                    msg = msg[: -len(le)]
            if msg:
                self.logger.log(self.level, msg)

    def flush(self):
        pass


if __name__ == "__main__":
    raise NotImplementedError
