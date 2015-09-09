#! python
#-*- coding: utf-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from contextlib import contextmanager
from io import StringIO
import logging
import os
import re
import sys
from tempfile import mkdtemp
from textwrap import dedent
import unittest


log = logging.getLogger(__name__)


DEFAULT_LOG_LEVEL = logging.INFO


# TODO: Make it public
def _init_logging(module_name, loglevel=DEFAULT_LOG_LEVEL):
    logging.basicConfig(level=loglevel)
    logging.getLogger().setLevel(level=loglevel)

    log = logging.getLogger(module_name)
    log.trace = lambda *args, **kws: log.log(0, *args, **kws)

    return log

_xl_installed = None

##############
#  Compatibility
#
try:  # pragma: no cover
    assertRaisesRegex = unittest.TestCase.assertRaisesRegex
except:  # pragma: no cover
    assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


def check_excell_installed():
    """Checks once and returns `True` if Excel-app is installed in the system."""
    global _xl_installed
    if _xl_installed is None:
        try:
            from win32com.client import dynamic  # @UnresolvedImport
            dynamic.Dispatch('Excel.Application')
            _xl_installed = True
        except Exception:  # pragma: no cover
            _xl_installed = False

    return _xl_installed


class CustomAssertions(object):
    # Idea from: http://stackoverflow.com/a/15868615/548792

    def _raise_file_msg(self, msg, path, user_msg):
        msg = msg % os.path.abspath(path)
        if user_msg:
            msg = '%s: %s' % (msg, user_msg)
        raise AssertionError(msg)

    def assertFileExists(self, path, user_msg=None):
        if not os.path.lexists(path):
            self._raise_file_msg(
                "File does not exist in path '%s'!", path, user_msg)

    def assertFileNotExists(self, path, user_msg=None):
        if os.path.lexists(path):
            self._raise_file_msg("File exists in path '%s'!", path, user_msg)

    def assertNumpyEqual(self, x, y, msg=''):
        """
        Adapted from :func:`numpy.testing.utils.assert_array_compare()`.
        """
        import numpy as np
        from numpy import testing as npt

        def _isXXX(a, op, when_ex):
            res = []
            for i in a.flatten():
                try:
                    t = op(i)
                except TypeError:
                    t = when_ex
                res.append(t)
            return np.array(res, dtype=bool)

        def _isNaN(a):
            return _isXXX(a, np.isnan, True)

        def _isInf(a):
            return _isXXX(a, np.isinf, False)

        def _isPInf(a):
            return _isXXX(a, lambda x: x == np.Inf, False)

        def _isMInf(a):
            return _isXXX(a, lambda x: x == -np.Inf, False)

        cond = (x.shape == () and y.shape == ()) or x.shape == y.shape
        if not cond:
            msg = npt.build_err_msg([x, y],
                                    msg
                                    + '\n(shapes %s, %s mismatch)' % (x.shape,
                                                                      y.shape))
        if x.dtype != y.dtype:
            msg = npt.build_err_msg([x, y],
                                    msg
                                    + '\n(dtypes %s, %s mismatch)' % (x.dtype,
                                                                      y.dtype))
            raise AssertionError(msg)
        x_isnan, y_isnan = _isNaN(x), _isNaN(y)
        x_isinf, y_isinf = _isInf(x), _isInf(y)

        # Validate that the special values are in the same place
        if any(x_isnan) or any(y_isnan):
            npt.assert_array_equal(x_isnan, y_isnan, 'nan: ' + msg)
        if any(x_isinf) or any(y_isinf):
            # Check +inf and -inf separately, since they are different
            npt.assert_array_equal(_isPInf(x), _isPInf(y), '+inf: ' + msg)
            npt.assert_array_equal(_isMInf(x), _isMInf(y), '-inf: ' + msg)

        # Combine all the special values
        x_id, y_id = x_isnan, y_isnan
        x_id |= x_isinf
        y_id |= y_isinf

        # Only do the comparison if actual values are left
        if all(x_id):
            return

        if any(x_id):
            val = np.equal(x[~x_id], y[~y_id])
        else:
            val = np.equal(x, y)

        if isinstance(val, bool):
            cond = val
            reduced = [0]
        else:
            reduced = val.ravel()
            cond = reduced.all()
            reduced = reduced.tolist()

        if not cond:
            match = 100 - 100.0 * reduced.count(1) / len(reduced)
            msg = npt.build_err_msg([x, y],
                                    msg
                                    + '\n(mismatch %s%%)' % (match,))
            if not cond:
                raise AssertionError(msg)

    def assertStrippedStringsEqual(self, st, nd, msg=None, context_chars=30):
        regex = re.compile('\\s+', re.DOTALL)
        st1 = regex.sub('', st)
        nd1 = regex.sub('', nd)
        if st1 != nd1:
            err_i = len(os.path.commonprefix((st1, nd1)))
            s_slice = slice(max(0, err_i - context_chars),
                            err_i + context_chars)
            c1, c2 = st1[s_slice], nd1[s_slice]
            frmt = dedent("""\
            Stripped-strings differ at char %i (lens: 1st=%i, 2nd=%s)!
              --1st: %s
                     %sX
              --2nd: %s
              ==1st original: %s
              ==2nd original: %s
            ----%s
            """)
            spcs = ' ' * context_chars
            err_msg = frmt % (err_i, len(st1), len(nd1), c1, spcs, c2,
                              st, nd, (msg or ''))
            self.fail(err_msg)

try:
    from tempfile import TemporaryDirectory  # @UnusedImport
except ImportError:
    class TemporaryDirectory(object):

        """Create and return a temporary directory.  This has the same
        behavior as mkdtemp but can be used as a context manager.  For
        example:

            with TemporaryDirectory() as tmpdir:
                ...

        Upon exiting the context, the directory and everything contained
        in it are removed.

        From: http://stackoverflow.com/questions/19296146/tempfile-temporarydirectory-context-manager-in-python-2-7
              http://stackoverflow.com/a/19299884/548792
        """

        def __init__(self, suffix="", prefix="tmp", dir=None):
            self._closed = False
            self.name = None  # Handle mkdtemp raising an exception
            self.name = mkdtemp(suffix, prefix, dir)

        def __repr__(self):
            return "<{} {!r}>".format(self.__class__.__name__, self.name)

        def __enter__(self):
            return self.name

        def cleanup(self, _warn=False):
            if self.name and not self._closed:
                try:
                    self._rmtree(self.name)
                except (TypeError, AttributeError) as ex:
                    # Issue #10188: Emit a warning on stderr
                    # if the directory could not be cleaned
                    # up due to missing globals
                    if "None" not in str(ex):
                        raise
                    log.warning("ERROR: %s while cleaning up %s", ex, self)
                    return
                self._closed = True
                log.warning("Implicitly cleaning up %s", self)

        def __exit__(self, exc, value, tb):
            self.cleanup()

        def __del__(self):
            # Issue a ResourceWarning if implicit cleanup needed
            self.cleanup(_warn=True)

        # XXX (ncoghlan): The following code attempts to make
        # this class tolerant of the module nulling out process
        # that happens during CPython interpreter shutdown
        # Alas, it doesn't actually manage it. See issue #10188
        _listdir = staticmethod(os.listdir)
        _path_join = staticmethod(os.path.join)
        _isdir = staticmethod(os.path.isdir)
        _islink = staticmethod(os.path.islink)
        _remove = staticmethod(os.remove)
        _rmdir = staticmethod(os.rmdir)

        def _rmtree(self, path):
            # Essentially a stripped down version of shutil.rmtree.  We can't
            # use globals because they may be None'ed out at shutdown.
            for name in self._listdir(path):
                fullname = self._path_join(path, name)
                try:
                    isdir = self._isdir(
                        fullname) and not self._islink(fullname)
                except OSError:
                    isdir = False
                if isdir:
                    self._rmtree(fullname)
                else:
                    try:
                        self._remove(fullname)
                    except OSError:
                        pass
            try:
                self._rmdir(path)
            except OSError:
                pass


@contextmanager
def capture(command, *args, **kwargs):
    # Unused
    out, sys.stdout = sys.stdout, StringIO()
    err, sys.stderr = sys.stderr, StringIO()
    try:
        command(*args, **kwargs)
        sys.stdout.seek(0)
        yield (sys.stdout.getvalue(), sys.stderr.getvalue())
    finally:
        sys.stdout = out
        sys.stderr = err


@contextmanager
def chdir(dirname=None):
    curdir = os.getcwd()
    try:
        os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)


def xw_close_workbook(wb):
    try:
        wb.close()
    except Exception:
        log.warning('Minor failure while closing Workbook!', exc_info=True)


@contextmanager
def xw_Workbook(*args, **kws):
    import xlwings

    wb = xlwings.Workbook(*args, **kws)
    try:
        # app = wb.application TODO: Upgrade xlwings
        yield wb
    finally:
        xw_close_workbook(wb)
    # app.quit()
