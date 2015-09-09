#! python
#-*- coding: utf-8 -*-
#
# Copyright 2013-2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, unicode_literals

import os
import re
import sys

from future.moves.collections import Sequence  # @UnresolvedImport
from past.types import basestring


__commit__ = ""

# Python-2 compatibility
#
try:  # pragma: no cover
    FileNotFoundError
except NameError:  # pragma: no cover
    FileNotFoundError = IOError  # @ReservedAssignment
else:  # pragma: no cover
    FileNotFoundError = OSError  # @ReservedAssignment


def fullmatch_py2(regex, string, flags=0):
    # NOTE: re.match("(?:" + regex + r")\Z", string, flags=flags)
    m = re.match(regex, string, flags=flags)
    if m and m.span()[1] == len(string):
        return m
try:  # pragma: no cover
    from re import fullmatch  # @UnusedImport
except ImportError:  # pragma: no cover
    fullmatch = fullmatch_py2

##############
#  Utilities
#


def str2bool(v):
    """
    Utility for parsing cmd-line args.

    :param str v: any of (case insensitive): yes/no, true/false, on/off

    Example::

        >>> str2bool('ON')
        True
        >>> str2bool('no')
        False

        >>> str2bool('')
        False
        >>> str2bool('  ')
        False

        >>> str2bool(0)
        Traceback (most recent call last):
        ValueError: Invalid str-boolean(0) due to: 'int' object has no attribute 'strip'
        >>> str2bool(None)
        Traceback (most recent call last):
        ValueError: Invalid str-boolean(None) due to: 'NoneType' object has no attribute 'strip'

    """
    try:
        vv = v.strip().lower()
        if (vv in ("yes", "true", "on")):
            return True
        if (vv in ("no", "false", "off")):
            return False
        return bool(vv)
    except Exception as ex:
        msg = 'Invalid str-boolean(%s) due to: %s'
        raise ValueError(msg % (v, ex))


def is_travis():  # pragma: no cover
    return 'TRAVIS' in os.environ


def as_list(o):
    if isinstance(o, Sequence) and not isinstance(o, basestring):
        o = list(o)
    else:
        o = [o]
    return o


def generate_filenames(filename):
    f, e = os.path.splitext(filename)
    yield filename
    i = 1
    while True:
        yield '%s%i%s' % (f, i, e)
        i += 1


def make_unique_filename(fname, filegen=generate_filenames):
    fname_genor = generate_filenames(fname)
    fname = next(fname_genor)
    while os.path.exists(fname):
        fname = next(fname_genor)
    return fname


def ensure_file_ext(fname, ext):
    """
    :param str ext: extension with dot(.)

    >>> assert ensure_file_ext('foo', '.bar')     == 'foo.bar'
    >>> assert ensure_file_ext('foo.bar', '.bar') == 'foo.bar'
    >>> assert ensure_file_ext('foo.', '.bar')    == 'foo..bar'
    >>> assert ensure_file_ext('foo.', 'bar')    == 'foo.bar'

    """
    _, e = os.path.splitext(fname)
    if e != ext:
        return '%s%s' % (fname, ext)
    return fname


def open_file_with_os(fpath):  # pragma: no cover
    # From http://stackoverflow.com/questions/434597/open-document-with-default-application-in-python
    #     and http://www.dwheeler.com/essays/open-files-urls.html
    import subprocess
    try:
        os.startfile(fpath)  # @UndefinedVariable
    except AttributeError:
        if sys.platform.startswith('darwin'):
            subprocess.call(('open', fpath))
        elif os.name == 'posix':
            subprocess.call(('xdg-open', fpath))
    return


class LoggerWriter:
    """From http://plumberjack.blogspot.gr/2009/09/how-to-treat-logger-like-output-stream.html"""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, msg):
        if msg:
            line_endings = ['\r\n', '\n\r', '\n']
            for le in line_endings:
                if msg.endswith(le):
                    msg = msg[:-len(le)]
            if msg:
                self.logger.log(self.level, msg)

    def flush(self):
        pass

if __name__ == '__main__':
    raise NotImplementedError
