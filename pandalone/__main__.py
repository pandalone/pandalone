#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""Delegates to *pandalone* `doit` tasks."""

from doit.cmd_base import ModuleTaskLoader
from doit.doit_cmd import DoitMain
from os import path
from pandalone import pndlcmd
import sys


def main(argv=None):
    myname = path.basename(sys.argv[0])  # @UnusedVariable
    mydir = path.dirname(__file__)  # @UnusedVariable
    if argv is None:
        argv = sys.argv[1:]
    opt_vals = {}  # 'dep_file': path.abspath(path.join(mydir, '.doit.db'))}
    commander = DoitMain(ModuleTaskLoader(pndlcmd),
                         extra_config={'GLOBAL': opt_vals})
    commander.run(argv)


if __name__ == "__main__":
    main()
