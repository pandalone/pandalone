#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""Delegates to *pandalone* `doit` tasks."""

import pndltasks
from doit.cmd_base import ModuleTaskLoader
from doit.doit_cmd import DoitMain
import os
import sys


def main(argv=None):
    myname = os.path.basename(sys.argv[0])  # @UnusedVariable
    mydir = os.path.dirname(__file__)  # @UnusedVariable
    if argv is None:
        argv = sys.argv[1:]
    commander = DoitMain(ModuleTaskLoader(pndltasks))
    commander.run(argv)


if __name__ == "__main__":
    main()
