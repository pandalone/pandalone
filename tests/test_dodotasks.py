#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, print_function, unicode_literals

import unittest
import doctest
import pandalone.utils

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(pandalone.utils))
    return tests