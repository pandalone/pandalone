import unittest
import doctest
import pandalone.utils

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(pandalone.utils))
    return tests