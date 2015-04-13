import doctest
import unittest

import pandalone.components
from pandalone.components import Assembly, FuncComponent


class TestDoctest(unittest.TestCase):

    def test_doctests(self):
        failure_count, test_count = doctest.testmod(
            pandalone.components, optionflags=doctest.NORMALIZE_WHITESPACE)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))

    def test_assembly(self):
        def cfunc_f1(comp, value_tree):
            comp.pinp().A
            comp.pout().B

        def cfunc_f2(comp, value_tree):
            comp.pinp().B
            comp.pout().C
        ass = Assembly(FuncComponent(cfunc) for cfunc in [cfunc_f1, cfunc_f2])
        ass._build()
        self.assertEqual(list(ass._iter_validations()), [])
        self.assertEqual(ass._inp, ['f1/A', 'f2/B'])
        self.assertEqual(ass._out, ['f1/B', 'f2/C'])

        pmods = {'f1': '/root', 'f2': '/root'}
        ass._build(pmods)
        self.assertEqual(
            sorted(ass._inp + ass._out), ['/root/A', '/root/B', '/root/B', '/root/C'])
