#! python
#-*- coding: utf-8 -*-
#
# Copyright 2013-2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, unicode_literals

import doctest
import re
import sre_constants
import unittest

import functools as ft
import numpy.testing as npt
from pandalone.mappings import (
    df_as_pmods_tuples, Pmod,
    pmods_from_tuples)
import pandalone.mappings
import pandas as pd
from tests.test_utils import _init_logging


log = _init_logging(__name__)


def pmod2regexstrs(pmod):
    if pmod._regxs:
        return [r.pattern for r in list(pmod._regxs.keys())]


class TestDoctest(unittest.TestCase):

    def test_doctests(self):
        failure_count, test_count = doctest.testmod(
            pandalone.mappings,
            optionflags=doctest.NORMALIZE_WHITESPACE)  # | doctest.ELLIPSIS)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))


class TestPmod(unittest.TestCase):

    def setUp(self):
        self.assertPmod_class_attributes_not_modified()

    def tearDown(self):
        self.assertPmod_class_attributes_not_modified()

    def assertPmod_class_attributes_not_modified(self):
        self.assertEqual(
            Pmod.__init__.__defaults__, (None, {}, {}))  # @UndefinedVariable

    def test_merge_name(self):
        pm1 = Pmod(_alias='pm1')
        pm2 = Pmod(_alias='pm2')
        pm = pm1._merge(pm2)
        self.assertEqual(pm._alias, 'pm2')
        self.assertEqual(pm1._alias, 'pm1')
        self.assertEqual(pm2._alias, 'pm2')
        pm = pm2._merge(pm1)
        self.assertEqual(pm._alias, 'pm1')
        self.assertEqual(pm1._alias, 'pm1')
        self.assertEqual(pm2._alias, 'pm2')

        pm1 = Pmod()
        pm2 = Pmod(_alias='pm2')
        pm = pm1._merge(pm2)
        self.assertEqual(pm._alias, 'pm2')
        self.assertEqual(pm1._alias, None)
        self.assertEqual(pm2._alias, 'pm2')
        pm = pm2._merge(pm1)
        self.assertEqual(pm._alias, 'pm2')
        self.assertEqual(pm1._alias, None)
        self.assertEqual(pm2._alias, 'pm2')

    def test_merge_name_recurse(self):
        pm1 = Pmod(_alias='pm1', _steps={'a': Pmod(_alias='R1')})
        pm2 = Pmod(_alias='pm2', _steps={'a': Pmod(_alias='R2'),
                                         'b': Pmod(_alias='R22')})
        pm = pm1._merge(pm2)
        self.assertEqual(pm._steps['a']._alias, 'R2')
        self.assertEqual(pm1._steps['a']._alias, 'R1')
        self.assertEqual(pm2._steps['a']._alias, 'R2')
        self.assertEqual(len(pm1._steps), 1)
        self.assertEqual(len(pm2._steps), 2)
        pm = pm2._merge(pm1)
        self.assertEqual(pm._steps['a']._alias, 'R1')
        self.assertEqual(pm1._steps['a']._alias, 'R1')
        self.assertEqual(pm2._steps['a']._alias, 'R2')
        self.assertEqual(len(pm1._steps), 1)
        self.assertEqual(len(pm2._steps), 2)

    def test_merge_steps(self):
        pm1 = Pmod(_alias='pm1', _steps={'a': Pmod(_alias='A'),
                                         'c': Pmod(_alias='C')})
        pm2 = Pmod(_alias='pm2', _steps={'b': Pmod(_alias='B'),
                                         'a': Pmod(_alias='AA'),
                                         'd': Pmod(_alias='DD'),
                                         })
        pm = pm1._merge(pm2)
        self.assertEqual(sorted(pm._steps.keys()), list('abcd'))
        pm = pm2._merge(pm1)
        self.assertEqual(sorted(pm._steps.keys()), list('abcd'))
        self.assertEqual(len(pm1._steps), 2)
        self.assertEqual(len(pm2._steps), 3)

        pm1 = Pmod(_steps={'a': Pmod(_alias='A'),
                           'c': Pmod(_alias='C')})
        pm2 = Pmod(_alias='pm2', _steps={'b': Pmod(_alias='B'),
                                         'a': Pmod(_alias='AA'),
                                         'd': Pmod(_alias='DD'),
                                         })
        pm = pm1._merge(pm2)
        self.assertEqual(sorted(pm._steps.keys()), list('abcd'))
        self.assertEqual(pm._steps['a']._alias, 'AA')
        self.assertEqual(len(pm1._steps), 2)
        self.assertEqual(len(pm2._steps), 3)
        self.assertEqual(pm._regxs, {})
        self.assertEqual(pm1._regxs, {})
        self.assertEqual(pm2._regxs, {})

        pm = pm2._merge(pm1)
        self.assertEqual(sorted(pm._steps.keys()), list('abcd'))
        self.assertEqual(pm._steps['a']._alias, 'A')
        self.assertEqual(len(pm1._steps), 2)
        self.assertEqual(len(pm2._steps), 3)
        self.assertEqual(pm._regxs, {})
        self.assertEqual(pm1._regxs, {})
        self.assertEqual(pm2._regxs, {})

    def test_merge_regxs(self):
        pm1 = Pmod(_alias='pm1', _regxs=[
            ('e', Pmod(_alias='E')),
            ('a', Pmod(_alias='A')),
            ('c', Pmod(_alias='C'))])
        pm2 = Pmod(_alias='pm2', _regxs=[
            ('b', Pmod(_alias='B')),
            ('a', Pmod(_alias='AA')),
            ('d', Pmod(_alias='DD')),
        ])

        pm = pm1._merge(pm2)
        self.assertSequenceEqual(pmod2regexstrs(pm), list('ecbad'))
        self.assertEqual(pm._regxs[re.compile('a')]._alias, 'AA')
        self.assertSequenceEqual(pmod2regexstrs(pm1), list('eac'))
        self.assertSequenceEqual(pmod2regexstrs(pm2), list('bad'))
        self.assertEqual(pm._steps, {})
        self.assertEqual(pm1._steps, {})
        self.assertEqual(pm2._steps, {})

        pm = pm2._merge(pm1)
        self.assertSequenceEqual(pmod2regexstrs(pm), list('bdeac'))
        self.assertEqual(pm._regxs[re.compile('a')]._alias, 'A')
        self.assertSequenceEqual(pmod2regexstrs(pm1), list('eac'))
        self.assertSequenceEqual(pmod2regexstrs(pm2), list('bad'))
        self.assertEqual(pm._steps, {})
        self.assertEqual(pm1._steps, {})
        self.assertEqual(pm2._steps, {})

    def test_merge_all(self):
        """Check merge_all behavior, but function has been inlined in Pmod. """
        def merge_all(pmods):
            return ft.reduce(Pmod._merge, pmods)

        pm1 = Pmod(_steps={'a': Pmod(_alias='A')})
        pm2 = Pmod(_alias='pm2',
                   _regxs=[
                       ('b', Pmod(_alias='BB')),
                       ('a', Pmod(_alias='AA'))
                   ])
        pm3 = Pmod(_alias='PM3',
                   _steps={'c': Pmod(_alias='CCC'),
                           'b': Pmod(_alias='BBB'), },
                   _regxs=[
                       ('b', Pmod(_alias='AAA')),
                       ('a', Pmod(_alias='BBB')),
                       ('c', Pmod(_alias='CCC')),
                   ])
        pm = merge_all([pm1, pm2, pm3])

        self.assertSetEqual(set(pm._steps.keys()), set(list('abc')))
        self.assertSetEqual(set(pm1._steps.keys()), set(list('a')))
        self.assertEqual(pm2._steps, {})
        self.assertSetEqual(set(pm3._steps.keys()), set(list('bc')))

        self.assertEqual(pmod2regexstrs(pm), list('bac'))
        self.assertEqual(pm1._regxs, {})
        self.assertEqual(pmod2regexstrs(pm2), list('ba'))
        self.assertEqual(pmod2regexstrs(pm3), list('bac'))

    def test_descend_None(self):
        pm = Pmod()
        self.assert_descend_stops(pm.descend('a'))

        pm = Pmod(_alias='a')
        self.assert_descend_stops(pm.descend('a'))

        pm = Pmod(_steps={'a': None})
        self.assert_descend_stops(pm.descend('a'))

    def test_alias_None(self):
        pm = Pmod()
        self.assertIsNone(pm.alias('a'))

        pm = Pmod(_alias='a')
        self.assertIsNone(pm.alias('a'))

        pm = Pmod(_steps={'a': None})
        self.assertIsNone(pm.alias('a'))

    def _build_pmod_c1r2(self):
        return Pmod(
            _steps={'a': Pmod(_alias='A')},
            _regxs=[('a(\w*)', Pmod('AWord')),
                    ('a(\d*)', Pmod(_alias='A_\\1')), ])

    def assert_descend_stops(self, cpmod, msg=None):
        self.assertEqual(cpmod, (None, None), msg)

    def test_descend_BAD(self):
        pm = self._build_pmod_c1r2()
        self.assert_descend_stops(pm.descend('BAD'))
        self.assert_descend_stops(pm.descend('a-'))
        self.assert_descend_stops(pm.descend('a$'))

    def test_alias_BAD(self):
        pm = self._build_pmod_c1r2()
        self.assertIsNone(pm.alias('BAD'))
        self.assertIsNone(pm.alias('a-'))
        self.assertIsNone(pm.alias('a$'))

    def test_descend(self):
        pm = Pmod(
            _steps={'a':
                    Pmod(_alias='A', _steps={1: 11})},
            _regxs=[
                ('a\w*', Pmod(_alias='AWord', _steps={2: 22})),
                ('a\d*', Pmod(_alias='ADigit', _steps={3: 33})),
            ])

        # All children and regexps match.
        self.assertDictEqual(pm.descend('a')[0]._steps, {1: 11, 2: 22, 3: 33})

        # Only 'a\w*' matches.
        self.assertDictEqual(pm.descend('aa')[0]._steps, {2: 22})

        # Both regexps matches.
        self.assertDictEqual(pm.descend('a1')[0]._steps, {2: 22, 3: 33})

    def test_alias(self):
        pm = self._build_pmod_c1r2()

        self.assertEqual(pm.alias('a'), 'A')
        self.assertEqual(pm.alias('abc'), 'AWord')
        self.assertEqual(pm.alias('a12'), 'A_12')
        self.assertIsNone(pm.alias('BAD'))

    def test_descend_group_zero_(self):
        pm = Pmod(_regxs=[('abc', Pmod(_alias='A\\g<0>'))])
        self.assertEqual(pm.descend('abc')[1], 'Aabc')

        pm = Pmod(_regxs=[('a(.*)', Pmod(_alias='A\\g<0>'))])
        self.assertEqual(pm.descend('abc')[1], 'Aabc')

    def test_alias_group_zero_(self):
        pm = Pmod(_regxs=[('abc', Pmod(_alias='A\\g<0>'))])
        self.assertEqual(pm.alias('abc'), 'Aabc')

        pm = Pmod(_regxs=[('a(.*)', Pmod(_alias='A\\g<0>'))])
        self.assertEqual(pm.alias('abc'), 'Aabc')

    def test_alias_invalid_group_ref_(self):
        pm = Pmod(_regxs=[
            ('a', Pmod(_alias='A\\1')),
            ('b(.*)', Pmod(_alias='B\\2')),
        ])

        with self.assertRaises(sre_constants.error):
            pm.alias('a')

        with self.assertRaises(sre_constants.error):
            pm.alias('bach')

    def test_map_path_rootUnmapped_slashPath(self):
        pm = Pmod()
        self.assertEqual(pm.map_path('/'), '/')

        pm = self._build_pmod_c1r2()
        self.assertEqual(pm.map_path('/'), '/')

    def test_map_path_rootUnmapped_emptyPath(self):
        pm = Pmod()
        self.assertEqual(pm.map_path(''), '')

        pm = self._build_pmod_c1r2()
        self.assertEqual(pm.map_path(''), '')

    def test_map_path_rootMapped(self):
        pm = Pmod(_alias='root')
        self.assertEqual(pm.map_path(''), '')
        self.assertEqual(pm.map_path('/'), '/root')
        self.assertEqual(pm.map_path('/a'), '/root/a')

        pm = self._build_pmod_c1r2()
        pm._alias = 'root'
        self.assertEqual(pm.map_path(''), '')
        self.assertEqual(pm.map_path('/'), '/root')
        self.assertEqual(pm.map_path('/a'), '/root/A')
        self.assertEqual(pm.map_path('/a_blah'), '/root/AWord')
        self.assertEqual(pm.map_path('/a0'), '/root/A_0')

        pm = pmods_from_tuples([
            ('', 'New/Root'),
            ('/a/b', 'B'),
        ])
        self.assertEqual(pm.map_path('/a/b'), '/New/Root/a/B')

    def test_map_path_missed(self):
        pm = self._build_pmod_c1r2()

        self.assertEqual(pm.map_path('/c'), '/c')
        self.assertEqual(pm.map_path('/c/a'), '/c/a')
        self.assertEqual(pm.map_path('/a$/a4'), '/a$/a4')

    def test_map_path_simple(self):
        pm = self._build_pmod_c1r2()

        self.assertEqual(pm.map_path('/a'), '/A')
        self.assertEqual(pm.map_path('/a/b'), '/A/b')

    def test_map_path_regex(self):
        pm = self._build_pmod_c1r2()

        self.assertEqual(pm.map_path('/a'), '/A')
        self.assertEqual(pm.map_path('/a_wo/b'), '/AWord/b')
        self.assertEqual(pm.map_path('/a12/b'), '/A_12/b')

    def test_convert_df_empty(self):
        df_orig = pd.DataFrame([])

        df = df_orig.copy()
        pmods = df_as_pmods_tuples(df)
        self.assertEqual(pmods, [])
        npt.assert_array_equal(df, df_orig)

        df = df_orig.copy()
        pmods = df_as_pmods_tuples(df, col_from='A', col_to='B')
        self.assertEqual(pmods, [])
        npt.assert_array_equal(df, df_orig)

    def test_convert_df_no_colnames(self):
        df_orig = pd.DataFrame([['/a/b', '/A/B']])

        df = df_orig.copy()
        pmods = df_as_pmods_tuples(df)
        self.assertEqual(tuple(pmods[0]), ('/a/b', '/A/B'))
        npt.assert_array_equal(df, df_orig)

        df = df_orig.copy()
        pmods = df_as_pmods_tuples(df, col_from='A', col_to='B')
        self.assertEqual(tuple(pmods[0]), ('/a/b', '/A/B'))
        npt.assert_array_equal(df, df_orig)

    def test_build_pmods_steps_no_regxs(self):
        pmods_tuples = [
            ('/a', 'A1/A2'),
            ('/a/b', 'B'),
        ]
        pmods = pmods_from_tuples(pmods_tuples)
        # pmod({'a': pmod('A1/A2', {'b': pmod('B')})})

        self.assertFalse(bool(pmods._regxs))
        self.assertSetEqual(set(pmods._steps), {'a'})
        self.assertEqual(pmods._steps['a']._alias, 'A1/A2')
        self.assertEqual(pmods._steps['a']._steps['b']._alias, 'B')

    def test_build_pmods_regex_no_steps(self):
        pmods_tuples = [
            ('/a*', 'A1/A2'),
            ('/a[1]?/b?', 'B'),
        ]
        pmods = pmods_from_tuples(pmods_tuples)

        self.assertFalse(bool(pmods._steps))
        self.assertEqual(pmod2regexstrs(pmods), ['a*', 'a[1]?'])

    def test_pmods_from_tuples_repr(self):
        pmods_tuples = [
            ('/a', 'A'),
            ('/a\w*', 'A1/A2'),
            ('/a\d*/b?', 'D/D'),
        ]
        pmods = pmods_from_tuples(pmods_tuples)

        s = [r"pmod({'a': pmod('A')}, OrderedDict([(re.compile('a\\w*'), "
             r"pmod('A1/A2')), (re.compile('a\\d*'), "
             r"pmod(OrderedDict([(re.compile('b?'), pmod('D/D'))])))]))"
             ]
        self.assertEqual(str(pmods), "".join(s))

    def test_pmods_from_tuples_rootMapping(self):
        self.assertEqual(pmods_from_tuples([('', 'A/B')]),
                         Pmod(_alias='A/B'))
        self.assertEqual(pmods_from_tuples([('/', 'A/B')]),
                         Pmod(_steps={'': Pmod(_alias='A/B')}))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
