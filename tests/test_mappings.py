#! python
#-*- coding: utf-8 -*-
#
# Copyright 2013-2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, unicode_literals

import doctest
import json
from pandalone.mappings import (
    df_as_pmods_tuples, Pmod,
    pmods_from_tuples, Pstep, _append_path)
import pandalone.mappings
import re
import sre_constants
from tests.test_utils import _init_logging
import unittest

import functools as ft
import numpy.testing as npt
import pandas as pd


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


class TestJoinPaths(unittest.TestCase):

    def test_append_paths_dotdot_on_empty(self):
        steps = []
        self.assertEqual(_append_path(steps, '..'), [])

    def test_append_paths_dotdot_on_rooted(self):
        steps = ['']
        self.assertEqual(_append_path(steps, '..'), [''])

    def test_append_paths_slash(self):
        steps = ['', 'a']
        self.assertEqual(_append_path(steps, '/'), ['', ''])

        steps = ['']
        self.assertEqual(_append_path(steps, '/'), ['', ''])


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
        self.assertEqual(pm.map_path('/'), '/')
        self.assertEqual(pm.map_path(''), '')
        self.assertEqual(pm.map_path('/a'), '/root/a')

        pm = self._build_pmod_c1r2()
        pm._alias = 'root'
        self.assertEqual(pm.map_path(''), '')
        self.assertEqual(pm.map_path('/'), '/')
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


class TestPstep(unittest.TestCase):

    def PMODS(self):
        return pmods_from_tuples([
            ('',            'root'),
            ('/a',          'b'),
            ('/abc',        'BAR'),
            ('/abc/def',    'DEF'),
            ('/abc/def/123', '234'),
            ('/for',        '/sub/path'),
        ])

    def test_equality(self):
        p = Pstep()
        self.assertEqual(str(p), '')
        self.assertEqual(p, Pstep(''))
        self.assertEquals(str(p['']), str(p))

        self.assertEqual(str(p.a), 'a')
        self.assertEqual(p.a, Pstep('a'))
        self.assertEquals(str(p['a']), str(p.a))

        n = 'foo'
        p = Pstep(n)
        self.assertEqual(str(p), n)
        self.assertEquals(p, Pstep(n))
        self.assertEquals(str(p.foo), str(p))

        n = '/foo'
        p = Pstep(n)
        self.assertEqual(str(p), n)
        self.assertEquals(p, Pstep(n))
        p['12']
        self.assertNotEquals(str(p), '12')

    def test_buildtree_valid_ops(self):
        p = Pstep()
        p.abc
        p.abc['def']
        p['abc'].defg
        p.n123
        p['321']
        p._some_hidden = 12
        p.a.b.c['123'][4]

    def test_buildtree_invalid_ops(self):

        p = Pstep()

        def f1(p):
            p.abc = 1

        def f2(p):
            p['a'] = 1

        def f3(p):
            p['a']['b'] = 1

        def f4(p):
            p['a'].c = 1

        def f5(p):
            p['_hid'] = 1

        for f in [f1, f2, f3, f4, f5]:
            p = Pstep()
            with self.assertRaises(AssertionError, msg=f):
                f(p),

    def test_paths_empty1ststep(self):
        p = Pstep()
        self.assertListEqual(sorted(p._paths), sorted(['']))

        p.a
        self.assertListEqual(sorted(p._paths), sorted(['/a']))

        p = Pstep('')
        self.assertListEqual(sorted(p._paths), sorted(['']))

        p = Pstep('')
        p.a
        self.assertListEqual(sorted(p._paths), sorted(['/a']))

    def test_paths(self):
        p = Pstep('r')
        self.assertListEqual(sorted(p._paths), sorted(['r']))

        p.a
        self.assertListEqual(sorted(p._paths), sorted(['r/a']))

    def test_paths_multi_empty1ststep(self):
        p = Pstep()
        p.abc
        p.abc['def']
        p['abc'].defg
        p.n123
        p['321']
        p._some_hidden = 12
        exp = [
            '//abc/def',
            '//abc/defg',
            '//321',
            '//n123',
        ]
        self.assertListEqual(sorted(p._paths), sorted(exp))
        self.assertEqual(
            Pstep._append_children.__defaults__[0], [])  # @UndefinedVariable

    def test_paths_multi_nonempty1ststep(self):
        p = Pstep('r')
        p.abc
        p.abc['def']
        p['abc'].defg
        p.n123
        p['321']
        p._some_hidden = 12
        exp = [
            '/r/abc/def',
            '/r/abc/defg',
            '/r/321',
            '/r/n123',
        ]
        self.assertListEqual(sorted(p._paths), sorted(exp))
        self.assertEqual(
            Pstep._append_children.__defaults__[0], [])  # @UndefinedVariable

    ### DOT ###

    def test_paths_dot_atroot_empty1ststep(self):
        p = Pstep()
        p['.']
        self.assertListEqual(sorted(p._paths), sorted(['']))

    def test_paths_dot_empty1ststep(self):
        p = Pstep()
        p.a['.']
        self.assertListEqual(sorted(p._paths), sorted(['/a']))

    def test_paths_dot_atroot(self):
        p = Pstep('r')
        p['.']
        self.assertListEqual(sorted(p._paths), sorted(['r']))

    def test_paths_dot(self):
        p = Pstep('r')
        p.a['.']
        self.assertListEqual(sorted(p._paths), sorted(['r/a']))

    def test_paths_dot_multi_empty1ststep(self):
        p = Pstep('')
        p['.'].a.b
        p.c['.'].d
        self.assertListEqual(sorted(p._paths), sorted(['/a/b', '/c/d']))

    def test_paths_dot_multi(self):
        p = Pstep('r')
        p['.'].a.b
        p.c['.'].d
        self.assertListEqual(sorted(p._paths), sorted(['r/a/b', 'r/c/d']))

    ### DOTDOT ###
    # NOT TO BE USED DIRECTLY
    # BC HIDS PATHS,
    # but testing for mapping.

    def test_paths_dotdot_atroot_empty1ststep(self):
        p = Pstep()
        p['..'].a
        self.assertListEqual(sorted(p._paths), sorted(['/a']))

        p = Pstep()
        p['..'].a
        p['..'].b
        self.assertListEqual(sorted(p._paths), sorted(['/a', '/b']))

    def test_paths_dotdot_empty1ststep(self):
        p = Pstep()
        p.a['..'].b
        self.assertListEqual(sorted(p._paths), sorted(['/b']))

        p = Pstep()
        p.a['..'].b
        p.a.c
        self.assertListEqual(sorted(p._paths), sorted(['/a/c', '/b']))

    def test_paths_dotdot_atroot(self):
        p = Pstep('r')
        p['..'].a
        self.assertListEqual(sorted(p._paths), sorted(['a']))

    def test_paths_dotdot(self):
        p = Pstep('r')
        p.a['..'].b
        self.assertListEqual(sorted(p._paths), sorted(['r/b']))

    def test_paths_dotdot_multi_empty1ststep(self):
        p = Pstep('')
        p['..'].a.b
        p.c['..'].d
        self.assertListEqual(sorted(p._paths), sorted(['/a/b', '/d']))

    def test_paths_dotdot_multi(self):
        p = Pstep('r')
        p['..'].a.b
        p.c['..'].d
        self.assertListEqual(sorted(p._paths), sorted(['a/b', 'r/d']))

    ### PMODS ###

    def test_pmods_miss(self):
        p = Pstep(_pmod=pmods_from_tuples([('/MISS', 'BOO')]))
        self.assertEquals(p, '', (p, p._paths))
        p = Pstep('foo', pmods_from_tuples([('/MISS', 'BOO')]))
        self.assertEquals(p, 'foo', (p, p._paths))

    def test_pmods_emptyroot_rootunmapped(self):
        p = Pstep(_pmod=pmods_from_tuples([('/a', 'bar')]))
        p.a
        self.assertEquals(p, '', (p, p._paths))
        self.assertEquals(p.a, 'bar', (p, p._paths))
        self.assertEquals(sorted(p._paths), ['//bar'], (p, sorted(p._paths)))

        self.assertEquals(p.a.b, 'b')
        self.assertEquals(sorted(p._paths), ['//bar/b'],
                          (p, sorted(p._paths)))

        self.assertEquals(p.c, 'c')
        self.assertEquals(sorted(p._paths), ['//bar/b', '//c'],
                          (p, sorted(p._paths)))

        self.assertEquals(p.a.f, 'f')
        self.assertEquals(sorted(p._paths),
                          ['//bar/b', '//bar/f', '//c'],
                          (p, sorted(p._paths)))

    def test_pmods_nonempty1ststep_rootunmapped(self):
        p = Pstep('root', _pmod=pmods_from_tuples([('/a', 'bar')]))
        p.a
        self.assertEquals(p, 'root', (p, p._paths))
        self.assertEquals(p.a, 'bar', (p, p._paths))
        self.assertEquals(
            sorted(p._paths), ['/root/bar'], (p, sorted(p._paths)))

        self.assertEquals(p.a.b, 'b')
        self.assertEquals(sorted(p._paths), ['/root/bar/b'],
                          (p, sorted(p._paths)))

        self.assertEquals(p.c, 'c')
        self.assertEquals(sorted(p._paths), ['/root/bar/b', '/root/c'],
                          (p, sorted(p._paths)))

        self.assertEquals(p.a.f, 'f')
        self.assertEquals(sorted(p._paths),
                          ['/root/bar/b', '/root/bar/f', '/root/c'],
                          (p, sorted(p._paths)))

    def test_pmods_emptyroot_rootmapped(self):
        p = Pstep(_pmod=pmods_from_tuples([('', 'root'), ('/a', 'bar')]))
        p.a
        self.assertEquals(p, 'root', (p, sorted(p._paths)))
        self.assertEquals(p.a, 'bar', (p, sorted(p._paths)))
        self.assertEquals(sorted(p._paths), ['/root/bar'],
                          (p, sorted(p._paths)))

        self.assertEquals(p.a.b, 'b')
        self.assertEquals(sorted(p._paths), ['/root/bar/b'],
                          (p, sorted(p._paths)))

        self.assertEquals(p.c, 'c')
        self.assertEquals(sorted(p._paths), ['/root/bar/b', '/root/c'],
                          (p, sorted(p._paths)))

        self.assertEquals(p.a.f, 'f')
        self.assertEquals(sorted(p._paths),
                          ['/root/bar/b', '/root/bar/f', '/root/c'],
                          (p, sorted(p._paths)))

    def test_pmods_nonemptyroot_rootmapped(self):
        p = Pstep(
            'root', _pmod=pmods_from_tuples([('', 'root'), ('/a', 'bar')]))
        p.a
        self.assertEquals(p, 'root', (p, sorted(p._paths)))
        self.assertEquals(p.a, 'bar', (p, sorted(p._paths)))
        self.assertEquals(
            sorted(p._paths), ['/root/bar'], (p, sorted(p._paths)))

        self.assertEquals(p.a.b, 'b')
        self.assertEquals(sorted(p._paths), ['/root/bar/b'],
                          (p, sorted(p._paths)))

        self.assertEquals(p.c, 'c')
        self.assertEquals(sorted(p._paths), ['/root/bar/b', '/root/c'],
                          (p, sorted(p._paths)))

        self.assertEquals(p.a.f, 'f')
        self.assertEquals(sorted(p._paths),
                          ['/root/bar/b', '/root/bar/f', '/root/c'],
                          (p, sorted(p._paths)))

    def _build_psteps(self, root='', pmods=None):
        p = Pstep(root, _pmod=pmods)
        p.a.b.c
        p.a.b.d
        p.a.c
        p.abc['def']
        p.n123
        p.a.n123
        p.cc[123]['123']
        p.cc[123].abc
        p['321']
        p['']

        return p

    def _assert_pstep_pmods_with_map_paths(self, root, pmods):
        ps1 = self._build_psteps(root, pmods=pmods)
        ps2 = self._build_psteps(root)

        self.assertEqual(sorted(ps1._paths),
                         sorted(pmods.map_paths(ps2._paths)))

    def test_pstep_vs_pmods_maproot(self):
        """Check ``pmod.map_path ()`` is equal with ``pstep(_pmod=pmod)``"""

        pmods = pmods_from_tuples([
            ('', 'AA'),
        ])
        self._assert_pstep_pmods_with_map_paths('P', pmods)

    def test_pstep_vs_pmods_mapslash(self):
        """Check ``pmod.map_path ()`` is equal with ``pstep(_pmod=pmod)``"""

        pmods = pmods_from_tuples([
            ('/', 'AA'),
        ])
        self._assert_pstep_pmods_with_map_paths('P', pmods)

    def test_pstep_vs_pmods_mapstep(self):
        """Check ``pmod.map_path ()`` is equal with ``pstep(_pmod=pmod)``"""

        pmods = pmods_from_tuples([
            ('/a', 'AA'),
        ])
        self._assert_pstep_pmods_with_map_paths('P', pmods)

    def test_pstep_vs_pmods_maproot_empty1ststep(self):
        """Check ``pmod.map_path ()`` is equal with ``pstep(_pmod=pmod)``"""

        pmods = pmods_from_tuples([
            ('', 'AA'),
        ])
        self._assert_pstep_pmods_with_map_paths('', pmods)

    def test_pstep_vs_pmods_mapslash_empty1ststep(self):
        """Check ``pmod.map_path ()`` is equal with ``pstep(_pmod=pmod)``"""

        pmods = pmods_from_tuples([
            ('/', 'AA'),
        ])
        self._assert_pstep_pmods_with_map_paths('', pmods)

    def test_pstep_vs_pmods_mapstep_empty1ststep(self):
        """Check ``pmod.map_path ()`` is equal with ``pstep(_pmod=pmod)``"""

        pmods = pmods_from_tuples([
            ('/a', 'AA'),
        ])
        self._assert_pstep_pmods_with_map_paths('', pmods)

    def test_pmods_mass(self):
        p = Pstep(_pmod=self.PMODS())
        p.a
        p.abc['def']['123']
        self.assertListEqual(
            sorted(p._paths),
            sorted(['/root/b', '/root/BAR/DEF/234']),
            (p, p._paths))

    def test_pmods_lock_not_applying(self):
        p = Pstep('not dot', _pmod=self.PMODS())
        p.nota

        pmods = self.PMODS()
        pmods._alias = None
        p = Pstep(_pmod=pmods)

    def test_pmods_lock_CAN_RELOCATE(self):
        pmods = self.PMODS()
        pmods._alias = 'deep/root'
        p = Pstep(_pmod=pmods)
        p._lock = Pstep.CAN_RELOCATE
        p['for']._lock = Pstep.CAN_RELOCATE

    def test_pmods_lock_CAN_RENAME(self):
        pmods = self.PMODS()
        pmods._alias = 'deep/root'
        p = Pstep(_pmod=pmods)
        with self.assertRaises(ValueError, msg=p._paths):
            p._lock = Pstep.CAN_RENAME

        p = Pstep(_pmod=self.PMODS())
        with self.assertRaises(ValueError, msg=p._paths):
            p['for']._lock = Pstep.CAN_RENAME

    def test_pmods_lock_LOCKED(self):
        p = Pstep(_pmod=self.PMODS())
        with self.assertRaises(ValueError, msg=p._paths):
            p._lock = Pstep.LOCKED

        pmods = self.PMODS()
        pmods._alias = 'deep/root'
        with self.assertRaises(ValueError, msg=p._paths):
            p._lock = Pstep.LOCKED

        pmods = self.PMODS()
        pmods._alias = None
        p = Pstep(_pmod=pmods)
        with self.assertRaises(ValueError, msg=p._paths):
            p.abc._lock = Pstep.LOCKED

        pmods = self.PMODS()
        pmods._alias = None
        p = Pstep(_pmod=pmods)
        with self.assertRaises(ValueError, msg=p._paths):
            p['for']._lock = Pstep.LOCKED

    def test_assign(self):
        p1 = Pstep('root')

        def f1():
            p1.a = 1

        def f2():
            p1.a = p1

        def f3():
            p1.a = Pstep()

        self.assertRaises(AssertionError, f1)
        self.assertRaises(AssertionError, f2)
        self.assertRaises(AssertionError, f3)

    def test_indexing(self):
        m = {'a': 1, 'b': 2, 'c': {'cc': 33}}
        n = 'a'
        p = Pstep(n)
        self.assertEqual(m[p], m[n])
        self.assertEqual(m[p[n]], m[n])

    def test_idex_assigning(self):
        m = {'a': 1, 'b': 2, 'c': {'cc': 33}}
        n = 'a'
        p = Pstep(n)
        self.assertEqual(m[p], m[n])
        self.assertEqual(m[p[n]], m[n])

    def test_schema(self):
        json.dumps(Pstep())
        json.dumps({Pstep(): 1})

    def test_json(self):
        p = Pstep()
        p._schema.allOf = {}
        p._schema.type = 'list'
        p.a._schema.kws = {'minimum': 1}

    @unittest.skip('Unknwon why sets fail with Pstep!')
    def test_json_sets(self):
        json.dumps({Pstep(), Pstep()})


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
