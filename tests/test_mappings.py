#! python
#-*- coding: utf-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, unicode_literals

import doctest
import json
import re
import sre_constants
import sys
import unittest

from numpy import testing as npt

import functools as ft
from pandalone.mappings import (
    Pmod, pmods_from_tuples, Pstep, _join_paths, _append_step)
import pandalone.mappings
import pandas as pd
from tests._tutils import _init_logging


log = _init_logging(__name__)


def pmod2regexstrs(pmod):
    if pmod._regxs:
        return [r.pattern for r in list(pmod._regxs.keys())]


@unittest.skipIf(sys.version_info < (3, 4), "Doctests are made for py >= 3.3")
class Doctest(unittest.TestCase):

    def test_doctests(self):
        failure_count, test_count = doctest.testmod(
            pandalone.mappings,
            optionflags=doctest.NORMALIZE_WHITESPACE)  # | doctest.ELLIPSIS)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))


class TestJoinPaths(unittest.TestCase):

    def _check_append_step(self, all_steps, exp_steps):
        steps = all_steps[:-1]
        step = all_steps[-1]
        msg = "{} + {} != {}".format(steps, step, exp_steps)
        self.assertEqual(_append_step(steps, step), exp_steps, msg)

    def _masscheck_append_step(self, cases):
        for all_steps, exp_steps in cases:
            self._check_append_step(all_steps, exp_steps)

    def test_append_step_simple(self):
        cases = [
            [('a',), ('a',)],
            [('a', 'b', '..'), ('a',)],
            [('a', 'b', '.'), ('a', 'b')],
        ]
        self._masscheck_append_step(cases)

    def test_append_step_absolute(self):
        cases = [
            [('a', 'b', ''), ('',)],
            [('', ''), ('',)],
            [('',), ('',)],
        ]
        self._masscheck_append_step(cases)

    def test_append_step_dotdot(self):
        cases = [
            [('', '..'), ('',)],
            [('', '.'), ('',)],
            [('a', '..'), ()],
            [('..',), ('..',)],
            [('..', '..'), ('..', '..')],
            [('.',), ()],
        ]
        self._masscheck_append_step(cases)

    def test_join_paths_dotdot_on_empty(self):
        self.assertEqual(_join_paths('..',), '..')
        self.assertEqual(_join_paths('a', '..', '..'), '..')
        self.assertEqual(_join_paths('..'), '..')

    def test_join_paths_dotdot_on_rooted(self):
        self.assertEqual(_join_paths('', '..'), '',)

    def test_join_paths_root_end(self):
        self.assertEqual(_join_paths('', 'a', ''), '')
        self.assertEqual(_join_paths('', ''), '')
        self.assertEqual(_join_paths(''), '')


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

    def _build_pmod_manually(self):
        return Pmod(
            _steps={'a': Pmod(_alias='A')},
            _regxs=[('a(\w*)', Pmod('AWord')),
                    ('a(\d*)', Pmod(_alias='A_\\1')), ])

    def assert_descend_stops(self, cpmod, msg=None):
        self.assertEqual(cpmod, (None, None), msg)

    def test_descend_BAD(self):
        pm = self._build_pmod_manually()
        self.assert_descend_stops(pm.descend('BAD'))
        self.assert_descend_stops(pm.descend('a-'))
        self.assert_descend_stops(pm.descend('a$'))

    def test_alias_BAD(self):
        pm = self._build_pmod_manually()
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
        pm = self._build_pmod_manually()

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
        self.assertEqual(pm.map_path('/'), '')

    def test_map_path_rootUnmapped_emptyPath(self):
        pm = Pmod()
        self.assertEqual(pm.map_path(''), '')

    def test_map_path_rootMapped(self):

        pm = pmods_from_tuples([
            ('', 'New/Root'),
            ('/a/b', 'B'),
        ])
        self.assertEqual(pm.map_path('/a/b'), 'New/Root/a/B')

    def _build_simple_pmod_from_tuples(self):
        return pmods_from_tuples([('/a', 'A'),
                                  ('/~a(\w*)', 'AWord'),
                                  ('/~a(\d*)', 'A_\\1'),
                                  ])

    def test_map_path_rootMapped_relative(self):
        pm = pmods_from_tuples([('', 'root')])
        self.assertEqual(pm.map_path(''), 'root')
        self.assertEqual(pm.map_path('/a'), 'root/a')
        # self.assertEqual(pm.map_path('/'), 'root')  ## TODO: FIX map2slash

        pm = pmods_from_tuples([('', 'root'), ('/', 'root2')])
        self.assertEqual(pm.map_path(''), 'root')
        self.assertEqual(pm.map_path('/a'), 'root/a')
        self.assertEqual(pm.map_path('/'), 'root/root2')

    def test_map_path_rootMapped_absolute(self):
        pm = pmods_from_tuples([('', '/root')])
        self.assertEqual(pm.map_path(''), '/root')
        self.assertEqual(pm.map_path('/a'), '/root/a')
        # self.assertEqual(pm.map_path('/'), '/root')  ## TODO: FIX map2slash

        pm = pmods_from_tuples([('', '/root'), ('/', 'root2')])
        self.assertEqual(pm.map_path(''), '/root')
        self.assertEqual(pm.map_path('/a'), '/root/a')
        self.assertEqual(pm.map_path('/'), '/root/root2')

    def test_map_path_missed(self):
        pm = self._build_simple_pmod_from_tuples()

        self.assertEqual(pm.map_path('/c'), '/c')
        self.assertEqual(pm.map_path('/c/a'), '/c/a')
        self.assertEqual(pm.map_path('/a$/a4'), '/a$/a4')

    def test_map_path_simple(self):
        pm = self._build_simple_pmod_from_tuples()

        self.assertEqual(pm.map_path('/a'), '/A')
        self.assertEqual(pm.map_path('/a/b'), '/A/b')

    def test_map_path_regex(self):
        pm = self._build_simple_pmod_from_tuples()

        self.assertEqual(pm.map_path('/a'), '/A')
        self.assertEqual(pm.map_path('/a_wo/b'), '/AWord/b')
        self.assertEqual(pm.map_path('/a12/b'), '/A_12/b')

    def test_map_path_regex_2_dot_dotdot_root_abs(self):
        pmods = pmods_from_tuples([('/~b.*/~c(.*)',  r'../C/\1')])
        self.assertEqual(pmods.map_path('/'), '')
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/big/child'), '/big/C/hild')
        self.assertEqual(pmods.map_path(''), '')

    def test_build_pmods_steps_no_regxs(self):
        pmods_tuples = [
            ('/a', 'A1/A2'),
            ('/a/b', 'B'),
        ]
        pmods = pmods_from_tuples(pmods_tuples)

        pmods = pmods._steps['']
        self.assertFalse(bool(pmods._regxs))
        self.assertSetEqual(set(pmods._steps), {'a'})
        self.assertEqual(pmods._steps['a']._alias, 'A1/A2')
        self.assertEqual(pmods._steps['a']._steps['b']._alias, 'B')

    def test_build_pmods_regex_no_steps(self):
        pmods_tuples = [
            ('/~a*', 'A1/A2'),
            ('/~a[1]?/~b?', 'B'),
        ]
        pmods = pmods_from_tuples(pmods_tuples)

        pmods = pmods._steps['']
        self.assertFalse(bool(pmods._steps))
        self.assertEqual(pmod2regexstrs(pmods), ['a*', 'a[1]?'])

    @unittest.skipIf(sys.version_info < (3, 4), "String repr differ in py-2.")
    def test_pmods_from_tuples_repr(self):
        pmods_tuples = [
            ('/a', 'A'),
            ('/~a\w*', 'A1/A2'),
            ('/~a\d*/~b?', 'D/D'),
        ]
        pmods = pmods_from_tuples(pmods_tuples)

        pmods = pmods._steps['']
        s = [r"pmod({'a': pmod('A')}, OrderedDict([(re.compile('a\\w*'), "
             r"pmod('A1/A2')), (re.compile('a\\d*'), "
             r"pmod(OrderedDict([(re.compile('b?'), pmod('D/D'))])))]))"
             ]
        self.assertEqual(str(pmods), "".join(s))

    def test_pmods_from_tuples_rootMapping(self):
        self.assertEqual(pmods_from_tuples([('', 'A/B')]),
                         Pmod(_steps={'': Pmod(_alias='A/B')}))
        self.assertEqual(pmods_from_tuples([('/', 'A/B')]),
                         Pmod(_steps={'':
                                      Pmod(_steps={'': Pmod(_alias='A/B')})}))

    def test_map_path_2_root(self):
        pmods = pmods_from_tuples([('/a', '')])
        self.assertEqual(pmods.map_path('/a'), '')
        self.assertEqual(pmods.map_path('/a/b'), '/b')
        self.assertEqual(pmods.map_path('/b'), '/b')

        pmods = pmods_from_tuples([('/a/b', '')])
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '')
        self.assertEqual(pmods.map_path('/a/b/c'), '/c')
        self.assertEqual(pmods.map_path('/c'), '/c')

    def test_map_path_2_abs(self):
        pmods = pmods_from_tuples([('/a', '/A')])
        self.assertEqual(pmods.map_path('/a'), '/A')
        self.assertEqual(pmods.map_path('/a/b'), '/A/b')
        self.assertEqual(pmods.map_path('/A/b'), '/A/b')

        pmods = pmods_from_tuples([('/a/b', '/A/AA')])
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/A/AA')
        self.assertEqual(pmods.map_path('/a/b/c'), '/A/AA/c')
        self.assertEqual(pmods.map_path('/A/AA/c'), '/A/AA/c')

    def test_map_path_2_dot(self):
        pmods = pmods_from_tuples([('/a', '.')])
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/a/b')
        self.assertEqual(pmods.map_path('/b'), '/b')

        pmods = pmods_from_tuples([('/a', './bb')])
        self.assertEqual(pmods.map_path('/a'), '/a/bb')
        self.assertEqual(pmods.map_path('/a/b'), '/a/bb/b')
        self.assertEqual(pmods.map_path('/b/c'), '/b/c')

    def test_map_path_2_dotdot(self):
        pmods = pmods_from_tuples([('/a', '..')])
        self.assertEqual(pmods.map_path('/a'), '')
        self.assertEqual(pmods.map_path('/a/b'), '/b')
        self.assertEqual(pmods.map_path('/b'), '/b')

        pmods = pmods_from_tuples([('/a/b', '..')])
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/a')
        self.assertEqual(pmods.map_path('/a/b/c'), '/a/c')

        pmods = pmods_from_tuples([('/a/b', '../cc')])
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/a/cc')
        self.assertEqual(pmods.map_path('/a/b/c'), '/a/cc/c')

    def test_map_path_slash_2_dot_dotdot_root_abs(self):
        pmods = pmods_from_tuples([('/', '.')])
        self.assertEqual(pmods.map_path('/'), '')
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/a/b')
        self.assertEqual(pmods.map_path(''), '')

        pmods = pmods_from_tuples([('/', '..')])
        self.assertEqual(pmods.map_path('/'), '')
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/a/b')
        self.assertEqual(pmods.map_path(''), '')

        pmods = pmods_from_tuples([('/', '')])
        self.assertEqual(pmods.map_path('/'), '')
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/a/b')
        self.assertEqual(pmods.map_path(''), '')

        pmods = pmods_from_tuples([('/', '/')])
        self.assertEqual(pmods.map_path('/'), '')
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/a/b')
        self.assertEqual(pmods.map_path(''), '')

        pmods = pmods_from_tuples([('/', '/A/B')])
        self.assertEqual(pmods.map_path('/'), '/A/B')
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/a/b')
        self.assertEqual(pmods.map_path(''), '')

    def test_map_path_root_2_dot_dotdot_root_abs(self):
        pmods = pmods_from_tuples([('', '.')])
        self.assertEqual(pmods.map_path('/'), '')
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/a/b')
        self.assertEqual(pmods.map_path(''), '')

        pmods = pmods_from_tuples([('', '..')])
        self.assertEqual(pmods.map_path('/'), '')
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/a/b')
        self.assertEqual(pmods.map_path(''), '')

        pmods = pmods_from_tuples([('', '/')])
        self.assertEqual(pmods.map_path('/'), '')
        self.assertEqual(pmods.map_path('/a'), '/a')
        self.assertEqual(pmods.map_path('/a/b'), '/a/b')
        self.assertEqual(pmods.map_path(''), '')

        pmods = pmods_from_tuples([('', '/A/B')])
        #self.assertEqual(pmods.map_path('/'), '/A/B')
        self.assertEqual(pmods.map_path('/a'), '/A/B/a')
        self.assertEqual(pmods.map_path('/a/b'), '/A/B/a/b')
        self.assertEqual(pmods.map_path(''), '/A/B')

    def test_pmods_from_tuple_ignores_empties(self):
        # Ignored by :func:`pmods_from_tuples().
        pmods = pmods_from_tuples([('', '')])
        self.assertEqual(pmods, Pmod())
        pmods = pmods_from_tuples([('spam', None)])
        self.assertEqual(pmods, Pmod())
        pmods = pmods_from_tuples([(None, 'spam')])
        self.assertEqual(pmods, Pmod())


class TestPstep(unittest.TestCase):

    def PMODS(self, absolute=False):
        return pmods_from_tuples([
            ('',            '/root' if absolute else 'root'),
            ('/a',          'b'),
            ('/abc',        'BAR'),
            ('/abc/df',     'DEF'),
            ('/abc/df/a123', '234'),
            ('/some',       '/sub/path'),
        ])

    def test_equality(self):
        p = Pstep()
        self.assertEqual(str(p), '')
        self.assertEqual(p, Pstep(''))

        self.assertEqual(str(p.a), 'a')
        self.assertEqual(p.a, Pstep('a'))

        n = 'foo'
        p = Pstep(n)
        self.assertEqual(str(p), n)
        self.assertEquals(p, Pstep(n))
        self.assertEquals(str(p.foo), str(p))

        n = '/foo'
        p = Pstep(n)
        self.assertEqual(str(p), n)
        self.assertEquals(p, Pstep(n))

    def test_buildtree_valid_ops(self):
        p = Pstep()
        p.abc
        p.abc.ddd = 'def'
        p.abc.dfg = 'defg'
        p.n123
        p.pp = '321'
        p._some_hidden = '12'
        p.a.b.c.d = '123'
        p.a.b.c.d.e = '123'

        return p

    def test_constructor(self):
        pmod = pmods_from_tuples([('/a', 'A'), ('/a/c', 'C'), ('/b/c', 'C1')])
        p = pmod.step()
        self.assertEquals(p.a, 'A')
        self.assertEquals(p.a.c, 'C')
        self.assertEqual(p.b, 'b')
        self.assertEqual(p.b.c, 'C1')

        p.a = 'AA'
        self.assertEqual(p.a, 'AA')
        self.assertIn('AA', p._steps())
        self.assertIn('c', p.a._steps(keys=True))
        self.assertIn('C', p.a._steps())
        self.assertIn('c', p.b._steps(keys=True))
        self.assertIn('C1', p.b._steps())

    def test_buildtree_invalid_ops(self):

        p = Pstep()

        def f1(p):
            p.abc = 1

        def f2(p):
            p.a = 1

        def f3(p):
            p.a.b = 1

        def f4(p):
            p.a.c = 1

        for f in [f1, f2, f3, f4]:
            p = Pstep()
            with self.assertRaises(AssertionError, msg=f):
                f(p),

    def test_paths_empty1ststep(self):
        p = Pstep()
        self.assertListEqual(sorted(p._paths()), sorted(['']))

        p.a
        self.assertListEqual(sorted(p._paths()), sorted(['/a']))

        p = Pstep('')
        self.assertListEqual(sorted(p._paths()), sorted(['']))

        p = Pstep('')
        p.a
        self.assertListEqual(sorted(p._paths()), sorted(['/a']))

    def test_paths(self):
        p = Pstep('r')
        self.assertListEqual(sorted(p._paths()), sorted(['r']))

        p.a
        self.assertListEqual(sorted(p._paths()), sorted(['r/a']))

    def test_paths_multi_empty1ststep(self):
        p = Pstep()
        p.abc
        p.abc.aaa = 'def'
        p.abc.defg
        p.n123
        p.pp = '321'
        p._some_hidden = 12
        exp = [
            '/abc/def',
            '/abc/defg',
            '/321',
            '/n123',
        ]
        self.assertListEqual(sorted(p._paths()), sorted(exp))

        exp = [
            '/abc/(aaa-->def)',
            '/abc/defg',
            '/(pp-->321)',
            '/n123',
        ]
        self.assertListEqual(sorted(p._paths(with_orig=True)), sorted(exp))

    def test_paths_multi_nonemptyroot(self):
        p = Pstep('r')
        p.abc
        p.abc.aaa = 'def'
        p.abc.defg
        p.n123
        p.pp = '321'
        p._some_hidden = 12
        exp = [
            'r/abc/def',
            'r/abc/defg',
            'r/321',
            'r/n123',
        ]
        self.assertListEqual(sorted(p._paths()), sorted(exp))

        exp = [
            'r/abc/(aaa-->def)',
            'r/abc/defg',
            'r/(pp-->321)',
            'r/n123',
        ]

    ### DOT ###

    def test_paths_dot_atroot_emptyroot(self):
        p = Pstep()
        p.pp = '.'
        self.assertListEqual(sorted(p._paths()), sorted(['']))

    def test_paths_dot_emptyroot(self):
        p = Pstep()
        p.a.b = '.'
        self.assertListEqual(sorted(p._paths()), sorted(['/a']))

    def test_paths_dot_atroot(self):
        p = Pstep('r')
        p.pp = '.'
        self.assertListEqual(sorted(p._paths()), sorted(['r']))
        p.pp = '.'
        self.assertListEqual(sorted(p._paths()), sorted(['r']))

    def test_paths_dot(self):
        p = Pstep('r')
        p.a.b = '.'
        self.assertListEqual(sorted(p._paths()), sorted(['r/a']))

    def test_paths_dot_multi_emptyroot(self):
        p = Pstep('')
        p.pp = '.'
        p.pp.a.b
        p.c.pp = '.'
        p.c.pp.d
        self.assertListEqual(sorted(p._paths()), sorted(['/a/b', '/c/d']))

    def test_paths_dot_multi(self):
        p = Pstep('r')
        p.pp = '.'
        p.pp.a.b
        p.c.cc = '.'
        p.c.cc.d
        self.assertListEqual(sorted(p._paths()), sorted(['r/a/b', 'r/c/d']))

    ### DOTDOT ###
    # NOT TO BE USED DIRECTLY
    # BC HIDS PATHS,
    # but testing for mapping.

    def test_paths_dotdot_atroot_emptyroot(self):
        p = Pstep()
        p.pp = '..'
        p.pp.a
        self.assertListEqual(sorted(p._paths()), sorted(['/a']))

        p = Pstep()
        p.pp = '..'
        p.pp.a
        p.pp.b
        self.assertListEqual(sorted(p._paths()), sorted(['/a', '/b']))

    def test_paths_dotdot_emptyroot(self):
        p = Pstep()
        p.a.aa = '..'
        p.a.aa.b
        self.assertListEqual(sorted(p._paths()), sorted(['/b']))

        p = Pstep()
        p.a.aa = '..'
        p.a.aa.b
        p.a.c
        self.assertListEqual(sorted(p._paths()), sorted(['/a/c', '/b']))

    def test_paths_dotdot_atroot(self):
        p = Pstep('r')
        p.pp = '..'
        p.pp.a
        self.assertListEqual(sorted(p._paths()), sorted(['a']))

    def test_paths_dotdot(self):
        p = Pstep('r')
        p.a.aa = '..'
        p.a.aa.b
        self.assertListEqual(sorted(p._paths()), sorted(['r/b']))

    def test_paths_dotdot_multi_emptyroot(self):
        p = Pstep('')
        p.pp = '..'
        p.pp.a.b
        p.c.cc = '..'
        p.c.cc.d
        self.assertListEqual(sorted(p._paths()), sorted(['/a/b', '/d']))

    def test_paths_dotdot_multi(self):
        p = Pstep('r')
        p.pp = '..'
        p.pp.a.b
        p.c.cc = '..'
        p.c.cc.d
        self.assertListEqual(sorted(p._paths()), sorted(['a/b', 'r/d']))

    ### PMODS ###

    def test_pmods_miss(self):
        p = pmods_from_tuples([('/MISS', 'BOO')]).step()
        self.assertEquals(p, '', (p, p._paths()))
        p = Pstep('foo', pmods_from_tuples([('/MISS', 'BOO')]))
        self.assertEquals(p, 'foo', (p, p._paths()))

    def test_pmods_emptyroot_rootunmapped(self):
        p = pmods_from_tuples([('/a', 'bar')]).step()
        p.a
        self.assertEquals(p, '', (p, p._paths()))
        self.assertEquals(p.a, 'bar', (p, p._paths()))
        self.assertEquals(sorted(p._paths()), ['/bar'])

        self.assertEquals(p.a.b, 'b')
        self.assertEquals(sorted(p._paths()), ['/bar/b'])

        self.assertEquals(p.c, 'c')
        self.assertEquals(sorted(p._paths()), ['/bar/b', '/c'])

        self.assertEquals(p.a.f, 'f')
        self.assertEquals(sorted(p._paths()), ['/bar/b', '/bar/f', '/c'])

    def test_pmods_nonemptyroot_rootunmapped(self):
        p = pmods_from_tuples([('root/a', 'bar')]).step('root')
        p.a
        self.assertEquals(p, 'root', (p, p._paths()))
        self.assertEquals(p.a, 'bar', (p, p._paths()))
        self.assertEquals(sorted(p._paths()), ['root/bar'])

        self.assertEquals(p.a.b, 'b')
        self.assertEquals(sorted(p._paths()), ['root/bar/b'])

        self.assertEquals(p.c, 'c')
        self.assertEquals(sorted(p._paths()), ['root/bar/b', 'root/c'])

        self.assertEquals(p.a.f, 'f')
        self.assertEquals(sorted(p._paths()),
                          ['root/bar/b', 'root/bar/f', 'root/c'])

    def test_pmods_emptyroot_rootmapped(self):
        p = pmods_from_tuples([('', 'root'), ('/a', 'bar')]).step()
        p.a
        self.assertEquals(p, 'root')
        self.assertEquals(p.a, 'bar')
        self.assertEquals(sorted(p._paths()), ['root/bar'])

        self.assertEquals(p.a.b, 'b')
        self.assertEquals(sorted(p._paths()), ['root/bar/b'])

        self.assertEquals(p.c, 'c')
        self.assertEquals(sorted(p._paths()), ['root/bar/b', 'root/c'])

        self.assertEquals(p.a.f, 'f')
        self.assertEquals(
            sorted(p._paths()), ['root/bar/b', 'root/bar/f', 'root/c'])

    def test_pmods_nonemptyroot_rootmapped(self):
        p = pmods_from_tuples(
            [('root', 'ROOT'), ('root/a', 'bar')]).step('root')
        p.a
        self.assertEquals(p, 'ROOT')
        self.assertEquals(p.a, 'bar')
        self.assertEquals(sorted(p._paths()), ['ROOT/bar'])

        self.assertEquals(p.a.b, 'b')
        self.assertEquals(sorted(p._paths()), ['ROOT/bar/b'])

        self.assertEquals(p.c, 'c')
        self.assertEquals(sorted(p._paths()), ['ROOT/bar/b', 'ROOT/c'])

        self.assertEquals(p.a.f, 'f')
        self.assertEquals(
            sorted(p._paths()), ['ROOT/bar/b', 'ROOT/bar/f', 'ROOT/c'])

    def _build_psteps(self, root='', pmods=None):
        p = Pstep(root, pmods)
        p.a.b.c
        p.a.b.d
        p.a.c
        p.abc.df
        p.n123
        p.a.n123
        p.cc.a123.a123
        p.cc.a123.abc
        p.b321
        # p[''] ##NOTE: Will always return to root.

        return p

    def _assert_pstep_pmods_with_map_paths(self, root, pmods):
        ps1 = self._build_psteps(root, pmods=pmods)
        ps2 = self._build_psteps(root)

        self.assertEqual(sorted(ps1._paths()),
                         sorted(pmods.map_paths(ps2._paths())))

    def test_pstep_vs_pmods_maproot(self):
        root = 'R'
        pmods = pmods_from_tuples([(root, 'AA'), ])
        self._assert_pstep_pmods_with_map_paths(root, pmods)

    def test_pstep_vs_pmods_mapslash(self):
        pmods = pmods_from_tuples([('/', 'AA'), ])
        self._assert_pstep_pmods_with_map_paths('P', pmods)

    def test_pstep_vs_pmods_mapstep(self):

        pmods = pmods_from_tuples([('/a', 'AA'), ])
        self._assert_pstep_pmods_with_map_paths('P', pmods)

    def test_pstep_vs_pmods_maproot_emptyroot(self):
        pmods = pmods_from_tuples([('', 'AA'), ])
        self._assert_pstep_pmods_with_map_paths('', pmods)

    def test_pstep_vs_pmods_mapslash_empty1ststep(self):
        pmods = pmods_from_tuples([('/', 'AA'), ])
        self._assert_pstep_pmods_with_map_paths('', pmods)

    def test_pstep_vs_pmods_mapstep_empty1ststep(self):
        pmods = pmods_from_tuples([('/a', 'AA'), ])
        self._assert_pstep_pmods_with_map_paths('', pmods)

    def test_paths_pmods_mass(self):
        p = self.PMODS().step()
        p.a
        p.abc.df.a123
        exp = ['root/b', 'root/BAR/DEF/234']
        self.assertListEqual(sorted(p._paths()), sorted(exp), (p, p._paths()))

        p = self.PMODS(absolute=True).step()
        p.a
        p.abc.df
        p.abc.df.a123
        exp = ['/root/b', '/root/BAR/DEF/234']
        self.assertListEqual(sorted(p._paths()), sorted(exp), (p, p._paths()))

    def test_paths_pmods_orig_mass(self):
        p = self.PMODS().step()
        p.a
        p.abc.df.a123
        exp = [
            '(-->root)/(a-->b)',
            '(-->root)/(abc-->BAR)/(df-->DEF)/(a123-->234)'
        ]
        self.assertListEqual(
            sorted(p._paths(with_orig=True)), sorted(exp), (p, p._paths()))

        p = self.PMODS(absolute=True).step()
        p.a
        p.abc.df.a123
        exp = [
            '(-->/root)/(a-->b)',
            '(-->/root)/(abc-->BAR)/(df-->DEF)/(a123-->234)'
        ]
        self.assertListEqual(
            sorted(p._paths(with_orig=True)), sorted(exp), (p, p._paths()))

    def test_pmods_lock_not_applying(self):
        p = self.PMODS().step('not dot')
        p.nota
        p = self.PMODS(absolute=True).step('not dot')
        p.nota

        pmods = self.PMODS()
        pmods._alias = None
        p = pmods.step()
        pmods = self.PMODS(absolute=True)
        pmods._alias = None
        p = pmods.step()

    def test_pmods_lock_CAN_RELOCATE_unmapped(self):
        p = Pstep()
        p._locked = Pstep.CAN_RELOCATE
        p2 = p._lock
        self.assertIs(p2, p)
        p2 = p._fix
        self.assertIs(p2, p)

        p2 = p.abc._lock
        self.assertIs(p2, p.abc)
        p2 = p.abc._fix
        self.assertIs(p2, p.abc)

    def test_pmods_lock_CAN_RELOCATE_mapped(self):
        pmods = self.PMODS()
        p = pmods.step()
        p._locked = Pstep.CAN_RELOCATE

        p2 = p.unmapped._lock
        self.assertIs(p2, p.unmapped)
        p2 = p.unmapped._fix
        self.assertIs(p2, p.unmapped)

    def test_pmods_lock_CAN_RENAME_root(self):
        pmods = self.PMODS(absolute=True)
        p = pmods.step()
        with self.assertRaises(ValueError, msg=p._paths()):
            p._locked = Pstep.CAN_RENAME
        with self.assertRaises(ValueError, msg=p._paths()):
            p._lock

    def test_pmods_lock_CAN_RENAME_rest(self):
        p = self.PMODS().step()
        p2 = p.a._fix  # `a` just renamed `b`.
        self.assertIs(p2, p.a)
        with self.assertRaises(ValueError, msg=p._paths()):
            p.some._locked = Pstep.CAN_RENAME
        with self.assertRaises(ValueError, msg=p._paths()):
            p.some._lock

    def test_pmods_lock_LOCKED_root(self):
        p = self.PMODS().step()
        with self.assertRaises(ValueError, msg=p._paths()):
            p._locked = Pstep.LOCKED
        with self.assertRaises(ValueError, msg=p._paths()):
            p._lock

    def test_pmods_lock_LOCKED_rest(self):
        pmods = self.PMODS()
        p = pmods.step()
        with self.assertRaises(ValueError, msg=p._paths()):
            p.abc._locked = Pstep.LOCKED
        with self.assertRaises(ValueError, msg=p._paths()):
            p.abc._lock

        with self.assertRaises(ValueError, msg=p._paths()):
            p.some._locked = Pstep.LOCKED
        with self.assertRaises(ValueError, msg=p._paths()):
            p.some._lock

    def test_map_from_pstep(self):
        p = self.test_buildtree_valid_ops()
        pairs = p._derrive_map_tuples()
        #print('\n'.join(str(p) for p in pairs))

        exp = [
            ('', ''),
            ('/a', 'a'),
            ('/a/b', 'b'),
            ('/a/b/c', 'c'),
            ('/a/b/c/d', '123'),
            ('/a/b/c/d/e', '123'),
            ('/abc', 'abc'),
            ('/abc/ddd', 'def'),
            ('/abc/dfg', 'defg'),
            ('/n123', 'n123'),
            ('/pp', '321'),
        ]
        self.assertEqual(pairs, exp)

    def test_assign(self):
        p1 = Pstep('root')

        def f1():
            p1.a = 1

        p1.a = 'a'
        p1.a = p1
        p1.a = Pstep()  # TODO: Make step assigning work?

        self.assertRaises(AssertionError, f1)

    def test_schema(self):
        json.dumps(Pstep())
        json.dumps({Pstep(): 1})

    def test_json(self):
        p = Pstep()
        p._schema.allOf = {}
        p._schema.type = 'list'
        p.a._schema.kws = {'minimum': 1}

    def test_json_list(self):
        json.dumps([Pstep(), Pstep('a')])

    def test_tags_paths(self):
        p = Pstep()
        self.assertEqual(p._tags, ())
        self.assertEqual(p._paths(tag='BAD'), [])

        p._tag('inp')
        self.assertSequenceEqual(sorted(p._tags), ['inp'])
        self.assertEqual(p._paths(tag='BAD'), [])
#         self.assertEqual(p._paths(tag='inp'), [''])

        self.assertEqual(p.a._tags, ())

        p.a._tag('t')
        self.assertSequenceEqual(sorted(p._paths(tag='t')), ['/a'])
        self.assertEqual(p._paths(tag='BAD'), [])
        p.b
        self.assertSequenceEqual(sorted(p._paths(tag='t')), ['/a'])
        p.a.aa
        self.assertSequenceEqual(sorted(p._paths(tag='t')), ['/a'])
        p.b.c
        self.assertSequenceEqual(sorted(p._paths(tag='t')), ['/a'])
        p.b._tag('t')
        self.assertSequenceEqual(sorted(p._paths(tag='t')), ['/a', '/b'])
        p.b.c._tag('t')
        self.assertSequenceEqual(
            sorted(p._paths(tag='t')), ['/a', '/b', '/b/c'])
        p.a.aa._tag('nt')
        self.assertSequenceEqual(
            sorted(p._paths(tag='t')), ['/a', '/b', '/b/c'])
        p.a.aa._tag('t')
        self.assertSequenceEqual(
            sorted(p._paths(tag='t')), ['/a', '/a/aa', '/b', '/b/c'])

    def test_tags_delete(self):
        p = Pstep()
        p.a._tag('u').aa._tag('t').aaa
        p.b.c._tag('u')
        self.assertSequenceEqual(sorted(p._paths()), ['/a/aa/aaa', '/b/c'])
        self.assertSequenceEqual(sorted(p._paths(tag='t')), ['/a/aa'])
        self.assertSequenceEqual(sorted(p._paths(tag='u')), ['/a', '/b/c'])
        self.assertSequenceEqual(sorted(p._paths(tag='BAD')), [])

        p.a._tag_remove('u').aa.aaa
        self.assertSequenceEqual(sorted(p._paths()), ['/a/aa/aaa', '/b/c'])
        self.assertSequenceEqual(sorted(p._paths(tag='t')), ['/a/aa'])
        self.assertSequenceEqual(sorted(p._paths(tag='u')), ['/b/c'])

        p.a.aa._tag_remove('t').aaa
        p.a.aa._tag_remove('u').aaa  # Useless
        p.b.c._tag_remove('u')
        self.assertSequenceEqual(sorted(p._paths()), ['/a/aa/aaa', '/b/c'])
        self.assertSequenceEqual(sorted(p._paths(tag='t')), [])
        self.assertSequenceEqual(sorted(p._paths(tag='u')), [])

        p.a.aa.aaa._tag_remove('u')  # Useless
        p.a.aa.aaa._tag_remove('t')
        self.assertSequenceEqual(sorted(p._paths()), ['/a/aa/aaa', '/b/c'])
        self.assertSequenceEqual(sorted(p._paths(tag='t')), [])
        self.assertSequenceEqual(sorted(p._paths(tag='u')), [])

    def test_as_pandas_index(self):
        df = pd.DataFrame([1, 3], columns=['a'])
        s = Pstep('a')
        npt.assert_array_equal(df[s], [1, 3])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
