from collections import OrderedDict
import doctest
import json
import re
import unittest

import functools as ft
import numpy.testing as npt
from pandalone.components import (
    Assembly, FuncComponent, Pstep, convert_df_as_pmods_tuples, _Pmod)
import pandalone.components
import pandas as pd


class TestDoctest(unittest.TestCase):

    def test_doctests(self):
        failure_count, test_count = doctest.testmod(
            pandalone.components,
            optionflags=doctest.NORMALIZE_WHITESPACE)  # | doctest.ELLIPSIS)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))


class Test_Pmod(unittest.TestCase):

    def setUp(self):
        self.assert_Pmod_class_attributes_not_modified()

    def tearDown(self):
        self.assert_Pmod_class_attributes_not_modified()

    def assert_Pmod_class_attributes_not_modified(self):
        self.assertIsNone(_Pmod.alias)
        self.assertEqual(_Pmod._steps, [])
        self.assertEqual(_Pmod._regxs, [])

    def pmod2regexstrs(self, pmod):
        if pmod._regxs:
            return [r.pattern for r in list(pmod._regxs.keys())]

    def test_Pmod_merge_name(self):
        pm1 = _Pmod(alias='pm1')
        pm2 = _Pmod(alias='pm2')
        pm = pm1._merge(pm2)
        self.assertEqual(pm.alias, 'pm2')
        self.assertEqual(pm1.alias, 'pm1')
        self.assertEqual(pm2.alias, 'pm2')
        pm = pm2._merge(pm1)
        self.assertEqual(pm.alias, 'pm1')
        self.assertEqual(pm1.alias, 'pm1')
        self.assertEqual(pm2.alias, 'pm2')

        pm1 = _Pmod()
        pm2 = _Pmod(alias='pm2')
        pm = pm1._merge(pm2)
        self.assertEqual(pm.alias, 'pm2')
        self.assertEqual(pm1.alias, None)
        self.assertEqual(pm2.alias, 'pm2')
        pm = pm2._merge(pm1)
        self.assertEqual(pm.alias, 'pm2')
        self.assertEqual(pm1.alias, None)
        self.assertEqual(pm2.alias, 'pm2')

    def test_Pmod_merge_name_recurse(self):
        pm1 = _Pmod(alias='pm1', _steps={'a': _Pmod(alias='R1')})
        pm2 = _Pmod(alias='pm2', _steps={'a': _Pmod(alias='R2'),
                                         'b': _Pmod(alias='R22')})
        pm = pm1._merge(pm2)
        self.assertEqual(pm._steps['a'].alias, 'R2')
        self.assertEqual(pm1._steps['a'].alias, 'R1')
        self.assertEqual(pm2._steps['a'].alias, 'R2')
        self.assertEqual(len(pm1._steps), 1)
        self.assertEqual(len(pm2._steps), 2)
        pm = pm2._merge(pm1)
        self.assertEqual(pm._steps['a'].alias, 'R1')
        self.assertEqual(pm1._steps['a'].alias, 'R1')
        self.assertEqual(pm2._steps['a'].alias, 'R2')
        self.assertEqual(len(pm1._steps), 1)
        self.assertEqual(len(pm2._steps), 2)

    def test_Pmod_merge_steps(self):
        pm1 = _Pmod(alias='pm1', _steps={'a': _Pmod(alias='A'),
                                         'c': _Pmod(alias='C')})
        pm2 = _Pmod(alias='pm2', _steps={'b': _Pmod(alias='B'),
                                         'a': _Pmod(alias='AA'),
                                         'd': _Pmod(alias='DD'),
                                         })
        pm = pm1._merge(pm2)
        self.assertEqual(sorted(pm._steps.keys()), list('abcd'))
        pm = pm2._merge(pm1)
        self.assertEqual(sorted(pm._steps.keys()), list('abcd'))
        self.assertEqual(len(pm1._steps), 2)
        self.assertEqual(len(pm2._steps), 3)

        pm1 = _Pmod(_steps={'a': _Pmod(alias='A'),
                            'c': _Pmod(alias='C')})
        pm2 = _Pmod(alias='pm2', _steps={'b': _Pmod(alias='B'),
                                         'a': _Pmod(alias='AA'),
                                         'd': _Pmod(alias='DD'),
                                         })
        pm = pm1._merge(pm2)
        self.assertEqual(sorted(pm._steps.keys()), list('abcd'))
        self.assertEqual(pm._steps['a'].alias, 'AA')
        self.assertEqual(len(pm1._steps), 2)
        self.assertEqual(len(pm2._steps), 3)
        self.assertEqual(pm._regxs, [])
        self.assertEqual(pm1._regxs, [])
        self.assertEqual(pm2._regxs, [])

        pm = pm2._merge(pm1)
        self.assertEqual(sorted(pm._steps.keys()), list('abcd'))
        self.assertEqual(pm._steps['a'].alias, 'A')
        self.assertEqual(len(pm1._steps), 2)
        self.assertEqual(len(pm2._steps), 3)
        self.assertEqual(pm._regxs, [])
        self.assertEqual(pm1._regxs, [])
        self.assertEqual(pm2._regxs, [])

    def test_Pmod_merge_regxs(self):
        pm1 = _Pmod(alias='pm1', _regxs=[
            ('e', _Pmod(alias='E')),
            ('a', _Pmod(alias='A')),
            ('c', _Pmod(alias='C'))])
        pm2 = _Pmod(alias='pm2', _regxs=[
            ('b', _Pmod(alias='B')),
            ('a', _Pmod(alias='AA')),
            ('d', _Pmod(alias='DD')),
        ])

        pm = pm1._merge(pm2)
        self.assertSequenceEqual(self.pmod2regexstrs(pm), list('ecbad'))
        self.assertEqual(pm._regxs[re.compile('a')].alias, 'AA')
        self.assertSequenceEqual(self.pmod2regexstrs(pm1), list('eac'))
        self.assertSequenceEqual(self.pmod2regexstrs(pm2), list('bad'))
        self.assertEqual(pm._steps, [])
        self.assertEqual(pm1._steps, [])
        self.assertEqual(pm2._steps, [])

        pm = pm2._merge(pm1)
        self.assertSequenceEqual(self.pmod2regexstrs(pm), list('bdeac'))
        self.assertEqual(pm._regxs[re.compile('a')].alias, 'A')
        self.assertSequenceEqual(self.pmod2regexstrs(pm1), list('eac'))
        self.assertSequenceEqual(self.pmod2regexstrs(pm2), list('bad'))
        self.assertEqual(pm._steps, [])
        self.assertEqual(pm1._steps, [])
        self.assertEqual(pm2._steps, [])

    def test_Pmod_merge_all(self):
        """Check merge_all behavior, but function has been inlined in _Pmod. """
        def merge_all(pmods):
            return ft.reduce(_Pmod._merge, pmods)

        pm1 = _Pmod(_steps={'a': _Pmod(alias='A')})
        pm2 = _Pmod(alias='pm2',
                    _regxs=[
                        ('b', _Pmod(alias='BB')),
                        ('a', _Pmod(alias='AA'))
                    ])
        pm3 = _Pmod(alias='PM3',
                    _steps={'c': _Pmod(alias='CCC'),
                            'b': _Pmod(alias='BBB'), },
                    _regxs=[
                        ('b', _Pmod(alias='AAA')),
                        ('a', _Pmod(alias='BBB')),
                        ('c', _Pmod(alias='CCC')),
                    ])
        pm = merge_all([pm1, pm2, pm3])

        self.assertSetEqual(set(pm._steps.keys()), set(list('abc')))
        self.assertSetEqual(set(pm1._steps.keys()), set(list('a')))
        self.assertEqual(pm2._steps, [])
        self.assertSetEqual(set(pm3._steps.keys()), set(list('bc')))

        self.assertEqual(self.pmod2regexstrs(pm), list('bac'))
        self.assertEqual(pm1._regxs, [])
        self.assertEqual(self.pmod2regexstrs(pm2), list('ba'))
        self.assertEqual(self.pmod2regexstrs(pm3), list('bac'))

    def test_indexing_None(self):
        pm = _Pmod()
        self.assertIsNone(pm['a'])

        pm = _Pmod(alias='a')
        self.assertIsNone(pm['a'])

        pm = _Pmod(_steps={'a': None})
        self.assertIsNone(pm['a'])

    def test_indexing(self):
        pm = _Pmod(
            _steps={'a':
                    _Pmod(alias='A', _steps={1: 11})},
            _regxs=[
                ('a\w*', _Pmod(alias='AWord', _steps={2: 22})),
                ('a\d*', _Pmod(alias='ADigit', _steps={3: 33})),
            ])

        # All children and regexps match.
        self.assertDictEqual(pm['a']._steps, {1: 11, 2: 22, 3: 33})

        # Only 'a\w*' matches.
        self.assertDictEqual(pm['aa']._steps, {2: 22})

        # Both regexps matches.
        self.assertDictEqual(pm['a1']._steps, {2: 22, 3: 33})


class TestPmods(unittest.TestCase):

    def test_convert_df_empty(self):
        df_orig = pd.DataFrame([])

        df = df_orig.copy()
        pmods = convert_df_as_pmods_tuples(df)
        self.assertEqual(pmods, [])
        npt.assert_array_equal(df, df_orig)

        df = df_orig.copy()
        pmods = convert_df_as_pmods_tuples(df, col_from='A', col_to='B')
        self.assertEqual(pmods, [])
        npt.assert_array_equal(df, df_orig)

    def test_convert_df_no_colnames(self):
        df_orig = pd.DataFrame([['/a/b', '/A/B']])

        df = df_orig.copy()
        pmods = convert_df_as_pmods_tuples(df)
        self.assertEqual(tuple(pmods[0]), ('/a/b', '/A/B'))
        npt.assert_array_equal(df, df_orig)

        df = df_orig.copy()
        pmods = convert_df_as_pmods_tuples(df, col_from='A', col_to='B')
        self.assertEqual(tuple(pmods[0]), ('/a/b', '/A/B'))
        npt.assert_array_equal(df, df_orig)

#     def test_build_pmods_names_orderedDict(self):
#         pmods_tuples = [
#             ('/a', 'A1/A2'),
#             ('/a/b', 'B'),
#         ]
#         pmods = build_pmods_from_tuples(pmods_tuples)
#         self.assertIsInstance(pmods, dict)
#         self.assertIsInstance(pmods[_PMOD_CHILD], OrderedDict)
#
#     def test_build_pmods_regex_orderedDict(self):
#         pmods_tuples = [
#             ('/a*', 'A1/A2'),
#             ('/a/b?', 'B'),
#         ]
#         pmods = build_pmods_from_tuples(pmods_tuples)
#         self.assertIsInstance(pmods, dict)
#         self.assertIsInstance(pmods[_PMOD_REGEX], OrderedDict)
#         self.assertIsInstance(pmods[_PMOD_CHILD], OrderedDict)
#         self.assertIsInstance(pmods[_PMOD_CHILD]['a'], OrderedDict)


class TestPstep(unittest.TestCase):

    def test_equality(self):
        p = Pstep()
        self.assertEqual(str(p), '.')
        self.assertEqual(p, Pstep('.'))
        self.assertEquals(str(p['.']), str(p))

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
        exp = [
            './abc/def',
            './abc/defg',
            './321',
            './n123',
        ]
        self.assertListEqual(sorted(p._paths), sorted(exp))

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

    def PMODS(self):
        return {
            '.': 'root',
            '_child_': {
                'a': 'b',
                'abc': 'BAR',
                'for': 'sub/path',
                '_child_': {
                    'def': 'DEF',
                    '_child_': {
                        '123': '234'
                    },
                }
            }
        }

    def test_pmods(self):
        p = Pstep(pmods={'MISS': 'BOO'})
        self.assertEquals(p, '.', (p, p._paths))
        p = Pstep('foo', pmods={'MISS': 'BOO'})
        self.assertEquals(p, 'foo', (p, p._paths))

        p = Pstep(pmods={'.': 'bar'})
        p.a
        self.assertEquals(p, 'bar', (p, p._paths))

        p = Pstep('root', pmods={'root': 'bar'})
        p.a
        self.assertEquals(p, 'bar', (p, p._paths))

        p = Pstep(pmods={'_child_': {'a': 'b'}})
        self.assertEquals(p.a, 'b', (p, p._paths))

        p = Pstep(pmods=self.PMODS())
        p.a
        p.abc['def']['123']
        self.assertListEqual(
            sorted(p._paths),
            sorted(['root/b', 'root/BAR/DEF/234']),
            (p, p._paths))

    def test_pmods_lock_not_applying(self):
        p = Pstep('not dot', pmods=self.PMODS())
        p.nota

        pmods = self.PMODS()
        pmods.pop('.')
        p = Pstep(pmods=pmods)

    def test_pmods_lock_CAN_RELOCATE(self):
        pmods = self.PMODS()
        pmods['.'] = 'deep/root'
        p = Pstep(pmods=pmods)
        p._lock = Pstep.CAN_RELOCATE
        p['for']._lock = Pstep.CAN_RELOCATE

    def test_pmods_lock_CAN_RENAME(self):
        pmods = self.PMODS()
        pmods['.'] = 'deep/root'
        p = Pstep(pmods=pmods)
        with self.assertRaises(ValueError, msg=p._paths):
            p._lock = Pstep.CAN_RENAME

        p = Pstep(pmods=self.PMODS())
        with self.assertRaises(ValueError, msg=p._paths):
            p['for']._lock = Pstep.CAN_RENAME

    def test_pmods_lock_LOCKED(self):
        p = Pstep(pmods=self.PMODS())
        with self.assertRaises(ValueError, msg=p._paths):
            p._lock = Pstep.LOCKED

        pmods = self.PMODS()
        pmods['.'] = 'deep/root'
        with self.assertRaises(ValueError, msg=p._paths):
            p._lock = Pstep.LOCKED

        pmods = self.PMODS()
        pmods.pop('.')
        p = Pstep(pmods=pmods)
        with self.assertRaises(ValueError, msg=p._paths):
            p.abc._lock = Pstep.LOCKED

        pmods = self.PMODS()
        pmods.pop('.')
        p = Pstep(pmods=pmods)
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


class TestComponents(unittest.TestCase):

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
