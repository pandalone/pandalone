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
import unittest

import pandalone
from pandalone.mappings import pmods_from_tuples, Pmod
from pandalone.psteps import Pstep
from tests.test_utils import _init_logging


log = _init_logging(__name__)


class TestDoctest(unittest.TestCase):

    def test_doctests(self):
        failure_count, test_count = doctest.testmod(
            pandalone.psteps,
            optionflags=doctest.NORMALIZE_WHITESPACE)  # | doctest.ELLIPSIS)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))


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

    def test_paths_empty_rootstep(self):
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

    def test_paths_nonempty_rootstep(self):
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

    def test_pmods_nonempty_rootstep_rootunmapped(self):
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

    def test_pmods_map_number(self):
        pmods = pmods_from_tuples([('123', 'str'), (123, 'num')])

        p = Pstep(_pmod=pmods)
        p['123']
        p[123]
        self.assertEquals(
            sorted(p._paths), ['root/bar'], (p, sorted(p._paths)))
        p.a

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
