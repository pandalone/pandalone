#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, print_function, unicode_literals

import doctest
import json
import unittest

import numpy as np
import numpy.testing as npt
from pandalone import pandata
from pandalone.pandata import JSONCodec, Pstep
from jsonschema.exceptions import RefResolutionError
import pandas as pd


class TestDoctest(unittest.TestCase):

    def runTest(self):
        failure_count, test_count = doctest.testmod(pandata)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))


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

        p = Pstep(pmods={'_cpmods': {'a': 'b'}})
        self.assertEquals(p.a, 'b', (p, p._paths))

        p = Pstep(pmods={'_cpmods': {
            'a': 'b',
            'abc': 'BAR', '_cpmods': {
                'def': 'DEF', '_cpmods': {
                    '123': '234'
                },
            }
        }})
        p.a
        p.abc['def']['123']
        self.assertListEqual(sorted(p._paths), sorted(['./b', './BAR/DEF/234']),
                             (p, p._paths))

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

    def test_json(self):
        json.dumps(Pstep())
        json.dumps({Pstep(): 1})

    @unittest.skip('Unknwon why sets fail with Pstep!')
    def test_json_sets(self):
        json.dumps({Pstep(), Pstep()})


class TestJSONCodec(unittest.TestCase):

    def test_lists(self):
        o = [1, 2, '3', 'fdgdg', None, []]
        s = json.dumps(o, cls=JSONCodec.Encoder)
        oo = json.loads(s, cls=pandata.JSONCodec.Decoder)
        self.assertEqual(oo, o)

    def test_dict(self):
        o = {'a': 1, 2: 'bb', 3: [], 4: None}
        s = json.dumps(o, cls=JSONCodec.Encoder)
        oo = json.loads(s, cls=pandata.JSONCodec.Decoder)
        self.assertEqual(oo, o)

    def test_Numpy(self):
        o = np.random.randn(6, 2)
        s = json.dumps(o, cls=JSONCodec.Encoder)
        oo = json.loads(s, cls=pandata.JSONCodec.Decoder)
        npt.assert_array_equal(oo, o)

    def test_DataFrame(self):
        o = pd.DataFrame(np.random.randn(10, 2))
        s = json.dumps(o, cls=JSONCodec.Encoder)
        oo = json.loads(s, cls=pandata.JSONCodec.Decoder)
        npt.assert_array_equal(oo, o)

    def test_Series(self):
        o = pd.Series({'a': 1, 2: 22})
        s = json.dumps(o, cls=JSONCodec.Encoder)
        oo = json.loads(s, cls=pandata.JSONCodec.Decoder)
        npt.assert_array_equal(oo, o)

    @unittest.skip(('TODO: Cannot test for recursive-equality with pandas.'))
    def test_mix(self):
        obj_list = [3.14,
                    {
                        'aa': pd.DataFrame([]),
                        2: np.array([]),
                        33: {'foo': 'bar'},
                    },
                    pd.DataFrame(np.random.randn(10, 2)),
                    ('b', pd.Series({})),
                    ]
        for o in obj_list + [obj_list]:
            s = json.dumps(o, cls=JSONCodec.Encoder)
            oo = json.loads(s, cls=pandata.JSONCodec.Decoder)
            self.assertTrue(pandata.trees_equal(o, oo), (o, oo))


class TestJsonPointer(unittest.TestCase):

    def test_resolve_jsonpointer_existing(self):
        doc = {
            'foo': 1,
            'bar': [11, {'a': 222}]
        }

        path = '/foo'
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), 1)

        path = '/bar/0'
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), 11)

        path = '/bar/1/a'
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), 222)

    def test_resolve_jsonpointer_sequence(self):
        doc = [1, [22, 33]]

        path = '/0'
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), 1)

        path = '/1'
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), [22, 33])

        path = '/1/0'
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), 22)
        path = '/1/1'
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), 33)

    def test_resolve_jsonpointer_missing_screams(self):
        doc = {}

        path = '/foo'
        with self.assertRaises(RefResolutionError):
            pandata.resolve_jsonpointer(doc, path)

    def test_resolve_jsonpointer_empty_path(self):
        doc = {}
        path = ''
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), doc)

        doc = {'foo': 1}
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), doc)

    def test_resolve_jsonpointer_examples_from_spec(self):
        def _doc():
            return {
                r"foo": ["bar", r"baz"],
                r"": 0,
                r"a/b": 1,
                r"c%d": 2,
                r"e^f": 3,
                r"g|h": 4,
                r"i\\j": 5,
                r"k\"l": 6,
                r" ": 7,
                r"m~n": 8
            }
        cases = [
            (r"",            _doc()),
            (r"/foo",        ["bar", "baz"]),
            (r"/foo/0",      "bar"),
            (r"/",           0),
            (r"/a~1b",       1),
            (r"/c%d",      2),
            (r"/e^f",      3),
            (r"/g|h",      4),
            (r"/i\\j",      5),
            (r"/k\"l",      6),
            (r"/ ",        7),
            (r"/m~0n",       8),
        ]
        for path, exp in cases:
            doc = _doc()
            self.assertEqual(pandata.resolve_jsonpointer(doc, path), exp)

    def test_set_jsonpointer_empty_doc(self):
        doc = {}
        path = '/foo'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)

        doc = {}
        path = '/foo/bar'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)

    def test_resolve_jsonpointer_root_path_only(self):
        doc = {}
        path = '/'
        with self.assertRaises(RefResolutionError):
            self.assertEqual(pandata.resolve_jsonpointer(doc, path), doc)

        doc = {'foo': 1}
        with self.assertRaises(RefResolutionError):
            self.assertEqual(pandata.resolve_jsonpointer(doc, path), doc)

        doc = {'': 1}
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), doc[''])

    def test_set_jsonpointer_replace_value(self):
        doc = {'foo': 'bar', 1: 2}
        path = '/foo'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)
        self.assertEqual(doc[1], 2)

        doc = {'foo': 1, 1: 2}
        path = '/foo'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)
        self.assertEqual(doc[1], 2)

        doc = {'foo': {'bar': 1}, 1: 2}
        path = '/foo'
        value = 2
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)
        self.assertEqual(doc[1], 2)

    def test_set_jsonpointer_append_path(self):
        doc = {'foo': 'bar', 1: 2}
        path = '/foo/bar'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)
        self.assertEqual(doc[1], 2)

        doc = {'foo': 'bar', 1: 2}
        path = '/foo/bar/some/other'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)
        self.assertEqual(doc[1], 2)

    def test_set_jsonpointer_append_path_preserves_intermediate(self):
        doc = {'foo': {'bar': 1}, 1: 2}
        path = '/foo/foo2'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        print(doc)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)
        self.assertEqual(doc[1], 2)
        self.assertEqual(pandata.resolve_jsonpointer(doc, '/foo/bar'), 1)

    def test_set_jsonpointer_missing(self):
        doc = {'foo': 1, 1: 2}
        path = '/foo/bar'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)
        self.assertEqual(doc[1], 2)

        doc = {'foo': 1, 1: 2}
        path = '/foo/bar/some/other'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)
        self.assertEqual(doc[1], 2)

    def test_set_jsonpointer_sequence(self):
        doc = [1, 2]
        path = '/1'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)

        doc = [1, 2]
        path = '/1/foo/bar'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)

    def test_set_jsonpointer_sequence_insert_end(self):
        doc = [0, 1]
        path = '/2'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, path), value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, '/0'), 0)
        self.assertEqual(pandata.resolve_jsonpointer(doc, '/1'), 1)

        doc = [0, 1]
        path = '/-'
        value = 'value'
        pandata.set_jsonpointer(doc, path, value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, '/2'), value)
        self.assertEqual(pandata.resolve_jsonpointer(doc, '/0'), 0)
        self.assertEqual(pandata.resolve_jsonpointer(doc, '/1'), 1)

    def test_set_jsonpointer_sequence_out_of_bounds(self):
        doc = [0, 1]
        path = '/3'
        value = 'value'
        with self.assertRaises(RefResolutionError):
            pandata.set_jsonpointer(doc, path, value)

    def test_set_jsonpointer_sequence_with_str_screams(self):
        doc = [0, 1]
        path = '/str'
        value = 'value'
        with self.assertRaises(RefResolutionError):
            pandata.set_jsonpointer(doc, path, value)

    def test_build_all_jsonpaths(self):
        from jsonschema._utils import load_schema
        schema = dict(properties=dict(a=dict(properties=dict(b={}))))
        paths = pandata.build_all_jsonpaths(schema)
        print('\n'.join(paths))
        # TODO: build and check_all_paths support $ref
        self.assertIn('/a/b', paths)
