#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

from __future__ import division, print_function, unicode_literals

import logging
import os
import sys
import tempfile
import doctest
import unittest


import pandalone.dispatcher as dsp

import networkx as nx



DEFAULT_LOG_LEVEL = logging.INFO


def _init_logging(loglevel):
    logging.basicConfig(level=loglevel)
    logging.getLogger().setLevel(level=loglevel)

    log = logging.getLogger(__name__)
    log.trace = lambda *args, **kws: log.log(0, *args, **kws)

    return log
log = _init_logging(DEFAULT_LOG_LEVEL)


def from_my_path(*parts):
    return os.path.join(os.path.dirname(__file__), *parts)


def _make_sample_workbook(path, matrix, sheet_name, startrow=0, startcol=0):
    df = pd.DataFrame(matrix)
    writer = pd.ExcelWriter(path)
    df.to_excel(writer, sheet_name, startrow=startrow, startcol=startcol)
    writer.save()


class TestDoctest(unittest.TestCase):

    def runTest(self):
        failure_count, test_count = doctest.testmod(
            dsp, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEquals(failure_count, 0, (failure_count, test_count))


class TestDispatcher(unittest.TestCase):

    def test_add_data(self):
        d = dsp.Dispatcher()

        self.assertEquals(dsp.add_data(d, data_id='/a'), '/a')
        self.assertEquals(dsp.add_data(d, data_id='/a'), '/a')

        self.assertEquals(dsp.add_data(d), 0)
        self.assertEquals(dsp.add_data(d, default_value='v'), 1)

        self.assertEquals(d.graph.node[1], {'wait_inputs': False,
                                            'type': 'data'})

        self.assertEquals(d.default_values[1], 'v')
        self.assertEquals(dsp.add_data(d, data_id=1), 1)
        self.assertFalse(1 in d.default_values)
        dsp.add_data(d,
                     data_id='/a',
                     wait_inputs=False,
                     function=lambda:None,
                     callback=lambda:None,
                     wait_exceptions=[d.start])

        attr = d.graph.node['/a']
        res = ['callback', 'function', 'wait_exceptions', 'wait_inputs', 'type']
        self.assertEquals(set(attr.keys()), set(res))


class TestGraphAlgorithms(unittest.TestCase):

    def test_scc_fun(self):
        graph = nx.DiGraph()
        graph.add_cycle([1, 2, 3, 4])
        graph.add_cycle([5, 6, 7, 8])
        graph.add_node(0)
        graph.add_edge(10, 9)

        res = [[1, 2, 3, 4], [9], [10]]
        self.assertEquals(list(dsp.scc_fun(graph, [1,10])), res)

        res = [[0], [1, 2, 3, 4], [5, 6, 7, 8], [9], [10]]
        self.assertEquals(list(dsp.scc_fun(graph)), res)

        res = [[1, 2, 3, 4]]
        self.assertEquals(list(dsp.scc_fun(graph, [1])), res)

    def test_dijkstra(self):
        graph = nx.DiGraph()
        graph.add_cycle([1, 2, 3, 4])
        graph.add_cycle([5, 6, 7, 8])
        graph.add_node(0)
        graph.add_edge(9, 10)
        graph.add_edge(3, 9)
        graph.add_edge(10, 7)

        dist, paths = dsp.dijkstra(graph, 1)
        res = {1: 0, 2: 1, 3: 2, 4: 3, 5: 7,
               6: 8, 7: 5, 8: 6, 9: 3, 10: 4}
        self.assertEquals(dist, res)
        res = {1: [1],
               2: [1, 2],
               3: [1, 2, 3],
               4: [1, 2, 3, 4],
               5: [1, 2, 3, 9, 10, 7, 8, 5],
               6: [1, 2, 3, 9, 10, 7, 8, 5, 6],
               7: [1, 2, 3, 9, 10, 7],
               8: [1, 2, 3, 9, 10, 7, 8],
               9: [1, 2, 3, 9],
               10: [1, 2, 3, 9, 10]}
        self.assertEqual(paths, res)

        res = {1: 0, 2: 1, 3: 2, 4: 3, 9: 3}
        dist, paths = dsp.dijkstra(graph, 1, [4])
        self.assertEqual(dist, res)
        res = {1: [1],
               2: [1, 2],
               3: [1, 2, 3],
               4: [1, 2, 3, 4],
               9: [1, 2, 3, 9],
               10: [1, 2, 3, 9, 10]}
        self.assertEquals(paths, res)

        res = {1: 0, 2: 1, 3: 2, 4: 3, 9: 3, 10: 4}
        dist, paths = dsp.dijkstra(graph, 1, [10])
        self.assertEquals(dist, res)
        res = {1: [1],
               2: [1, 2],
               3: [1, 2, 3],
               4: [1, 2, 3, 4],
               9: [1, 2, 3, 9],
               10: [1, 2, 3, 9, 10]}
        self.assertEquals(paths, res)

        res = {1: 0, 2: 1, 3: 2, 4: 3, 7: 5, 9: 3, 10: 4}
        dist, paths = dsp.dijkstra(graph, 1, [4, 7])
        self.assertEquals(dist, res)
        res = {1: [1],
               2: [1, 2],
               3: [1, 2, 3],
               4: [1, 2, 3, 4],
               7: [1, 2, 3, 9, 10, 7],
               9: [1, 2, 3, 9],
               10: [1, 2, 3, 9, 10]}
        self.assertEquals(paths, res)
