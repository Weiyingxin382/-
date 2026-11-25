import unittest

from src.analysis.drain_core import Drain


class DrainTests(unittest.TestCase):
    def test_template_generalization_and_length_routing(self):
        drain = Drain()
        first = drain.add("VM alpha started")
        self.assertEqual(drain.add("VM beta started"), first)
        self.assertEqual(drain.templates()[first], "VM <*> started")
        self.assertNotEqual(drain.add("VM alpha has stopped"), first)
        self.assertEqual(drain.clusters[first].count, 2)

    def test_numeric_prefix_and_empty_messages(self):
        drain = Drain()
        self.assertEqual(drain.add("12 bytes received"), drain.add("24 bytes received"))
        empty = drain.add("")
        self.assertEqual(empty, drain.add(""))

    def test_status_can_be_generalized_despite_semantic_difference(self):
        drain = Drain()
        key = drain.add("GET request status: 200")
        self.assertEqual(drain.add("GET request status: 503"), key)
        self.assertEqual(drain.templates()[key], "GET request status: <*>")
