import unittest
from models.dag import DAG
from models.node import Node


class TestDAG(unittest.TestCase):
    def setUp(self):
        self.dag = DAG()
        self.node1 = Node("node1", "Conv2d", 100)
        self.node2 = Node("node2", "ReLU", 100)
        self.dag.add_node(self.node1)
        self.dag.add_node(self.node2)

    def test_add_node(self):
        self.assertEqual(len(self.dag.nodes), 2)
        self.assertIn("node1", self.dag.nodes)

    def test_add_edge(self):
        self.dag.add_edge("node1", "node2")
        self.assertIn(self.node2, self.node1.children)
        self.assertIn(self.node1, self.node2.parents)

    def test_topological_sort(self):
        node3 = Node("node3", "MaxPool2d", 80)
        self.dag.add_node(node3)
        self.dag.add_edge("node1", "node2")
        self.dag.add_edge("node2", "node3")

        sorted_nodes = self.dag.topological_sort()
        self.assertEqual(sorted_nodes[0].id, "node1")
        self.assertEqual(sorted_nodes[1].id, "node2")
        self.assertEqual(sorted_nodes[2].id, "node3")


if __name__ == "__main__":
    unittest.main()