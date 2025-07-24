from typing import Dict, List
from .node import Node


class DAG:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.root_nodes: List[Node] = []

    def add_node(self, node: Node):
        """添加节点到DAG"""
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists in DAG")
        self.nodes[node.id] = node
        if not node.parents:
            self.root_nodes.append(node)

    def add_edge(self, from_node_id: str, to_node_id: str):
        """添加边到DAG"""
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("One or both nodes not found in DAG")

        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]

        from_node.add_child(to_node)
        to_node.add_parent(from_node)

    def get_node(self, node_id: str) -> Node:
        """获取指定ID的节点"""
        return self.nodes.get(node_id)

    def get_all_nodes(self) -> List[Node]:
        """获取所有节点"""
        return list(self.nodes.values())

    def topological_sort(self) -> List[Node]:
        """拓扑排序"""
        in_degree = {node.id: 0 for node in self.nodes.values()}
        for node in self.nodes.values():
            for child in node.children:
                in_degree[child.id] += 1

        queue = [node for node in self.nodes.values() if in_degree[node.id] == 0]
        sorted_nodes = []

        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)

            for child in node.children:
                in_degree[child.id] -= 1
                if in_degree[child.id] == 0:
                    queue.append(child)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("DAG contains cycles")

        return sorted_nodes