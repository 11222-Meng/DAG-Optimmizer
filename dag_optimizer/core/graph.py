import networkx as nx
from typing import Dict, List, Tuple, Optional
from ..models.operator import Operator


class DAG:
    """表示一个DAG图，包含节点和边"""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_operator(self, op: Operator, op_id: str):
        """添加一个算子节点"""
        self.graph.add_node(op_id, operator=op)

    def add_dependency(self, from_op_id: str, to_op_id: str):
        """添加依赖关系"""
        self.graph.add_edge(from_op_id, to_op_id)

    def get_operator(self, op_id: str) -> Operator:
        """获取指定ID的算子"""
        return self.graph.nodes[op_id]['operator']

    def get_all_operators(self) -> Dict[str, Operator]:
        """获取所有算子"""
        return {node: data['operator'] for node, data in self.graph.nodes(data=True)}

    def get_operator_count(self) -> int:
        """获取算子数量"""
        return len(self.graph.nodes)

    def get_predecessors(self, op_id: str) -> List[str]:
        """获取前驱节点"""
        return list(self.graph.predecessors(op_id))

    def get_successors(self, op_id: str) -> List[str]:
        """获取后继节点"""
        return list(self.graph.successors(op_id))

    def is_valid_cut_point(self, op_id: str) -> bool:
        """检查一个节点是否可以作为有效的切片点"""
        # 不能是源节点或汇节点
        if len(self.get_predecessors(op_id)) == 0 or len(self.get_successors(op_id)) == 0:
            return False
        return True

    def find_potential_cut_points(self) -> List[str]:
        """寻找所有潜在的切片点"""
        return [node for node in self.graph.nodes if self.is_valid_cut_point(node)]

    def visualize(self, filename: Optional[str] = None):
        """可视化DAG图"""
        import matplotlib.pyplot as plt

        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', node_size=700,
                edge_color='k', linewidths=1, font_size=10)

        if filename:
            plt.savefig(filename)
        else:
            plt.show()