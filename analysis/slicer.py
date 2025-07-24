from typing import List, Tuple
from models.dag import DAG
from models.node import Node
from .scorer import Scorer
from utils.math_utils import MathUtils


class Slicer:
    def __init__(self, dag: DAG, alpha: float = 1.0):  # 添加alpha参数
        self.dag = dag
        self.scorer = Scorer(alpha=alpha)  # 将alpha传递给Scorer

    def find_all_edges_with_scores(self) -> List[Tuple[Node, Node, float]]:
        """返回所有边及其得分（未排序）"""
        return [
            (node, child, self.scorer.calculate_edge_score(node, child))
            for node in self.dag.get_all_nodes()
            for child in node.children
        ]

    def find_potential_slice_edges(self) -> List[Tuple[Node, Node, float]]:
        """找出所有边并计算得分"""
        edges_with_scores = []
        for node in self.dag.get_all_nodes():
            for child in node.children:
                score = self.scorer.calculate_edge_score(node, child)
                edges_with_scores.append((node, child, score))
        return edges_with_scores

    def select_top_k_slice_edges(self) -> List[Tuple[Node, Node, float]]:
        """选择得分最高的前k条边"""
        edges = self.find_potential_slice_edges()
        edges.sort(key=lambda x: x[2], reverse=True)  # 按score降序排序

        k = MathUtils.floor_sqrt(len(edges))
        return edges[:k]