from typing import Dict
from models.node import Node
from models.bottleneck import BottleneckAnalyzer

class Scorer:
    def __init__(self, alpha: float = 1.0):  # 确保接受alpha参数
        self.alpha = alpha

    def calculate_edge_score(self, from_node: Node, to_node: Node) -> float:
        """计算边的得分: score = gain(to_node) - α * cost(from_node)"""
        gain = self._calculate_gain(to_node)
        cost = self._calculate_cost(from_node)
        return gain - self.alpha * cost

    def _calculate_gain(self, node: Node) -> float:
        """计算下游节点的增益"""
        return node.output_tensor_size * BottleneckAnalyzer.get_bottleneck_score(node.type)

    def _calculate_cost(self, node: Node) -> float:
        """计算上游节点的传输成本"""
        return node.output_tensor_size