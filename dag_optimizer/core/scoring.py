from typing import Dict, Tuple, List
from networkx import DiGraph
import networkx as nx
from dag_optimizer.models.operator import OptimizationMetric
from dag_optimizer.core.metrics import MetricsCalculator  # 添加这行


class ScoringSystem:
    """评估切片点的收益和消耗"""

    def __init__(self, graph: DiGraph, target_metric: OptimizationMetric):
        self.graph = graph
        self.target_metric = target_metric
        self.metrics_calculator = MetricsCalculator()  # 现在应该可以正常工作了

    def evaluate_cut_point(self, cut_point: str) -> Tuple[float, float]:
        """
        评估一个切片点的收益和消耗
        返回: (收益分数, 消耗分数)
        """
        # 计算切片前的关键路径指标
        original_critical_path = self.metrics_calculator.calculate_critical_path(
            self.graph, self.target_metric)
        original_metric = self.metrics_calculator.calculate_path_metrics(
            self.graph, original_critical_path, self.target_metric)

        # 创建两个子图
        subgraph1, subgraph2 = self._split_graph_at_cut_point(cut_point)

        # 计算两个子图的关键路径指标
        cp1 = self.metrics_calculator.calculate_critical_path(subgraph1, self.target_metric)
        cp2 = self.metrics_calculator.calculate_critical_path(subgraph2, self.target_metric)

        metric1 = self.metrics_calculator.calculate_path_metrics(subgraph1, cp1, self.target_metric)
        metric2 = self.metrics_calculator.calculate_path_metrics(subgraph2, cp2, self.target_metric)

        # 收益: 原始关键路径指标 - max(子图1关键路径, 子图2关键路径)
        benefit = original_metric - max(metric1, metric2)

        # 消耗: 主要是通信开销，这里简化为1 (实际中可以更复杂)
        cost = 1.0

        return benefit, cost

    def _split_graph_at_cut_point(self, cut_point: str) -> Tuple[DiGraph, DiGraph]:
        """在切片点处分割DAG"""
        # 获取切片点后的所有节点
        successors = set(nx.descendants(self.graph, cut_point))
        successors.add(cut_point)

        # 创建两个子图
        subgraph1 = self.graph.subgraph(set(self.graph.nodes) - successors).copy()
        subgraph2 = self.graph.subgraph(successors).copy()

        return subgraph1, subgraph2

    def score_cut_points(self, cut_points: List[str]) -> Dict[str, Tuple[float, float]]:
        """为多个切片点打分"""
        scores = {}
        for cp in cut_points:
            benefit, cost = self.evaluate_cut_point(cp)
            scores[cp] = (benefit, cost)
        return scores