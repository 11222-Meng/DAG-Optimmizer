from typing import List, Dict, Tuple
import math
from networkx import DiGraph
from ..models.operator import OptimizationMetric
from .scoring import ScoringSystem  # 添加这行
from .metrics import MetricsCalculator  # 添加这行
from .graph import DAG  # 添加这行
class DAGOptimizer:
    """DAG优化器，负责选择最佳切片点"""

    def __init__(self, graph: DiGraph, target_metric: OptimizationMetric):
        self.graph = graph
        self.target_metric = target_metric
        self.scoring_system = ScoringSystem(graph, target_metric)
        self.dag = DAG()
        self.dag.graph = graph

    def determine_k(self) -> int:
        """更保守的k值计算，确保小DAG至少有一个切点"""
        n = self.dag.get_operator_count()
        return max(1, int(math.sqrt(n)) ) # 至少选1个

    def find_optimal_cut_points(self) -> List[Tuple[str, float, float]]:
        """寻找最优的切片点"""
        potential_cut_points = self.dag.find_potential_cut_points()
        if not potential_cut_points:
            return []

        # 评估所有潜在切片点
        scores = self.scoring_system.score_cut_points(potential_cut_points)

        # 按收益/消耗比排序
        sorted_points = sorted(
            scores.items(),
            key=lambda x: x[1][0] / x[1][1],  # 按收益/消耗比排序
            reverse=True
        )

        # 选择前k个
        k = self.determine_k()
        selected = sorted_points[:k]

        # 返回格式: [(op_id, benefit, cost), ...]
        return [(item[0], item[1][0], item[1][1]) for item in selected]

    def optimize(self) -> Dict:
        """执行优化并返回结果"""
        optimal_cut_points = self.find_optimal_cut_points()

        return {
            "target_metric": self.target_metric.value,
            "total_operators": self.dag.get_operator_count(),
            "k": self.determine_k(),
            "optimal_cut_points": optimal_cut_points,
            "original_critical_path": MetricsCalculator().calculate_critical_path(
                self.graph, self.target_metric),
            "original_metric": MetricsCalculator().calculate_path_metrics(
                self.graph,
                MetricsCalculator().calculate_critical_path(self.graph, self.target_metric),
                self.target_metric
            )
        }