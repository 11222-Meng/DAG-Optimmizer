from typing import Dict, List
from networkx import DiGraph
import networkx as nx
from ..models.operator import Operator, OptimizationMetric


class MetricsCalculator:
    """计算DAG的各种指标"""

    @staticmethod
    def calculate_path_metrics(graph: DiGraph, path: List[str], metric: OptimizationMetric) -> float:
        """计算路径上指定指标的总和"""
        total = 0.0
        for node in path:
            op = graph.nodes[node]['operator']
            total += op.get_metric(metric)
        return total

    @staticmethod
    def calculate_critical_path(graph: DiGraph, metric: OptimizationMetric) -> List[str]:
        """计算关键路径"""
        # 使用动态规划方法计算最长路径(关键路径)
        longest_path = {}
        predecessors = {}

        # 拓扑排序
        topo_order = list(nx.topological_sort(graph))

        for node in topo_order:
            op = graph.nodes[node]['operator']
            current_metric = op.get_metric(metric)

            max_pred_metric = 0.0
            best_pred = None
            for pred in graph.predecessors(node):
                pred_metric = longest_path.get(pred, 0.0)
                if pred_metric > max_pred_metric:
                    max_pred_metric = pred_metric
                    best_pred = pred

            longest_path[node] = current_metric + max_pred_metric
            predecessors[node] = best_pred

        # 回溯找到关键路径
        end_node = max(longest_path, key=longest_path.get)
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = predecessors.get(current)

        return path[::-1]  # 反转得到从开始到结束的路径

    @staticmethod
    def calculate_total_metric(graph: DiGraph, metric: OptimizationMetric) -> float:
        """计算整个DAG在指定指标上的总和"""
        total = 0.0
        for node in graph.nodes:
            op = graph.nodes[node]['operator']
            total += op.get_metric(metric)
        return total