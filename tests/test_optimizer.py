import pytest
from dag_optimizer.core.optimizer import DAGOptimizer
from dag_optimizer.models.operator import OptimizationMetric
from .test_graph import sample_dag

def test_determine_k(sample_dag):
    optimizer = DAGOptimizer(sample_dag.graph, OptimizationMetric.MEMORY)
    assert optimizer.determine_k() == 2  # sqrt(5) ≈ 2.23 -> 2

def test_find_optimal_cut_points(sample_dag):
    optimizer = DAGOptimizer(sample_dag.graph, OptimizationMetric.MEMORY)
    cut_points = optimizer.find_optimal_cut_points()
    assert len(cut_points) == 2
    # 具体哪个节点会被选中取决于评分系统