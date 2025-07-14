import pytest
from dag_optimizer.core.scoring import ScoringSystem
from dag_optimizer.models.operator import OptimizationMetric
from .test_graph import sample_dag

def test_evaluate_cut_point(sample_dag):
    scoring = ScoringSystem(sample_dag.graph, OptimizationMetric.MEMORY)
    benefit, cost = scoring.evaluate_cut_point("op3")
    assert isinstance(benefit, float)
    assert isinstance(cost, float)