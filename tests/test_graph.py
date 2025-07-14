import pytest
from dag_optimizer.core.graph import DAG
from dag_optimizer.models.operator import Operator


@pytest.fixture
def sample_dag():
    """创建一个示例DAG用于测试"""
    dag = DAG()

    # 添加算子
    ops = [
        Operator("conv", memory=10, computation=5, latency=2),
        Operator("pool", memory=5, computation=2, latency=1),
        Operator("fc", memory=20, computation=10, latency=5),
        Operator("relu", memory=2, computation=1, latency=0.5),
        Operator("softmax", memory=3, computation=3, latency=1)
    ]

    for i, op in enumerate(ops):
        dag.add_operator(op, f"op{i + 1}")

    # 添加依赖关系
    dag.add_dependency("op1", "op2")
    dag.add_dependency("op2", "op3")
    dag.add_dependency("op3", "op4")
    dag.add_dependency("op4", "op5")

    return dag


def test_add_operator(sample_dag):
    assert sample_dag.get_operator_count() == 5
    assert sample_dag.get_operator("op1").op_type == "conv"


def test_find_potential_cut_points(sample_dag):
    cut_points = sample_dag.find_potential_cut_points()
    assert set(cut_points) == {"op2", "op3", "op4"}