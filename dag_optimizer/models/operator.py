from enum import Enum
from dataclasses import dataclass
from typing import Dict


class OptimizationMetric(Enum):
    MEMORY = "memory"
    COMPUTATION = "computation"
    LATENCY = "latency"


# 算子类型的默认指标值
OPERATOR_DEFAULTS = {
    "Conv2d": {"memory": 20, "computation": 15, "latency": 5},
    "BatchNorm2d": {"memory": 5, "computation": 2, "latency": 1},
    "ReLU": {"memory": 1, "computation": 1, "latency": 0.5},
    "MaxPool2d": {"memory": 8, "computation": 3, "latency": 2},
    "AvgPool2d": {"memory": 8, "computation": 3, "latency": 2},
    "AdaptiveAvgPool2d": {"memory": 6, "computation": 2, "latency": 1.5},
    "Linear": {"memory": 25, "computation": 10, "latency": 4},
    "InvertedResidual": {"memory": 15, "computation": 8, "latency": 3},
    "FireModule": {"memory": 18, "computation": 12, "latency": 4}
}


@dataclass
class Operator:
    """表示DAG中的一个算子节点"""
    op_type: str
    name: str = ""  # 算子名称
    memory: float = 0.0
    computation: float = 0.0
    latency: float = 0.0

    def __post_init__(self):
        """如果未指定指标值，使用默认值"""
        defaults = OPERATOR_DEFAULTS.get(self.op_type, {})
        self.memory = self.memory or defaults.get("memory", 0)
        self.computation = self.computation or defaults.get("computation", 0)
        self.latency = self.latency or defaults.get("latency", 0)

    def get_metric(self, metric: OptimizationMetric) -> float:
        """获取指定指标的值"""
        if metric == OptimizationMetric.MEMORY:
            return self.memory
        elif metric == OptimizationMetric.COMPUTATION:
            return self.computation
        elif metric == OptimizationMetric.LATENCY:
            return self.latency
        raise ValueError(f"Unknown metric: {metric}")