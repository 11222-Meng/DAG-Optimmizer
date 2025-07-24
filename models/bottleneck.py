class BottleneckAnalyzer:
    @staticmethod
    def get_bottleneck_score(node_type: str) -> float:
        """
        根据节点类型获取瓶颈评分
        评分越高表示该节点作为瓶颈的可能性越大
        """
        bottleneck_scores = {
            'Conv2d': 1.0,
            'MaxPool2d': 0.8,
            'BatchNorm2d': 0.6,
            'ReLU': 0.3,
            'Linear': 0.7,
            'AvgPool2d': 0.5,
            'AdaptiveAvgPool2d': 0.4,
            'InvertedResidual': 0.9,
            'FireModule': 0.8
        }
        return bottleneck_scores.get(node_type, 0.2)