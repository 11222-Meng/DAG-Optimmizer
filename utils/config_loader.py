import json
from pathlib import Path
from typing import List, Tuple
from models.dag import DAG
from models.node import Node


class ConfigLoader:
    @staticmethod
    def get_available_models() -> List[str]:
        """获取可用的模型配置列表"""
        config_dir = Path(__file__).parent.parent / 'configs'
        return [f.stem for f in config_dir.glob('*.json')]

    @staticmethod
    def load_dag_and_bottleneck(config_path: str) -> Tuple[DAG, str]:
        """
        从配置文件加载DAG和瓶颈类型
        返回: (DAG对象, 瓶颈类型)
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 创建DAG对象
        dag = DAG()
        for node_id, node_info in config['nodes'].items():
            dag.add_node(Node(
                node_id=node_id,
                node_type=node_info['type'],
                output_tensor_size=node_info['output_size']
            ))

        # 添加边
        for edge in config['edges']:
            dag.add_edge(edge['from'], edge['to'])

        # 提取瓶颈类型
        bottleneck_type = config.get('bottleneck', {}).get('type', 'unknown')
        if bottleneck_type == 'unknown':
            desc = config.get('description', '').lower()
            if 'memory' in desc:
                bottleneck_type = 'memory'
            elif 'flops' in desc:
                bottleneck_type = 'flops'
            elif 'latency' in desc:
                bottleneck_type = 'latency'

        return dag, bottleneck_type