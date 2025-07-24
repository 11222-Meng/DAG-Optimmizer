import json
from pathlib import Path
from typing import Dict, List
from models.dag import DAG
from models.node import Node


class ConfigLoader:
    @staticmethod
    def load_dag_from_config(config_path: str) -> DAG:
        """
        从配置文件加载DAG结构
        :param config_path: 配置文件路径
        :return: 构建好的DAG对象
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        dag = DAG()

        # 添加所有节点
        for node_id, node_info in config['nodes'].items():
            node = Node(
                node_id=node_id,
                node_type=node_info['type'],
                output_tensor_size=node_info['output_size']
            )
            dag.add_node(node)

        # 添加所有边
        for edge in config['edges']:
            dag.add_edge(edge['from'], edge['to'])

        return dag

    @staticmethod
    def get_available_models() -> List[str]:
        """获取可用的模型配置列表"""
        config_dir = Path(__file__).parent.parent / 'configs'
        return [f.stem for f in config_dir.glob('*.json')]