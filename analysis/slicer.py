from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from models.dag import DAG
from models.node import Node
from .scorer import Scorer
from utils.math_utils import MathUtils
from typing import List, Optional
import os
import matplotlib.pyplot as plt
import networkx as nx
import hashlib
import json


class Slicer:
    def __init__(self, dag: DAG, alpha: float = 1.0):
        self.dag = dag
        self.scorer = Scorer(alpha=alpha)

    def find_potential_slice_edges(self) -> List[Tuple[Node, Node, float]]:
        """获取所有边及其得分"""
        edges = []
        for node in self.dag.get_all_nodes():
            for child in node.children:
                score = self.scorer.calculate_edge_score(node, child)
                edges.append((node, child, score))
        return edges

    def select_top_k_slice_edges(self) -> List[Tuple[Node, Node, float]]:
        """选择得分最高的前k条边"""
        edges = self.find_potential_slice_edges()
        edges.sort(key=lambda x: x[2], reverse=True)
        k = MathUtils.floor_sqrt(len(edges))
        return edges[:k]

    def split_into_subgraphs(self, slice_edges: List[Tuple[Node, Node, float]]) -> List[nx.DiGraph]:
        """改进的子图划分方法"""
        G = nx.DiGraph()
        node_mapping = {node.id: node for node in self.dag.get_all_nodes()}

        # 构建完整图
        for node in self.dag.get_all_nodes():
            G.add_node(node.id, type=node.type, size=node.output_tensor_size)
            for child in node.children:
                G.add_edge(node.id, child.id)

        # 标记要移除的边
        edges_to_remove = [(from_node.id, to_node.id) for from_node, to_node, _ in slice_edges]

        # 确保至少保留一个连通分量
        if len(edges_to_remove) >= len(G.edges):
            edges_to_remove = edges_to_remove[:len(G.edges) - 1]

        # 移除边
        G.remove_edges_from(edges_to_remove)

        # 获取连通分量
        subgraphs = []
        for c in nx.weakly_connected_components(G):
            subgraph = G.subgraph(c).copy()

            # 过滤掉空子图
            if len(subgraph.nodes) > 0:
                subgraphs.append(subgraph)

        return subgraphs

    def visualize_subgraphs(
            self,
            subgraphs: List[nx.DiGraph],
            model_name: str,
            bottleneck_type: str,
            save_dir: str = "dag_visualizations"
    ) -> Dict[str, List[str]]:
        """
        完整的可视化保存方法（包含去重、元数据保存和类型校验）

        Args:
            subgraphs: 子图列表（NetworkX图对象）
            model_name: 模型名称（如'mobilenet_v2'）
            bottleneck_type: 瓶颈类型（'memory'/'flops'/'latency'）
            save_dir: 保存根目录（默认'dag_visualizations'）

        Returns:
            {
                "images": ["路径/subgraph_1.png", ...],
                "metadata": ["路径/subgraph_1.json", ...]
            }
        """
        # 初始化返回结构和目录
        result = {"images": [], "metadata": []}
        os.makedirs(save_dir, exist_ok=True)

        # 创建模型专属子目录（格式：模型名_瓶颈类型）
        model_dir = os.path.join(save_dir, f"{model_name}_{bottleneck_type}")
        os.makedirs(model_dir, exist_ok=True)

        # 校验合法的节点类型并设置颜色映射
        type_colors = {
            "Conv2d": "#FF6B6B",  # 红色-卷积层
            "BatchNorm2d": "#FFE66D",  # 黄色-BN层
            "ReLU": "#4ECDC4",  # 青色-激活层
            "MaxPool2d": "#A5D8FF",  # 蓝色-池化层
            "Linear": "#C8A2C8",  # 紫色-全连接
            "default": "#C8C8C8"  # 灰色-其他类型
        }

        # 用于检测重复子图的哈希集合
        unique_hashes = set()

        for i, subgraph in enumerate(subgraphs, 1):
            # 计算子图哈希值（基于节点和边的特征）
            graph_hash = self._get_graph_hash(subgraph)
            if graph_hash in unique_hashes:
                print(f"⏩ 跳过重复子图 {i} (哈希: {graph_hash[:8]}...)")
                continue
            unique_hashes.add(graph_hash)

            # 创建图形
            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(subgraph, seed=42, k=0.8)

            # 绘制节点（自动处理未知类型）
            node_colors = [
                type_colors.get(subgraph.nodes[n]['type'], type_colors['default'])
                for n in subgraph.nodes
            ]
            nx.draw_networkx_nodes(
                subgraph, pos,
                node_color=node_colors,
                node_size=2500,
                edgecolors='black',
                linewidths=1.5,
                alpha=0.9
            )

            # 绘制边
            nx.draw_networkx_edges(
                subgraph, pos,
                arrowstyle='->',
                arrowsize=25,
                width=2,
                edge_color='#555555',
                alpha=0.7
            )

            # 添加节点标签
            node_labels = {
                n: f"{n}\nType: {subgraph.nodes[n]['type']}\nSize: {subgraph.nodes[n]['size']}"
                for n in subgraph.nodes
            }
            nx.draw_networkx_labels(
                subgraph, pos,
                labels=node_labels,
                font_size=10,
                font_family='sans-serif',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5)
            )

            # 添加边标签（显示数据流大小）
            edge_labels = {
                (u, v): f"{subgraph.nodes[u]['size']}→{subgraph.nodes[v]['size']}"
                for u, v in subgraph.edges
            }
            nx.draw_networkx_edge_labels(
                subgraph, pos,
                edge_labels=edge_labels,
                font_size=9,
                label_pos=0.5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )

            # 设置标题
            plt.title(
                f"{model_name} - {bottleneck_type.upper()} Bottleneck\n"
                f"Subgraph {len(unique_hashes)} | "
                f"Nodes: {len(subgraph.nodes)} | "
                f"Edges: {len(subgraph.edges)}",
                pad=20,
                fontsize=12
            )
            plt.axis('off')
            plt.tight_layout()

            # 保存图片
            img_path = os.path.join(model_dir, f"subgraph_{len(unique_hashes)}.png")
            plt.savefig(img_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close()
            result["images"].append(img_path)

            # 保存元数据
            meta_path = os.path.join(model_dir, f"subgraph_{len(unique_hashes)}.json")
            self._save_subgraph_metadata(meta_path, subgraph)
            result["metadata"].append(meta_path)

            print(f"✅ 已保存子图 {i} -> {img_path}")

        return result

    # 必需的辅助方法（需在同一个class中）
    def _get_graph_hash(self, graph: nx.DiGraph) -> str:
        """生成子图唯一哈希值"""
        import hashlib
        import json

        graph_data = {
            "nodes": sorted(
                (n, graph.nodes[n]['type'], graph.nodes[n]['size'])
                for n in graph.nodes
            ),
            "edges": sorted(
                (u, v, graph.nodes[u]['type'], graph.nodes[v]['type'])
                for u, v in graph.edges
            )
        }
        return hashlib.md5(
            json.dumps(graph_data, sort_keys=True).encode('utf-8')
        ).hexdigest()

    def _save_subgraph_metadata(self, path: str, graph: nx.DiGraph):
        """保存子图元数据到JSON文件"""
        import json
        metadata = {
            "nodes": {
                n: {
                    "type": graph.nodes[n]["type"],
                    "size": graph.nodes[n]["size"]
                } for n in graph.nodes
            },
            "edges": [
                {
                    "from": u,
                    "to": v,
                    "from_type": graph.nodes[u]["type"],
                    "to_type": graph.nodes[v]["type"],
                    "data_flow": f"{graph.nodes[u]['size']}→{graph.nodes[v]['size']}"
                } for u, v in graph.edges
            ]
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def enumerate_fusion_options(self, subgraphs: List[nx.DiGraph]) -> Dict[str, List[nx.DiGraph]]:
        """枚举所有可能的融合方案"""
        options = {
            "Original": subgraphs,
            "Fully_Fused": [nx.compose_all(subgraphs)]
        }

        # 添加两两融合方案
        for i in range(len(subgraphs) - 1):
            fused = nx.compose(subgraphs[i], subgraphs[i + 1])
            option = [fused] + [g for j, g in enumerate(subgraphs) if j not in (i, i + 1)]
            options[f"Fused_{i + 1}"] = option

        return options

    def generate_report(self, subgraphs: List[nx.DiGraph], fusion_options: Dict[str, List[nx.DiGraph]]) -> str:
        """生成分析报告"""
        report = ["\n" + "=" * 50 + " DAG Analysis Report " + "=" * 50]

        report.append("\n[Subgraphs]")
        for i, subgraph in enumerate(subgraphs, 1):
            report.append(
                f"Subgraph {i}: {len(subgraph.nodes)} nodes, "
                f"{len(subgraph.edges)} edges | "
                f"Node types: {set(nx.get_node_attributes(subgraph, 'type').values())}"
            )

        report.append("\n[Fusion Options]")
        for name, graphs in fusion_options.items():
            report.append(
                f"{name}: {len(graphs)} graphs | "
                f"Total nodes: {sum(len(g.nodes) for g in graphs)}"
            )

        report.append("\n" + "=" * 120)
        return "\n".join(report)
