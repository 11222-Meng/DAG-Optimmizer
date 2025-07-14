import matplotlib.pyplot as plt
import networkx as nx
from typing import List
import os


def visualize_dag_with_cut_points(graph, cut_points: List[str], filepath: str):
    """生成带切点标注的DAG可视化并保存到指定路径"""
    plt.figure(figsize=(10, 4))

    # 水平布局
    pos = {node: (i, 0) for i, node in enumerate(sorted(graph.nodes))}

    # 绘制节点
    node_colors = ['red' if node in cut_points else 'lightblue' for node in graph.nodes]
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=2000,
        edgecolors='black',
        linewidths=1.5
    )

    # 绘制边
    nx.draw_networkx_edges(
        graph, pos,
        edge_color='grey',
        width=2,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=20
    )

    # 突出显示切点后的边
    cut_edges = [(u, v) for u, v in graph.edges() if u in cut_points]
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=cut_edges,
        edge_color='red',
        style='dashed',
        width=2.5,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=25
    )

    # 节点标签
    labels = {
        n: f"{graph.nodes[n]['operator'].op_type}\n{graph.nodes[n]['operator'].name or ''}"
        for n in graph.nodes
    }
    nx.draw_networkx_labels(
        graph, pos, labels,
        font_size=10,
        font_family='sans-serif',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    plt.title(f"DAG Structure with Cut Points (k={len(cut_points)})", pad=20)
    plt.xlim(-0.5, len(graph.nodes) - 0.5)
    plt.axis('off')

    # 确保目录存在并保存
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, bbox_inches='tight', dpi=120)
    plt.close()