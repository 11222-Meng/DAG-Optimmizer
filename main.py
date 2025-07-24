#!/usr/bin/env python3
"""
DAG 切片点选择工具 - 完整边评分输出版
"""

import argparse
import os
import sys
from typing import List, Tuple

from models.bottleneck import BottleneckAnalyzer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import ConfigLoader
from analysis.slicer import Slicer
from models.node import Node


def print_all_edges(model: str, all_edges: List[Tuple[Node, Node, float]],
                    selected_edges: List[Tuple[Node, Node, float]]) -> None:
    """打印所有边的评分及选择结果"""
    print(f"\n[详细评分] {model} 所有边的得分分析:")
    print("=" * 80)
    print("{:<5} {:<30} {:<10} {:<10} {:<10} {:<10}".format(
        "Rank", "Edge", "Score", "Gain", "Cost", "Selected"))
    print("-" * 80)

    # 按得分排序所有边
    sorted_edges = sorted(all_edges, key=lambda x: x[2], reverse=True)

    for i, (from_node, to_node, score) in enumerate(sorted_edges, 1):
        is_selected = "★" if (from_node, to_node, score) in selected_edges else ""
        print("{:<5} {:<15} → {:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
            i,
            f"{from_node.id}",
            f"{to_node.id}",
            score,
            to_node.output_tensor_size * BottleneckAnalyzer.get_bottleneck_score(to_node.type),
            from_node.output_tensor_size,
            is_selected
        ))
    print("=" * 80)

    # 打印选中的边详情
    print("\n[最终选择] 推荐的切片边:")
    for i, (from_node, to_node, score) in enumerate(selected_edges, 1):
        print(f"{i}. {from_node.id} → {to_node.id}")
        print(
            f"   Score: {score:.2f} | Gain: {to_node.output_tensor_size * BottleneckAnalyzer.get_bottleneck_score(to_node.type):.2f}")
        print(f"   From: {from_node.type}(size={from_node.output_tensor_size})")
        print(f"   To: {to_node.type}(size={to_node.output_tensor_size})\n")


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", required=True, choices=ConfigLoader.get_available_models())
        parser.add_argument("--alpha", type=float, default=1.0)
        args = parser.parse_args()

        # 加载DAG
        dag = ConfigLoader.load_dag_from_config(f"configs/{args.model}.json")

        # 计算所有边得分
        slicer = Slicer(dag, alpha=args.alpha)
        all_edges = slicer.find_potential_slice_edges()  # 获取所有边
        selected_edges = slicer.select_top_k_slice_edges()  # 选择top-k

        # 打印结果
        print_all_edges(args.model, all_edges, selected_edges)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()