import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from models.dag import DAG
from models.node import Node
from analysis.slicer import Slicer
from utils.config_loader import ConfigLoader


def main():
    try:
        # 参数解析
        parser = argparse.ArgumentParser(
            description='DAG Optimizer - Automatic Graph Partitioning',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "--model",
            required=True,
            choices=ConfigLoader.get_available_models(),
            help="Model to analyze"
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=1.0,
            help="Cost coefficient (score = gain - α * cost)"
        )
        args = parser.parse_args()

        # 加载配置
        config_path = f"configs/{args.model}.json"
        dag, bottleneck_type = ConfigLoader.load_dag_and_bottleneck(config_path)

        # 选择切片边
        slicer = Slicer(dag, alpha=args.alpha)
        selected_edges = slicer.select_top_k_slice_edges()

        # 打印选择的边
        print_slice_edges(selected_edges)

        # 划分子图
        subgraphs = slicer.split_into_subgraphs(selected_edges)

        # 可视化保存
        saved_files = slicer.visualize_subgraphs(
            subgraphs=subgraphs,
            model_name=args.model,
            bottleneck_type=bottleneck_type,
            save_dir="dag_results"
        )

        # 打印结果
        print_saved_results(saved_files)

    except Exception as e:
        print(f"\n[Error] {str(e)}", file=sys.stderr)
        sys.exit(1)


def print_slice_edges(edges: List[Tuple[Node, Node, float]]):
    """打印切片边表格"""
    print("\n[Selected Slice Edges]")
    print("-" * 80)
    print("{:<5} {:<25} {:<10} {:<15} {:<15}".format(
        "Rank", "Edge", "Score", "From Type", "To Type"))
    print("-" * 80)
    for i, (from_node, to_node, score) in enumerate(edges, 1):
        print("{:<5} {:<15} → {:<10} {:<10.2f} {:<15} {:<15}".format(
            i, from_node.id, to_node.id, score,
            from_node.type, to_node.type))


def print_saved_results(saved_files: dict):
    """打印保存结果"""
    if saved_files and saved_files["images"]:
        print("\nSaved visualization results:")
        for img, meta in zip(saved_files["images"], saved_files["metadata"]):
            print(f"  - {img}")
            print(f"  - {meta}")
    else:
        print("\nNo subgraphs were saved")


if __name__ == "__main__":
    main()