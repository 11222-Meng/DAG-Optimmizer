from typing import List, Tuple, Dict
import argparse
import os
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
from dag_optimizer.core.graph import DAG
from dag_optimizer.core.optimizer import DAGOptimizer
from dag_optimizer.models.operator import Operator, OptimizationMetric
from dag_optimizer.utils.visualization import visualize_dag_with_cut_points

# 常量定义
OUTPUT_DIR = Path("dag_visualizations")


def ensure_output_dir():
    """确保可视化输出目录存在"""
    OUTPUT_DIR.mkdir(exist_ok=True)


def create_dag_from_ops(operators: List[Tuple[str, str]], edges: List[Tuple[int, int]]) -> DAG:
    """通用DAG构建函数"""
    dag = DAG()
    for idx, (op_type, op_name) in enumerate(operators):
        dag.add_operator(Operator(op_type, op_name), f"op{idx + 1}")
    for from_idx, to_idx in edges:
        dag.add_dependency(f"op{from_idx + 1}", f"op{to_idx + 1}")
    return dag


def create_dag1() -> DAG:
    """4节点CNN结构"""
    operators = [
        ("Conv2d", "conv1"),
        ("BatchNorm2d", "bn1"),
        ("ReLU", "relu"),
        ("Linear", "fc1")
    ]
    edges = [(0, 1), (1, 2), (2, 3)]
    return create_dag_from_ops(operators, edges)


def create_dag2() -> DAG:
    """4节点混合结构"""
    operators = [
        ("MaxPool2d", "pool1"),
        ("FireModule", "fire1"),
        ("AvgPool2d", "pool2"),
        ("Linear", "fc1")
    ]
    edges = [(0, 1), (1, 2), (2, 3)]
    return create_dag_from_ops(operators, edges)


def create_dag3() -> DAG:
    """4节点残差结构"""
    operators = [
        ("InvertedResidual", "inv1"),
        ("AdaptiveAvgPool2d", "pool1"),
        ("Conv2d", "conv1"),
        ("BatchNorm2d", "bn1")
    ]
    edges = [(0, 1), (1, 2), (2, 3)]
    return create_dag_from_ops(operators, edges)


def analyze_dag(dag_creator, dag_num: int, metric: OptimizationMetric) -> Dict:
    """执行单个DAG分析并生成可视化"""
    ensure_output_dir()
    dag = dag_creator()
    optimizer = DAGOptimizer(dag.graph, metric)
    result = optimizer.optimize()

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"dag{dag_num}_{metric.value}_{timestamp}.png"

    # 生成可视化
    cut_points = [cp[0] for cp in result['optimal_cut_points']]
    visualize_dag_with_cut_points(
        dag.graph,
        cut_points,
        str(filename)
    )

    return {
        **result,
        'dag_name': f"DAG{dag_num}",
        'metric': metric.value,
        'dag': dag,
        'visualization': filename.name
    }


def print_single_result(result: Dict):
    """打印单个DAG的详细结果"""
    print(f"\n{'=' * 50}")
    print(f"=== {result['dag_name']} - {result['target_metric'].upper()} OPTIMIZATION ===")
    print(f"Operators: {result['total_operators']} | Calculated k: {result['k']}")
    print(f"Visualization: {OUTPUT_DIR}/{result['visualization']}")

    # 关键路径信息
    cp_ops = " → ".join(
        f"{result['dag'].get_operator(n).op_type}({result['dag'].get_operator(n).name or ''})"
        for n in result['original_critical_path']
    )
    print(f"\nCritical Path ({result['original_metric']:.1f}): {cp_ops}")

    # 切片点分析
    if result['optimal_cut_points']:
        print("\nSELECTED CUT POINTS:")
        headers = ["Rank", "Operator", "Name", "Benefit", "Cost", "Ratio"]
        table = []
        for i, (op_id, benefit, cost) in enumerate(result['optimal_cut_points'], 1):
            op = result['dag'].get_operator(op_id)
            table.append([
                i, op.op_type, op.name or "unnamed",
                f"{benefit:.1f}", f"{cost:.1f}", f"{benefit / cost:.1f}"
            ])
        print(tabulate(table, headers=headers, tablefmt="grid"))
    else:
        print("\nNo valid cut points found")


def print_batch_summary(results: List[Dict]):
    """打印批量分析汇总结果"""
    print("\n\n" + "=" * 50)
    print("=== BATCH ANALYSIS SUMMARY ===")
    print(f"All visualizations saved to: {OUTPUT_DIR.resolve()}")

    # 汇总表格
    summary_headers = ["DAG", "Metric", "Best Cut", "Benefit", "Cost", "Ratio", "Visualization"]
    summary_table = []

    for res in results:
        if res['optimal_cut_points']:
            best = res['optimal_cut_points'][0]
            op = res['dag'].get_operator(best[0])
            summary_table.append([
                res['dag_name'],
                res['target_metric'],
                f"{op.op_type}({op.name or ''})",
                f"{best[1]:.1f}",
                f"{best[2]:.1f}",
                f"{best[1] / best[2]:.1f}",
                res['visualization']
            ])

    print(tabulate(summary_table, headers=summary_headers, tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser(
        description="DAG Optimizer - Analyze and optimize computational graphs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode",
                        choices=["single", "batch"],
                        default="batch",
                        help="Analysis mode")
    parser.add_argument("--dag",
                        type=int,
                        choices=[1, 2, 3],
                        default=1,
                        help="DAG selection (1-3) for single mode")
    parser.add_argument("--metric",
                        choices=["memory", "computation", "latency"],
                        default="memory",
                        help="Optimization target metric")
    args = parser.parse_args()

    dag_creators = {
        1: create_dag1,
        2: create_dag2,
        3: create_dag3
    }

    if args.mode == "single":
        # 单DAG分析模式
        print(f"\nAnalyzing DAG{args.dag} with {args.metric} optimization...")
        result = analyze_dag(
            dag_creators[args.dag],
            args.dag,
            OptimizationMetric(args.metric)
        )
        print_single_result(result)
    else:
        # 批量分析模式
        print("=== BATCH ANALYSIS STARTED ===")
        all_results = []
        for dag_num, creator in dag_creators.items():
            for metric in OptimizationMetric:
                print(f"  Processing DAG{dag_num} - {metric.value}...")
                all_results.append(analyze_dag(creator, dag_num, metric))

        print_batch_summary(all_results)


if __name__ == "__main__":
    main()