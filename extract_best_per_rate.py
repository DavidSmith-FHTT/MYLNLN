#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_best_per_rate.py

提取每个缺失率(missing rate)下的最优值详情，并生成汇总表格(best/avg)。
从已有的log文件中提取数据，无需重新运行评估。

主要功能：
1. 解析日志文件，提取不同缺失率下的模型性能指标
2. 计算每个缺失率下的最优值（多个seed中取最优）和平均值
3. 生成格式化的汇总表格，便于结果分析和论文写作

用法示例:
    python extract_best_per_rate.py mosi
    python extract_best_per_rate.py mosei --log_dir ./log/main_experiment/mosei/
    python extract_best_per_rate.py sims --experiment wo_reconstructor
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# 默认缺失率列表：从0%到90%，步长10%
DEFAULT_MISSING_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# 默认随机种子列表：用于多次实验取平均/最优
DEFAULT_SEEDS = [1111, 1112, 1113]


def _get_metrics_config(dataset: str) -> Dict[str, Dict]:
    """
    根据数据集返回指标配置:
    - pattern_seed: 正则表达式，用于从 log 中提取 (seed, value) 对
    - suffix: log 文件后缀名，用于定位具体的日志文件
    - lower_better: 布尔值，指示该指标是否越小越好（影响最优值选择）
    
    这个函数是核心配置函数，定义了如何从不同数据集的日志中提取各种性能指标
    """
    dataset = dataset.lower()
    if dataset == "sims":
        # SIMS数据集的指标配置：主要用于情感分析任务
        return {
            "Mult_acc_2": {
                "pattern_seed": r"Seed (\d+).*'Mult_acc_2': (?:np\.float\d+\()?([\d.]+)\)?",
                "suffix": "Mult_acc_2.log",      # 二分类准确率日志文件
                "lower_better": False,           # 准确率越高越好
            },
            "F1_score": {
                "pattern_seed": r"Seed (\d+).*'F1_score': ([\d.]+)",
                "suffix": "Mult_acc_2.log",      # F1分数与准确率在同一文件
                "lower_better": False,           # F1分数越高越好
            },
            "Mult_acc_3": {
                "pattern_seed": r"Seed (\d+).*'Mult_acc_3': (?:np\.float\d+\()?([\d.]+)\)?",
                "suffix": "Mult_acc_3.log",      # 三分类准确率日志文件
                "lower_better": False,
            },
            "Mult_acc_5": {
                "pattern_seed": r"Seed (\d+).*'Mult_acc_5': (?:np\.float\d+\()?([\d.]+)\)?",
                "suffix": "Mult_acc_5.log",      # 五分类准确率日志文件
                "lower_better": False,
            },
            "MAE": {
                "pattern_seed": r"Seed (\d+).*'MAE': (?:np\.float\d+\()?([\d.]+)\)?",
                "suffix": "MAE.log",             # 平均绝对误差日志文件
                "lower_better": True,            # MAE越小越好
            },
            "Corr": {
                "pattern_seed": r"Seed (\d+).*'Corr': (?:np\.float\d+\()?([-]?[\d.]+)\)?",
                "suffix": "MAE.log",             # 相关系数与MAE在同一文件
                "lower_better": False,           # 相关系数越高越好
            },
        }
    elif dataset in ("mosi", "mosei"):
        # MOSI/MOSEI数据集的指标配置：用于多模态情感分析任务
        return {
            "Has0_acc_2": {
                "pattern_seed": r"Seed (\d+).*'Has0_acc_2': ([\d.]+)",
                "suffix": "Has0_acc_2.log",      # 包含零值的二分类准确率
                "lower_better": False,
            },
            "Has0_F1_score": {
                "pattern_seed": r"Seed (\d+).*'Has0_F1_score': ([\d.]+)",
                "suffix": "Has0_acc_2.log",      # 与Has0_acc_2在同一文件
                "lower_better": False,
            },
            "Non0_acc_2": {
                "pattern_seed": r"Seed (\d+).*'Non0_acc_2': ([\d.]+)",
                "suffix": "Non0_acc_2.log",      # 排除零值的二分类准确率
                "lower_better": False,
            },
            "Non0_F1_score": {
                "pattern_seed": r"Seed (\d+).*'Non0_F1_score': ([\d.]+)",
                "suffix": "Non0_acc_2.log",      # 与Non0_acc_2在同一文件
                "lower_better": False,
            },
            "Mult_acc_5": {
                "pattern_seed": r"Seed (\d+).*'Mult_acc_5': (?:np\.float\d+\()?([\d.]+)\)?",
                "suffix": "Mult_acc_5.log",      # 五分类准确率
                "lower_better": False,
            },
            "Mult_acc_7": {
                "pattern_seed": r"Seed (\d+).*'Mult_acc_7': (?:np\.float\d+\()?([\d.]+)\)?",
                "suffix": "Mult_acc_7.log",      # 七分类准确率
                "lower_better": False,
            },
            "MAE": {
                "pattern_seed": r"Seed (\d+).*'MAE': (?:np\.float\d+\()?([\d.]+)\)?",
                "suffix": "MAE.log",             # 平均绝对误差
                "lower_better": True,            # MAE越小越好
            },
            "Corr": {
                "pattern_seed": r"Seed (\d+).*'Corr': (?:np\.float\d+\()?([-]?[\d.]+)\)?",
                "suffix": "MAE.log",             # 相关系数
                "lower_better": False,
            },
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Expected one of: mosi/mosei/sims")


def _resolve_log_file(log_dir: str, file_prefix: str, suffix: str) -> Optional[Path]:
    """
    解析日志文件路径，支持两种命名模式：
    1. 优先尝试: {file_prefix}_{suffix} (如: robust_eval_Mult_acc_2.log)
    2. 备选方案: robust_eval_{suffix} (如: robust_eval_Mult_acc_2.log)
    
    这种设计允许灵活的日志文件命名，适应不同的实验设置
    """
    log_dir = Path(log_dir)
    p1 = log_dir / f"{file_prefix}_{suffix}"
    if p1.exists():
        return p1
    p2 = log_dir / f"robust_eval_{suffix}"
    if p2.exists():
        return p2
    return None


def _parse_seed_values(content: str, pattern_seed: str, seeds: List[int]) -> Dict[int, List[float]]:
    """
    从日志内容中按 seed 提取序列值.
    
    Args:
        content: 日志文件的完整文本内容
        pattern_seed: 正则表达式，用于匹配 "Seed XXXX 'Metric': value" 格式
        seeds: 需要提取的种子列表
    
    Returns:
        字典格式: {seed: [v0, v1, v2, ...]} 
        其中每个seed对应一个列表，包含该seed在不同缺失率下的指标值
    
    这个函数是数据提取的核心，通过正则表达式匹配将日志文本结构化
    """
    matches = re.findall(pattern_seed, content)
    seed_values = {s: [] for s in seeds}  # 初始化每个seed的空列表
    for seed_str, value_str in matches:
        seed = int(seed_str)
        if seed in seed_values:
            seed_values[seed].append(float(value_str))
    return seed_values


def _is_percent_metric(metric_name: str) -> bool:
    """
    判断指标是否为百分比指标（准确率、F1分数等）
    用于决定输出格式：百分比指标显示为xx.xx%，其他指标显示为小数
    """
    n = metric_name.lower()
    return ("acc" in n) or ("f1" in n)


def print_table(table_data: Dict[str, List[float]], dataset: str, missing_rates: List[float], title: str) -> None:
    """
    打印格式化的汇总表格，支持最优值表格和平均值表格
    
    Args:
        table_data: 字典，键为指标名，值为该指标在各缺失率下的数值列表
        dataset: 数据集名称，决定表格格式和指标顺序
        missing_rates: 缺失率列表，决定表格行数
        title: 表格标题，区分是最优值还是平均值表格
    
    这个函数负责生成论文中常见的实验结果表格格式，包括：
    - 表头和分隔线
    - 按缺失率逐行显示数据
    - 百分比指标自动转换为百分制显示
    - 最后一行显示所有缺失率的平均值
    """
    dataset = dataset.lower()
    print()
    print("=" * 80)
    print(f"【{dataset.upper()}】{title}")
    print("=" * 80)
    print()

    if not table_data:
        print("警告: 没有足够的数据生成表格")
        return

    # 根据数据集类型设置不同的表头和指标顺序
    if dataset == "sims":
        header = "Missing Rate | Acc-2  | F1     | Acc-3  | Acc-5  | MAE    | Corr   |"
        sep =    "-------------|--------|--------|--------|--------|--------|--------|"
        metric_order = ["Mult_acc_2", "F1_score", "Mult_acc_3", "Mult_acc_5", "MAE", "Corr"]
    else:
        header = "Missing Rate | Acc-2(Has0) | Acc-2(Non0) | F1(Has0) | F1(Non0) | Acc-5  | Acc-7  | MAE    | Corr   |"
        sep =    "-------------|-------------|-------------|----------|----------|--------|--------|--------|--------|"
        metric_order = ["Has0_acc_2", "Non0_acc_2", "Has0_F1_score", "Non0_F1_score", "Mult_acc_5", "Mult_acc_7", "MAE", "Corr"]

    print(header)
    print(sep)

    # 逐行生成每个缺失率的数据
    for i, rate in enumerate(missing_rates):
        row = f"    {rate:.1f}     |"
        for m in metric_order:
            if m in table_data and i < len(table_data[m]):
                v = table_data[m][i]
                # 百分比指标转换为百分制显示
                if _is_percent_metric(m):
                    row += f" {v*100:7.2f} |" if dataset != "sims" else f" {v*100:6.2f} |"
                else:
                    row += f" {v:6.4f} |"
            else:
                row += "   -   |"
        print(row)

    print(sep)

    # 生成平均值行：计算每个指标在所有缺失率下的平均性能
    avg_row = "    Avg.     |"
    for m in metric_order:
        if m in table_data and table_data[m]:
            av = sum(table_data[m]) / len(table_data[m])
            if _is_percent_metric(m):
                avg_row += f" {av*100:7.2f} |" if dataset != "sims" else f" {av*100:6.2f} |"
            else:
                avg_row += f" {av:6.4f} |"
        else:
            avg_row += "   -   |"
    print(avg_row)


def generate_summary_tables(
    metrics_config: Dict[str, Dict],
    log_dir: str,
    dataset: str,
    missing_rates: List[float],
    file_prefix: str,
    seeds: List[int],
) -> None:
    """
    生成并打印最优值表格和平均值表格
    
    Args:
        metrics_config: 指标配置字典，包含每个指标的正则表达式和文件信息
        log_dir: 日志文件目录路径
        dataset: 数据集名称
        missing_rates: 缺失率列表
        file_prefix: 日志文件前缀
        seeds: 种子列表
    
    这个函数是整个脚本的核心，负责：
    1. 遍历所有指标，从对应的日志文件中提取数据
    2. 对每个缺失率，计算多个seed的最优值和平均值
    3. 调用print_table生成格式化的结果表格
    """
    table_best: Dict[str, List[float]] = {}  # 存储最优值表格数据
    table_avg: Dict[str, List[float]] = {}   # 存储平均值表格数据

    # 遍历每个指标，处理对应的日志文件
    for metric_name, cfg in metrics_config.items():
        log_file = _resolve_log_file(log_dir, file_prefix, cfg["suffix"])
        if log_file is None:
            continue  # 跳过找不到文件的指标

        # 读取日志文件内容并解析seed-value数据
        content = log_file.read_text(encoding="utf-8", errors="ignore")
        seed_values = _parse_seed_values(content, cfg["pattern_seed"], seeds)

        # 完整性检查：确保每个seed都有对应所有缺失率的数据
        if not all(len(seed_values[s]) == len(missing_rates) for s in seeds):
            continue

        # 对每个缺失率，计算最优值和平均值
        best_vals, avg_vals = [], []
        for i in range(len(missing_rates)):
            vals = [seed_values[s][i] for s in seeds]  # 获取该缺失率下所有seed的值
            avg_vals.append(sum(vals) / len(vals))     # 计算平均值
            
            # 根据指标类型选择最优值（最大值或最小值）
            if cfg["lower_better"]:
                best_vals.append(min(vals))  # 越小越好的指标取最小值
            else:
                best_vals.append(max(vals))  # 越大越好的指标取最大值

        # 将计算结果存入表格数据字典
        table_best[metric_name] = best_vals
        table_avg[metric_name] = avg_vals

    # 生成并打印两个汇总表格
    print_table(table_best, dataset, missing_rates, "不同缺失率下的性能汇总表格 (最优值)")
    print_table(table_avg, dataset, missing_rates, "不同缺失率下的性能汇总表格 (平均值)")

    # 打印详细的指标说明
    print()
    print("=" * 80)
    print("说明:")
    if dataset.lower() == "sims":
        print("  - Acc-2, Acc-3, Acc-5: 分类准确率（百分制）")
        print("  - F1: F1分数（百分制）")
    else:
        print("  - Acc-2(Has0): 二分类准确率-包含零值（负/非负，百分制）")
        print("  - F1(Has0): F1分数-包含零值（百分制）")
        print("  - Acc-2(Non0): 二分类准确率-排除零值（负/正，百分制）")
        print("  - F1(Non0): F1分数-排除零值（百分制）")
        print("  - Acc-5, Acc-7: 五分类/七分类准确率（百分制）")
    print("  - MAE: 平均绝对误差（越小越好）")
    print("  - Corr: 相关系数")
    print("  - 最优值表格: 每个缺失率下的值为 3 个 seed 中的最优值")
    print("  - 平均值表格: 每个缺失率下的值为 3 个 seed 的平均值")
    print("=" * 80)


def extract_best_per_rate(log_dir: str, dataset: str, experiment_name: str = "main_experiment") -> None:
    """
    主函数：从log文件中提取每个缺失率的最优值详情，并打印 + 生成汇总表格
    
    Args:
        log_dir: log目录路径，包含各种性能指标的日志文件
        dataset: 数据集名称 (mosi/mosei/sims)，决定指标配置和表格格式
        experiment_name: 实验名称，用于确定日志文件名前缀
    
    这个函数是整个脚本的入口点，执行完整的分析流程：
    1. 获取数据集对应的指标配置
    2. 确定日志文件前缀和默认参数
    3. 逐个指标提取详细的最优值信息
    4. 生成汇总表格
    """
    dataset = dataset.lower()
    metrics_config = _get_metrics_config(dataset)

    # 根据实验名称确定日志文件前缀
    file_prefix = "robust_eval" if experiment_name == "main_experiment" else experiment_name
    missing_rates = DEFAULT_MISSING_RATES  # 使用默认缺失率列表
    seeds = DEFAULT_SEEDS                  # 使用默认种子列表

    # 打印详细的最优值信息标题
    print("=" * 80)
    print(f"【{dataset.upper()}】每个缺失率的最优值详情")
    print("=" * 80)
    print()

    # 逐个指标处理，显示每个缺失率下的详细最优值信息
    for metric_name, cfg in metrics_config.items():
        log_file = _resolve_log_file(log_dir, file_prefix, cfg["suffix"])
        if log_file is None:
            print(f"警告: 找不到 {file_prefix}_{cfg['suffix']} 或 robust_eval_{cfg['suffix']}，跳过 {metric_name}")
            continue

        # 读取并解析日志文件
        content = log_file.read_text(encoding="utf-8", errors="ignore")
        seed_values = _parse_seed_values(content, cfg["pattern_seed"], seeds)

        # 数据完整性检查
        if not any(seed_values[s] for s in seeds):
            print(f"警告: 在 {log_file} 中未找到 {metric_name} 的数据")
            continue

        if not all(len(seed_values[s]) == len(missing_rates) for s in seeds):
            print(f"警告: {metric_name} 的数据不完整（期望 {len(missing_rates)} 个缺失率点）")
            continue

        # 确定最优值选择策略（最大值或最小值）
        is_lower_better = cfg["lower_better"]
        opt_label = "最小值" if is_lower_better else "最大值"

        print(f"【{metric_name}】 (每个缺失率取{opt_label})")
        print("-" * 80)

        # 对每个缺失率，显示所有seed的值和最优值
        for i, rate in enumerate(missing_rates):
            values_at_rate = {s: seed_values[s][i] for s in seeds}
            avg_value = sum(values_at_rate.values()) / len(values_at_rate)

            # 根据指标类型选择最优seed
            if is_lower_better:
                best_seed = min(values_at_rate, key=values_at_rate.get)
            else:
                best_seed = max(values_at_rate, key=values_at_rate.get)
            best_value = values_at_rate[best_seed]

            # 格式化输出：百分比指标显示为百分制
            if _is_percent_metric(metric_name):
                values_str = ", ".join([f"seed{s}={v*100:.2f}%" for s, v in sorted(values_at_rate.items())])
                print(f"  Missing Rate {rate:.1f}: {opt_label}={best_value*100:.2f}% (来自seed{best_seed}), Avg={avg_value*100:.2f}%  [{values_str}]")
            else:
                values_str = ", ".join([f"seed{s}={v:.4f}" for s, v in sorted(values_at_rate.items())])
                print(f"  Missing Rate {rate:.1f}: {opt_label}={best_value:.4f} (来自seed{best_seed}), Avg={avg_value:.4f}  [{values_str}]")

        print()  # 空行分隔不同指标

    print("=" * 80)
    print()

    # 生成汇总表格
    generate_summary_tables(metrics_config, log_dir, dataset, missing_rates, file_prefix, seeds)


def main():
    """
    命令行入口函数，处理用户输入的参数并调用主分析函数
    
    支持的参数：
    - dataset: 必需参数，数据集名称（mosi/mosei/sims）
    --log_dir: 可选参数，自定义日志目录路径
    --experiment: 可选参数，实验名称，用于确定日志文件前缀
    
    使用示例：
    python extract_best_per_rate.py mosi
    python extract_best_per_rate.py mosei --log_dir ./custom_log_dir/
    python extract_best_per_rate.py sims --experiment wo_reconstructor
    """
    parser = argparse.ArgumentParser(description="提取每个缺失率下的最优值详情（从log解析，无需重新评估）")
    parser.add_argument("dataset", type=str, choices=["mosi", "mosei", "sims"], help="数据集名称")
    parser.add_argument("--log_dir", type=str, default=None, help="log目录路径 (默认: ./log/main_experiment/{dataset}/)")
    parser.add_argument("--experiment", type=str, default="main_experiment", help="实验名称 (默认: main_experiment)")
    args = parser.parse_args()

    # 如果用户没有指定log目录，使用默认路径
    if args.log_dir is None:
        if args.experiment == "main_experiment":
            args.log_dir = f"./log/main_experiment/{args.dataset}/"
        else:
            args.log_dir = f"./log/{args.experiment}/{args.dataset}/"

    # 调用主分析函数
    extract_best_per_rate(args.log_dir, args.dataset, args.experiment)


if __name__ == "__main__":
    """
    脚本执行入口点
    
    当直接运行此脚本时，会调用main()函数
    这允许脚本既可以作为独立程序运行，也可以被其他模块导入使用
    """
    main()
