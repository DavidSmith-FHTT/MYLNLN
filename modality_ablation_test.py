#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modality_ablation_test.py

对已训练好的 best_{key_eval}_{seed}.pth 模型做“模态消融”评估：
- 仅文本 / 仅音频 / 仅视频 / 两模态 / 三模态
- 不使用随机 missing rate（固定 missing_rate_eval_test=0.0）
- 从 ckpt 读模型进行评估，可选评估后删除 ckpt 节省磁盘

用法示例:
    python modality_ablation_test.py --config_file config/mosi.yaml --key_eval Has0_acc_2
    python modality_ablation_test.py --config_file config/sims.yaml --key_eval MAE --delete_after_eval
"""

import os
import copy
import yaml
import argparse
import torch

from core.dataset import MMDataEvaluationLoader
from core.metric import MetricsTop
# from core.utils import dict_to_namespace


DEFAULT_SEEDS = [1111, 1112, 1113]


def apply_modality_mask(data, text_available: bool, audio_available: bool, vision_available: bool):
    """
    对数据应用模态掩码，将不可用的模态完全置零

    注意：
    - 文本: 你这里使用 data['text_m'] 作为 BERT 的 input_ids（或类似结构）
      你原始写法是 data['text_m'][:, 0, :] = 103.0，这假设 text_m 的形状是 [B, 1, L] 或 [B, 1, ?]
      为了兼容更常见的 [B, L] / [B, 1, L] 两种，这里做了更稳健的判断。
    """
    if not text_available:
        # 尽量把 input_ids 全部替换为 [MASK] = 103
        # 兼容 [B, L] 或 [B, 1, L]
        if data["text_m"].dim() == 2:
            data["text_m"][:] = 103
        elif data["text_m"].dim() == 3:
            data["text_m"][:, 0, :] = 103
        else:
            # 兜底：全部置 103
            data["text_m"] = torch.full_like(data["text_m"], 103)

    if not audio_available:
        data["audio_m"] = torch.zeros_like(data["audio_m"])

    if not vision_available:
        data["vision_m"] = torch.zeros_like(data["vision_m"])

    return data


@torch.no_grad()
def evaluate(model, eval_loader, metrics, device, text_available: bool, audio_available: bool, vision_available: bool):
    """
    评估函数，支持指定模态缺失（硬置零）
    """
    y_pred, y_true = [], []
    model.eval()

    for _, data in enumerate(eval_loader):
        data = apply_modality_mask(data, text_available, audio_available, vision_available)

        img = data["vision_m"].to(device)
        audio = data["audio_m"].to(device)
        text = data["text_m"].to(device)
        sentiment_labels = data["labels"]["M"].to(device)

        out = model(img, audio, text)
        y_pred.append(out.detach().cpu())
        y_true.append(sentiment_labels.detach().cpu())

    pred = torch.cat(y_pred, dim=0)
    true = torch.cat(y_true, dim=0)
    results = metrics(pred, true)
    return results


def _collect_avg_best(dataset_name: str, key_eval: str, results_list: list):
    """
    根据数据集/指标计算 avg 和 best（遵循你原先逻辑：acc/f1取max, MAE取min, Corr取max）
    返回一个 dict，便于打印。
    """
    if len(results_list) == 0:
        return {}

    out = {}

    if dataset_name == "sims":
        if key_eval == "Mult_acc_2":
            out["avg"] = {
                "Mult_acc_2": sum(r["Mult_acc_2"] for r in results_list) / len(results_list),
                "F1_score": sum(r["F1_score"] for r in results_list) / len(results_list),
            }
            out["best"] = {
                "Mult_acc_2": max(r["Mult_acc_2"] for r in results_list),
                "F1_score": max(r["F1_score"] for r in results_list),
            }
        elif key_eval == "Mult_acc_3":
            out["avg"] = {"Mult_acc_3": sum(r["Mult_acc_3"] for r in results_list) / len(results_list)}
            out["best"] = {"Mult_acc_3": max(r["Mult_acc_3"] for r in results_list)}
        elif key_eval == "Mult_acc_5":
            out["avg"] = {"Mult_acc_5": sum(r["Mult_acc_5"] for r in results_list) / len(results_list)}
            out["best"] = {"Mult_acc_5": max(r["Mult_acc_5"] for r in results_list)}
        elif key_eval == "MAE":
            out["avg"] = {
                "MAE": sum(r["MAE"] for r in results_list) / len(results_list),
                "Corr": sum(r["Corr"] for r in results_list) / len(results_list),
            }
            out["best"] = {
                "MAE": min(r["MAE"] for r in results_list),
                "Corr": max(r["Corr"] for r in results_list),
            }
        else:
            # 兜底：只对 key_eval 做 avg/max
            out["avg"] = {key_eval: sum(r[key_eval] for r in results_list) / len(results_list)}
            out["best"] = {key_eval: max(r[key_eval] for r in results_list)}
    else:
        # mosi / mosei
        if key_eval == "Has0_acc_2":
            out["avg"] = {
                "Has0_acc_2": sum(r["Has0_acc_2"] for r in results_list) / len(results_list),
                "Has0_F1_score": sum(r["Has0_F1_score"] for r in results_list) / len(results_list),
            }
            out["best"] = {
                "Has0_acc_2": max(r["Has0_acc_2"] for r in results_list),
                "Has0_F1_score": max(r["Has0_F1_score"] for r in results_list),
            }
        elif key_eval == "Non0_acc_2":
            out["avg"] = {
                "Non0_acc_2": sum(r["Non0_acc_2"] for r in results_list) / len(results_list),
                "Non0_F1_score": sum(r["Non0_F1_score"] for r in results_list) / len(results_list),
            }
            out["best"] = {
                "Non0_acc_2": max(r["Non0_acc_2"] for r in results_list),
                "Non0_F1_score": max(r["Non0_F1_score"] for r in results_list),
            }
        elif key_eval == "Mult_acc_5":
            out["avg"] = {"Mult_acc_5": sum(r["Mult_acc_5"] for r in results_list) / len(results_list)}
            out["best"] = {"Mult_acc_5": max(r["Mult_acc_5"] for r in results_list)}
        elif key_eval == "Mult_acc_7":
            out["avg"] = {"Mult_acc_7": sum(r["Mult_acc_7"] for r in results_list) / len(results_list)}
            out["best"] = {"Mult_acc_7": max(r["Mult_acc_7"] for r in results_list)}
        elif key_eval == "MAE":
            out["avg"] = {
                "MAE": sum(r["MAE"] for r in results_list) / len(results_list),
                "Corr": sum(r["Corr"] for r in results_list) / len(results_list),
            }
            out["best"] = {
                "MAE": min(r["MAE"] for r in results_list),
                "Corr": max(r["Corr"] for r in results_list),
            }
        else:
            out["avg"] = {key_eval: sum(r[key_eval] for r in results_list) / len(results_list)}
            out["best"] = {key_eval: max(r[key_eval] for r in results_list)}

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="配置文件路径")
    parser.add_argument("--key_eval", type=str, required=True, help="评估指标（用于选择 best_{key_eval}_{seed}.pth）")
    parser.add_argument("--delete_after_eval", action="store_true", help="评估完成后立即删除模型文件以节省磁盘空间")
    opt = parser.parse_args()
    print(opt)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)

    # load config
    with open(opt.config_file, "r", encoding="utf-8") as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)

    # 注意：你原工程 eval loader 用 dict，因此我们保留 args_dict 供 loader 使用；
    # model 仍使用 namespace args。
    args = dict_to_namespace(copy.deepcopy(args_dict))
    dataset_name = args.dataset.datasetName
    key_eval = opt.key_eval

    print("Using ALMT model")
    model = build_model(args).to(device)
    metrics = MetricsTop().getMetics(dataset_name)

    # 模态组合
    modality_combinations = [
        {"name": "{l}", "text": True, "audio": False, "vision": False, "desc": "仅文本 (缺失: 音频, 视频)"},
        {"name": "{a}", "text": False, "audio": True, "vision": False, "desc": "仅音频 (缺失: 文本, 视频)"},
        {"name": "{v}", "text": False, "audio": False, "vision": True, "desc": "仅视频 (缺失: 文本, 音频)"},
        {"name": "{l,a}", "text": True, "audio": True, "vision": False, "desc": "文本+音频 (缺失: 视频)"},
        {"name": "{l,v}", "text": True, "audio": False, "vision": True, "desc": "文本+视频 (缺失: 音频)"},
        {"name": "{a,v}", "text": False, "audio": True, "vision": True, "desc": "音频+视频 (缺失: 文本)"},
        {"name": "{l,a,v}", "text": True, "audio": True, "vision": True, "desc": "完整三模态 (无缺失)"},
    ]

    # 固定评估时不使用随机缺失
    args_dict.setdefault("base", {})
    args_dict["base"]["missing_rate_eval_test"] = 0.0

    for combo in modality_combinations:
        print("\n" + "=" * 80)
        print(f"测试模态组合: {combo['name']} - {combo['desc']}")
        print("=" * 80)

        test_results_list = []

        for cur_seed in DEFAULT_SEEDS:
            best_ckpt = os.path.join(f"ckpt/{dataset_name}/best_{key_eval}_{cur_seed}.pth")
            if not os.path.exists(best_ckpt):
                print(f"警告: 模型文件不存在 {best_ckpt}")
                continue

            ckpt = torch.load(best_ckpt, map_location="cpu")
            # 兼容你原先格式：ckpt['state_dict']
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=True)

            # 每次都重新创建 loader（确保 missing_rate_eval_test=0.0）
            args_dict["base"]["missing_rate_eval_test"] = 0.0
            eval_loader = MMDataEvaluationLoader(args_dict)

            results = evaluate(
                model=model,
                eval_loader=eval_loader,
                metrics=metrics,
                device=device,
                text_available=combo["text"],
                audio_available=combo["audio"],
                vision_available=combo["vision"],
            )

            print(f"Seed {cur_seed} - 模态组合: {combo['name']} ({combo['desc']})")
            print(f"结果: {results}")

            test_results_list.append(results)

            if opt.delete_after_eval:
                try:
                    os.remove(best_ckpt)
                    print(f"✓ 已删除模型文件: {best_ckpt}")
                except Exception as e:
                    print(f"✗ 删除模型文件失败: {best_ckpt}, 错误: {e}")

        # 统计
        if len(test_results_list) == 3:
            stats = _collect_avg_best(dataset_name, key_eval, test_results_list)
            if stats:
                print("\n" + "-" * 80)
                if "avg" in stats:
                    avg_str = ", ".join([f"{k}={v:.4f}" for k, v in stats["avg"].items()])
                    print(f"【平均值】模态组合: {combo['name']}, {avg_str}")
                if "best" in stats:
                    best_str = ", ".join([f"{k}={v:.4f}" for k, v in stats["best"].items()])
                    print(f"【最优值】模态组合: {combo['name']}, {best_str}")
                print("-" * 80)


if __name__ == "__main__":
    main()
