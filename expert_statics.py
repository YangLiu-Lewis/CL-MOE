import json
import sys
from pathlib import Path
from collections import Counter


def load_routing_from_file(path):
    """读取单个 routing JSON 文件，返回 routing dict"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["routing"]


def count_expert_usage_per_file(path):
    """统计单个文件中所有 expert 的全局使用次数"""
    routing = load_routing_from_file(path)
    counter = Counter()

    # routing: {module_name: {"topk_idx": [[...], [...], ...]}, ...}
    for info in routing.values():
        # for indices in info["topk_idx"]:
        counter.update(info["topk_idx"][0])

    return counter


def count_expert_usage_from_folder(folder):
    """统计文件夹内所有 JSON 文件的总 expert 使用次数"""
    folder = Path(folder)
    assert folder.is_dir(), f"{folder} 不是有效文件夹"

    global_counter = Counter()

    # 如果你只有 .json 日志，用这一行就够：
    for file in folder.glob("*.json"):
        c = count_expert_usage_per_file(file)
        global_counter.update(c)

    # 如果你后面既有 .json 又可能有 .txt，可改成：
    # for file in folder.glob("*"):
    #     if file.suffix.lower() in {".json", ".txt"}:
    #         c = count_expert_usage_per_file(file)
    #         global_counter.update(c)

    return global_counter


def print_counter(counter, title="结果"):
    print(f"\n=== {title} ===")
    total = sum(counter.values())
    total = total if total > 0 else 1
    for expert_id, cnt in counter.most_common():
        print(f"expert {expert_id:2d}: {cnt:8d} ({cnt/total:.2%})")


def main():
    if len(sys.argv) != 2:
        print("用法: python expert_usage_count.py <文件或文件夹路径>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_file():
        counter = count_expert_usage_per_file(target)
        print_counter(counter, title=f"单文件统计: {target.name}")
    elif target.is_dir():
        counter = count_expert_usage_from_folder(target)
        print_counter(counter, title=f"counting: {target}")
    else:
        print("路径无效")
        sys.exit(1)


if __name__ == "__main__":
    main()
