import json
from os import path
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

def score(x, y):
    if y > x:
        return 100 * max(0, 1. - (y / x - 1) / 3)
    else:
        return 100 * max(0, 1. - (x / y - 1) / 2)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["phi4-unsloth", "phi4", "mistral","qwen2.5","llama3.1-8b-instruct","llama2-7b-chat-4k", "llama-2-7B-32k-instruct", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument("--pred_path", type=str, default="pred")
    return parser.parse_args(args)

args = parse_args()
model_name = args.model
res_path = f"{args.pred_path}/{model_name}/"
result = {}
for file in os.listdir(res_path):
    if "json" not in file or "result" in file or "judge" in file or "profile" in file or "log" in file:
        continue
    print(file)
    try:
        prediction = [json.loads(line) for line in open(f"{res_path}{file}", encoding='utf-8')]
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file}: {e}")
        continue
    x, y, scores = [], [], []
    for pred in prediction:
        x.append(pred["length"])
        y.append(pred["response_length"])
        try:
            scores.append(score(pred["length"], pred["response_length"]))
        except ZeroDivisionError:
            scores.append(0)
    result[".".join(file.split(".")[:-1])] = np.mean(scores)
# print(np.mean(scores))

    # set plt size 6x6
    plt.figure(figsize=(6, 6))
    lmt = 25000
    # plot x, y
    plt.scatter(x, y, s=100, c='r', alpha=0.3)
    # plot x=y
    plt.plot([0, lmt], [0, lmt], 'k--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(50, lmt)
    plt.ylim(50, lmt)
    plt.xlabel('Required Length', fontsize=20, fontweight='bold')
    plt.ylabel('Output Length', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{args.pred_path}/{model_name}/{".".join(file.split(".")[:-1])}_scatter.png')
json.dump(result, open(f"{args.pred_path}/{model_name}/result.json","w"), ensure_ascii=False, indent=4)