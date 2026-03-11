import os
from datasets import load_dataset
import torch
import json
import transformers
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig

from morphkv.monkeypatch import patch_morphkv
print("Applying MorphKV patches...")
patch_morphkv()
print("MorphKV patches applied successfully!")

from tqdm import tqdm
import numpy as np
import random
import argparse
import warnings
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.profiler

import logging
import logging.handlers
from pathlib import Path

DATASET_SPLITS = {
    1: ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    2: ["hotpotqa", "2wikimqa", "musique", "dureader"],
    3: ["gov_report", "qmsum", "multi_news", "vcsum", "trec"],
    4: ["triviaqa", "samsum", "lsht", "passage_count"],
    5: ["passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"],
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["phi4-unsloth","phi4","mistral","qwen2.5","llama3.1-8b-instruct","llama2-7b-chat-4k", "llama-2-7B-32k-instruct", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--pred_path', type=str, default="pred")
    parser.add_argument('--morph_type', type=str, default="max_fused")
    parser.add_argument('--len', "-l", type=int, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--ablation', action='store_true', help="Evaluate on LongBench Ablation")
    parser.add_argument("--window_size", "-ws", type=int, default=3, help="Window size for morphkv")
    parser.add_argument("--max_capacity", "-mc", type=float, default=100, help="Max cache capacity")
    parser.add_argument("--evict_after", "-ea", type=float, default=1.0, help="Evict after exceeding this times the KV limit")
    
    parser.add_argument("--fuse_temperature", "-ft", type=float, default=1.0, help="Temperature for fuse")
    parser.add_argument("--no_morph", action='store_true', help="Disable morphkv")  # Updated line
    parser.add_argument("--use_attn_offsets", action='store_true', help="Use attn offsets")
    parser.add_argument("--imp_budget", "-ib", type=float, default=0.5, help="Importance budget for morphkv")
    parser.add_argument("--pre_rope", "-pr", action='store_true', help="Use pre-rope for morphkv")
    parser.add_argument("--score_percentile", "-sp", type=float, default=0.9, help="Score percentile for morphkv")
    parser.add_argument("--batch_size", "-bs", type=int, default=1, help="Batch size for inference (>1 enables batched generation)")
    parser.add_argument("--dataset_split", "-ds", type=int, default=None, choices=list(DATASET_SPLITS.keys()),
                        help="Dataset split index (1-5) for parallel execution across machines")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def _prepare_prompt(json_obj, prompt_format, model_name, tokenizer, max_length, dataset):
    """Pre-process a single sample: format, truncate, and apply chat template."""
    prompt = prompt_format.format(**json_obj)
    if "qwen" in model_name:
        prompt = f"[INST]{prompt}[/INST]"

    if "chatglm3" in model_name:
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    else:
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                 tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        prompt = build_chat(tokenizer, prompt, model_name)

    return prompt


def get_pred(data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path, args):
    device = torch.device(f'cuda')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, args)
    logger = logging.getLogger(__name__)

    batch_size = args.batch_size

    all_prompts = [
        _prepare_prompt(obj, prompt_format, model_name, tokenizer, max_length, dataset)
        for obj in data
    ]

    generate_kwargs = dict(
        max_new_tokens=max_gen,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
    )
    if dataset == "samsum":
        generate_kwargs['min_new_tokens'] = 1
        generate_kwargs['eos_token_id'] = [
            tokenizer.eos_token_id,
            tokenizer.encode("\n", add_special_tokens=False)[-1],
        ]
    elif "phi4" in model_name or "qwen" in model_name:
        generate_kwargs['min_new_tokens'] = 1

    for batch_start in tqdm(range(0, len(all_prompts), batch_size)):
        batch_prompts = all_prompts[batch_start : batch_start + batch_size]
        batch_data = data[batch_start : batch_start + batch_size]

        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                inputs = tokenizer(batch_prompts[0], truncation=False, return_tensors="pt").to(device)
            else:
                inputs = batch_prompts[0].to(device)
        else:
            inputs = tokenizer(
                batch_prompts,
                truncation=False,
                return_tensors="pt",
                padding=batch_size > 1,
            ).to(device)

        input_length = inputs.input_ids.shape[-1]

        outputs = model.generate(**inputs, **generate_kwargs)

        for i, (output, json_obj) in enumerate(zip(outputs, batch_data)):
            pred = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            pred = post_process(pred, model_name)
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                }, f, ensure_ascii=False)
                f.write('\n')

        if "prof" in args.morph_type:
            break

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

from importlib.metadata import version
def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def load_model_and_tokenizer(path, model_name, device, args):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama" in model_name or "qwen" in model_name or "mistral" in model_name or "phi4" in model_name:
        # replace_llama_attn_with_flash_attn()
        # tokenizer = LlamaTokenizer.from_pretrained(path)
        # model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
        # Set cache directory
        transformers_version = check_version()
        version_list = ['4.45']
        warning_flag = True
        for version in version_list:
            if version in transformers_version:
                warning_flag = False
                break
        assert warning_flag==False, f"Transformers version {transformers_version} is not compatible with MorphKV. MorphKV is tested with Transformers version {version_list}. Please install this by: pip install transformers==4.45.0"
        cache_dir = "../model_chkpts/"
        os.makedirs(cache_dir, exist_ok=True)

        # Load the model and tokenizer
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)

        # Create a custom configuration with morphkv parameters
        config = AutoConfig.from_pretrained(path, cache_dir=cache_dir)
        config._attn_implementation = "flash_attention_2"
        
        config.morphkv = None if args.no_morph else {
            'window_size': int(args.window_size),
            'morph_type': args.morph_type,
            'evict_after': args.evict_after,
            'max_capacity': args.max_capacity,
            'fuse_temperature': args.fuse_temperature,
            'use_attn_offsets': args.use_attn_offsets,
            'imp_budget': args.imp_budget,
            'pre_rope': args.pre_rope,
            'score_percentile': args.score_percentile
        }
        print(f"MorphKV is: {config.morphkv}")
        
        model = AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            cache_dir=cache_dir).to(device)

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif "longchat" in model_name or "vicuna" in model_name:
        assert False, "Models unsupported\n"
    model = model.eval()
    
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    # world_size = torch.cuda.device_count()
    # mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    pred_path = args.pred_path
    # define your model
    max_length = model2maxlen[model_name] if args.len is None else args.len
    if args.ablation:
        datasets = ["narrativeqa", "2wikimqa", "hotpotqa", "multifieldqa_en", "musique", "passage_retrieval_en"]
    elif args.e:
        datasets = ["qasper","2wikimqa","hotpotqa","multi_news","passage_retrieval_en","lcc"] #["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            #"trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        if args.dataset_split is not None:
            datasets = DATASET_SPLITS[args.dataset_split]
        else:
            datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                        "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                        "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    if args.dataset!=None:
        datasets = [args.dataset]
    for dataset in datasets:
        # assert dataset in datasets, "Unknown Dataset Error!"
        if args.ablation:
            if not os.path.exists("pred_mem"):
                os.makedirs("pred_mem")
            # data = load_dataset('THUDM/LongBench', f"{dataset}", split='test')
            data = [json.loads(line) for line in open(f"data/{dataset}.jsonl", "r")]
            if not os.path.exists(f"pred_mem/{model_name}"):
                os.makedirs(f"pred_mem/{model_name}")
            out_path = f"pred_mem/{model_name}/{dataset}_ws{args.window_size}_mc{args.max_capacity}_ft{args.fuse_temperature}_ao_{args.use_attn_offsets}_morphkv_{not(args.no_morph)}_type_{args.morph_type}_prerope_{args.pre_rope}_len{args.len}.jsonl"
            logfile = f"pred_mem/{model_name}/{dataset}_ws{args.window_size}_mc{args.max_capacity}_ft{args.fuse_temperature}_ao_{args.use_attn_offsets}_morphkv_{not(args.no_morph)}_type_{args.morph_type}_prerope_{args.pre_rope}_len{args.len}.log"
        elif args.e:
            # data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            data = [json.loads(line) for line in open(f"data/{dataset}_e.jsonl", "r")]
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}_ws{args.window_size}_mc{args.max_capacity}_ft{args.fuse_temperature}_ao_{args.use_attn_offsets}_morphkv_{not(args.no_morph)}_type_{args.morph_type}_prerope_{args.pre_rope}_len{args.len}.jsonl"
            logfile = f"pred_e/{model_name}/{dataset}_ws{args.window_size}_mc{args.max_capacity}_ft{args.fuse_temperature}_ao_{args.use_attn_offsets}_morphkv_{not(args.no_morph)}_type_{args.morph_type}_prerope_{args.pre_rope}_len{args.len}.log"
        else:
            # For .jsonl files (JSON Lines format):
            data = [json.loads(line) for line in open(f"data/{dataset}.jsonl", "r")]#load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"{pred_path}/{model_name}"):
                os.makedirs(f"{pred_path}/{model_name}")
            out_path = f"{pred_path}/{model_name}/{dataset}_ws{args.window_size}_mc{args.max_capacity}_ft{args.fuse_temperature}_ao_{args.use_attn_offsets}_morphkv_{not(args.no_morph)}_type_{args.morph_type}_prerope_{args.pre_rope}_len{args.len}.jsonl"
            logfile = f"{pred_path}/{model_name}/{dataset}_ws{args.window_size}_mc{args.max_capacity}_ft{args.fuse_temperature}_ao_{args.use_attn_offsets}_morphkv_{not(args.no_morph)}_type_{args.morph_type}_prerope_{args.pre_rope}_len{args.len}.log"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                lines = len(f.readlines())
                if(lines==len(data_all)):
                    print(f"Skipping {dataset}\n")
                    continue
                else:
                    data_all = data_all[lines:]
        except FileNotFoundError:
            pass
        # data_subsets = [data_all[i::world_size] for i in range(world_size)]
        
        # import logging
        # log_queue = mp.Queue()
        # listener = mp.Process(target=listener_process, args=(log_queue, logfile))
        # listener.start()

        # processes = []
        # for rank in range(world_size):
        #     p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
        #                 max_gen, prompt_format, dataset, device, model_name, model2path, out_path, args, log_queue))
        #     p.start()
        #     processes.append(p)
        logging.basicConfig(filename=logfile,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
        get_pred(data_all, 
                 max_length,
                 max_gen, 
                 prompt_format, 
                 dataset, 
                 device, 
                 model_name, 
                 model2path, 
                 out_path, 
                 args)