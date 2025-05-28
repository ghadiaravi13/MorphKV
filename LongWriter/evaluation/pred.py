import logging
import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
import re

from morphkv.monkeypatch import patch_morphkv
print("Applying MorphKV patches...")
patch_morphkv()
print("MorphKV patches applied successfully!")

os.environ['HF_HOME'] = "/home/shared/model_chkpts"
cache_dir = "/home/shared/model_chkpts"

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["phi4-unsloth","phi4","mistral","qwen2.5","llama3.1-8b-instruct","llama2-7b-chat-4k", "llama-2-7B-32k-instruct", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--morph_type', type=str, default="max_fused")
    parser.add_argument('--len', "-l", type=int, default=None)
    parser.add_argument("--window_size", "-ws", type=int, default=3, help="Window size for morphkv")
    parser.add_argument("--max_capacity", "-mc", type=float, default=1024.0, help="Max capacity for morphkv")
    parser.add_argument("--evict_after", "-ea", type=float, default=1.0, help="Evict after exceeding this times the KV limit")
    parser.add_argument("--no_morph", action='store_true', help="Disable morphkv")  # Updated line
    parser.add_argument("--pred_path", type=str, default="pred")
    return parser.parse_args(args)

def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)
    
    total_count = chinese_char_count + english_word_count
    
    return total_count

def get_pred(data, path, max_new_tokens, temperature, tokenizer, fout, args):
    device = torch.device(f'cuda')
    logger = logging.getLogger(__name__)
    # model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    # Create a custom configuration with morphkv parameters
    config = AutoConfig.from_pretrained(path, cache_dir=cache_dir)
    config._attn_implementation = "flash_attention_2"
    if args.morph_type=='indp':
        assert args.window_size==1, "Window size > 1 is not supported for independent morphkv. Try 'max_fused' instead"
    config.morphkv = None if args.no_morph else {
        'window_size': int(args.window_size),
        'max_capacity': int(args.max_capacity),
        'morph_type': args.morph_type,
        'evict_after': args.evict_after
    }
    print(f"morphkv is: {config.morphkv}")
    
    # Load the model with the custom configuration
    model = AutoModelForCausalLM.from_pretrained(path,
                                                config=config,
                                                torch_dtype=torch.bfloat16,
                                                cache_dir=cache_dir).to(device)
    
    model = model.eval()
    with torch.no_grad():
        for dt in tqdm(data):
            prompt = dt['prompt']
            if "llama" in path.lower() or "mistral" in path.lower() or "phi" in path.lower() or "qwen" in path.lower():
                prompt = f"[INST]{prompt}[/INST]"
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                context_length = input.input_ids.shape[-1]
                if "phi" in path.lower():
                    output = model.generate(
                        **input,
                        max_new_tokens=min(max_new_tokens,dt['length']*2),
                        num_beams=1,
                        do_sample=True,
                        temperature=temperature,
                        min_length=context_length+1,
                    )[0]
                else:
                    output = model.generate(
                        **input,
                        max_new_tokens=min(max_new_tokens,dt['length']*2),
                        num_beams=1,
                        do_sample=True,
                        temperature=temperature
                    )[0]
                response = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            else:
                response, history = model.chat(tokenizer, prompt, history=[], max_new_tokens=max_new_tokens, temperature=temperature)
            dt["response_length"] = count_words(response)
            dt["response"] = response
            fout.write(json.dumps(dt, ensure_ascii=False)+'\n')
            fout.flush()
            # print(response)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open('../../model2path.json','r'))
    model_name = args.model#'LongWriter-glm4-9b' # LongWriter-llama3.1-8b
    path = model2path[model_name]#"THUDM/LongWriter-glm4-9b" # THUDM/LongWriter-llama3.1-8b
    os.makedirs(f"{args.pred_path}/{model_name}", exist_ok=True)
    fout = open(f"{args.pred_path}/{model_name}/preds_ws{args.window_size}_mc{args.max_capacity}_ea{args.evict_after}_morph_{not(args.no_morph)}_type_{args.morph_type}_len{args.len}.jsonl", 'w', encoding='utf-8')
    
    logfile = f"{args.pred_path}/{model_name}/preds_ws{args.window_size}_mc{args.max_capacity}_ea{args.evict_after}_morph_{not(args.no_morph)}_type_{args.morph_type}_len{args.len}.log"
    logging.basicConfig(filename=logfile,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
    
    max_new_tokens = 32768
    temperature = 0.5
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir=cache_dir)

    with open('longbench_write_en.jsonl', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    get_pred(data, path, max_new_tokens, temperature, tokenizer, fout, args)
        
