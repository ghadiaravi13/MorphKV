import argparse
import logging
import time
import json
# from vllm import LLM, SamplingParams
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import transformers
import os
from tqdm import tqdm

import sys
sys.path.append("/home/rhg659/MorphKV")

from morphkv.monkeypatch import patch_mistral
print("Applying MorphKV patches...")
patch_mistral()
print("MorphKV patches applied successfully!")

os.environ['HF_HOME'] = "/home/shared/model_chkpts/"
cache_dir = "/home/shared/model_chkpts/"

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM with command line arguments.')
    parser.add_argument('--model', type=str, default=None, choices=["phi4-unsloth","phi4","mistral","qwen2.5","qwen2.5-30b","llama3.1-8b-instruct","llama2-7b-chat-4k", "llama-2-7B-32k-instruct", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--morph_type', type=str, default="max_fused")
    parser.add_argument('--len', "-l", type=int, default=None)
    parser.add_argument("--window_size", "-ws", type=int, default=3, help="Window size for MorphKV")
    parser.add_argument("--max_capacity", "-mc", type=float, default=100, help="Window size for MorphKV")
    parser.add_argument("--evict_after", "-ea", type=float, default=1.0, help="Evict after exceeding this times the KV limit")
    parser.add_argument("--sim_threshold", "-st", type=float, default=20.0, help="Similarity threshold for MorphKV")
    parser.add_argument("--no_morph", action='store_true', help="Disable MorphKV")  # Updated line
    parser.add_argument("--num_samples", "-ns", type=int, default=10, help="Num of samples to eval on")

    parser.add_argument('--max_length', type=int, default=8000, help='Maximum length of generation.')
    parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path.')
    parser.add_argument('--input_file', type=str, required=True, help='input file path.')
    parser.add_argument('--preds_path', default="preds_greedy", type=str, required=True, help='preds file path.')

  
    args = parser.parse_args()
    return args

# Process output to split blocks and count words
def process_output(output: str) -> dict:
    blocks = output.split('#*#')
    word_count = len(output.split())
    return {"blocks": blocks, "word_count": word_count}

def read_json(file_path):
    with open(file_path, 'r',encoding='utf-8') as file:
        return json.load(file)
    
def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def save_to_json(data: list, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_inputs(filename: str) -> list:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Combine inputs, results and word counts and save them
def process_and_save_results(inputs: list, results: list, filename: str) -> None:
    combined = []
    for input_data, result_data in zip(inputs, results):
        combined.append({
            "input": input_data["prompt"],
            "checks_once": input_data["checks_once"],
            "checks_range": input_data["checks_range"],
            "checks_periodic": input_data["checks_periodic"],
            "type": input_data["type"],
            "number": input_data['number'],
            "output_blocks": result_data["blocks"],
            "word_count": result_data["word_count"]  # Adding word count here
        })
    save_to_json(combined, filename)

args = parse_args()

#input_file = '/home/yuhao/THREADING-THE-NEEDLE/Dataset/Dataset_short.json'
inputs = load_inputs(args.input_file)

# sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=args.max_length, seed=42, repetition_penalty = 1.005)
# sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=args.max_length, seed=6211027, stop = '*** finished')

model2path = json.load(open('../../model2path.json','r'))
model_name = args.model#'LongWriter-glm4-9b' # LongWriter-llama3.1-8b
model_path = model2path[model_name]

# Load the model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path, cache_dir=cache_dir)
config._attn_implementation = "flash_attention_2"
if args.morph_type=='indp':
    assert args.window_size==1, "Window size > 1 is not supported for independent MorphKV. Try 'max_fused' instead"
config.morphkv = None if args.no_morph else {
    'window_size': int(args.window_size),
    'max_capacity': int(args.max_capacity),
    'morph_type': args.morph_type,
    'evict_after': args.evict_after,
    'max_capacity': args.max_capacity,
}
print(f"MorphKV is: {config.morphkv}")

# Load the model with the custom configuration
device = torch.device(f'cuda')
    
model = AutoModelForCausalLM.from_pretrained(model_path,
                                            config=config,
                                            torch_dtype=torch.bfloat16,
                                            cache_dir=cache_dir).to(device)

# model = torch.compile(model)

inputs_used = []
results = []

os.makedirs(f"{args.preds_path}/{model_name}", exist_ok=True)
fout = open(f"{args.preds_path}/{model_name}/preds_ns{args.num_samples}_ws{args.window_size}_mc{args.max_capacity}_ea{args.evict_after}_morph_{not(args.no_morph)}_type_{args.morph_type}_len{args.len}.jsonl", 'w', encoding='utf-8')

logfile = f"{args.preds_path}/{model_name}/preds_ns{args.num_samples}_ws{args.window_size}_mc{args.max_capacity}_ea{args.evict_after}_morph_{not(args.no_morph)}_type_{args.morph_type}_len{args.len}.log"
logging.basicConfig(filename=logfile,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)

num_week = 0 ; num_floor = 0; num_menu = 0; num_block = 0;
for input_data in tqdm(inputs):
    if input_data['type']=="Week" and num_week>=args.num_samples or \
       input_data['type']=="Floor" and num_floor>=args.num_samples or \
       input_data['type']=="Menu Week" and num_menu>=args.num_samples or \
       input_data['type']=="Block" and num_block>=args.num_samples :
        continue
    else:
        if input_data['type']=="Week": num_week+=1
        elif input_data['type']=="Floor": num_floor+=1
        elif input_data['type']=="Menu Week": num_menu+=1
        elif input_data['type']=="Block": num_block+=1

        prompt = input_data['prompt']
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        start_time = time.time()
        output = model.generate(**input, max_new_tokens=args.max_length, do_sample=False, num_beams=1)
        inference_time = time.time() - start_time
        
        print(f"Inference time: {inference_time:.2f} seconds")
        # import pdb; pdb.set_trace()
        input_data['response'] = tokenizer.decode(output[0][context_length:], skip_special_tokens=True)
        fout.write(json.dumps(input_data, ensure_ascii=False)+'\n')
        fout.flush()
        
        inputs_used.append(input_data)
        results.append(process_output( input_data['prefix']+ input_data['response']))
    if "prof" in args.morph_type: break

process_and_save_results(inputs_used, results, args.output_file)
print(f"\nSaved result to {args.output_file}")
