from transformers import AutoTokenizer
import os
import json
import re

os.environ['HF_HOME'] = "/home/shared/model_chkpts/"
cache_dir = "/home/shared/model_chkpts/"

model_name="mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

def count_tokens(text):
    """
    Count tokens in text using the specified model's tokenizer.
    
    Args:
        text (str): The text to tokenize and count
        model_name (str): HuggingFace model name for the tokenizer
        
    Returns:
        int: Number of tokens in the text
    """
    # Load the tokenizer
    # print(f"Loading tokenizer for {model_name}...")
    
    # Tokenize the text
    # print("Tokenizing text...")
    tokens = tokenizer.encode(text)
    
    # Print the result
    token_count = len(tokens)
    # print(f"Token count: {token_count}")
    
    # # Optional: Detailed information
    # print(f"Approximate words: {len(text.split())}")
    # print(f"Characters: {len(text)}")
    # print(f"Average tokens per word: {token_count / max(1, len(text.split())):.2f}")
    
    return token_count

def main():
    # Read text from a file
    try:
        filepath = "Evalution/preds_rerun_new/mistral/preds_ns5_ws200_mc4000_ea1.002_snks0_hopf_True_type_max_fused_opt_qcache_new_burst_lenNone_gblFalse.jsonl"
        logpath = "Evalution/preds_rerun_new/mistral/preds_ns5_ws200_mc4000_ea1.002_snks0_hopf_True_type_max_fused_opt_qcache_new_burst_lenNone_gblFalse.log"
        
        new_log = open(filepath[:-5]+".lognew","w")

        logs = open(logpath,"r")
        responses = open(filepath,"r")
        loglines = []
        for l in logs.readlines():
            if l!="\n": loglines.append(l)

        for d,l in zip(responses.readlines(),loglines):
            text = json.loads(d)['response']

            # try:
            #     while True:
            #         line = input()
            #         text += line + "\n"
            # except EOFError:
            #     pass

            input_size_match = re.search(r"Input size: (\d+)", l)
            if not input_size_match:
                print("Warning: Could not find 'Input size' in the log line.")
                input_size = 0
            else:
                input_size = int(input_size_match.group(1))
            
        # Count tokens
            tokens = count_tokens(text)
            pattern = r"'len': \d+"

            # Calculate the final len value
            final_len = tokens + input_size  - 3
    # Replacement string with the new token count
            replacement = f"'len': {final_len}"
            
            # Perform the replacement
            updated_line = re.sub(pattern, replacement, l)
            new_log.write(updated_line+"\n")
        
        new_log.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()