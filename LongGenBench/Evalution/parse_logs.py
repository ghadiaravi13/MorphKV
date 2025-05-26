import argparse
import datetime
import json
import os
import pandas as pd

from networkx import cut_size

attn_heads_dict = {"llama3.1-8b-instruct": 32, "phi4": 40, "mistral": 32, "qwen2.5": 28, "phi4-unsloth": 40}
layers_dict = {"llama3.1-8b-instruct": 32, "phi4": 40, "mistral": 32, "qwen2.5": 28, "phi4-unsloth": 40}
key_heads_dict = {"llama3.1-8b-instruct": 8, "phi4": 10, "mistral": 8, "qwen2.5": 4, "phi4-unsloth": 10}
hid_dim_dict = {"llama3.1-8b-instruct": 128, "phi4": 128, "mistral": 128, "qwen2.5": 128, "phi4-unsloth": 128}



def parse(args,dataset="NA"):
    print(dataset)
    model = args.model
    files = sorted(os.listdir(f"preds_rerun_new/{model}/"),key=lambda x: "hopf_False" not in x)
    if args.e:
        files = sorted(os.listdir(f"pred_e/{model}/"),key=lambda x: "hopf_False" not in x)
    data_list = [dataset]
    if dataset=="all":
        data_list = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    data = dict()
    for dataset in data_list:
        for file in files:
            if ".log" in file and (dataset in file or dataset=="NA"):
                if args.e:
                    logfile = open(f"pred_e/{model}/"+file,"r")
                    res = open(f"pred_e/{model}/"+file[:-3]+"jsonl","r").readlines()
                else:
                    logfile = open(f"preds_rerun_new/{model}/"+file,"r")
                    res = open(f"preds_rerun_new/{model}/"+file[:-3]+"jsonl","r").readlines()
                last_time = 0
                run_name = file.split("/")[-1][:-4]
                
                n_attn_heads = attn_heads_dict[model]
                n_key_heads = n_attn_heads if "snapkv" in file or "h2o" in file else key_heads_dict[model]
                n_layers = layers_dict[model]
                hid_dim_per_head = hid_dim_dict[model]
                
                # if "hopf_False" in file:
                #     data = dict()
                data[run_name] = dict()
                data[run_name]['gen_time(s)'] = []
                data[run_name]['time_per_token'] = []
                data[run_name]['inp_size'] = [] #if "hopf_False" in file else inp_size
                data[run_name]['out_size'] = [] #if "hopf_False" in file else out_size
                data[run_name]['resp_len'] = []
                data[run_name]['KV_size'] = []
                data[run_name]['attn_size'] = []
                data[run_name]['total_cache'] = []

                i = 0
                for l in logfile.readlines():
                    if(l!="\n"):
                        curr_time = datetime.datetime.strptime(l.split(" - ")[0], "%Y-%m-%d %H:%M:%S,%f").timestamp()
                        data[run_name]['gen_time(s)'].append(curr_time - last_time)
                        # if "hopf_False" in file:
                        data[run_name]['inp_size'].append(int(l.split("Input size: ")[1].split(" ")[0]))
                        data[run_name]['out_size'].append(int(l.split("key': ")[1].split(",")[0]) - int(l.split("Input size: ")[1].split(" ")[0]))
                        try:
                            data[run_name]['resp_len'].append(int(l.split("len': ")[1].split("}")[0]))
                        except IndexError:
                            data[run_name]['resp_len'].append(data[run_name]['out_size'][-1])
                        data[run_name]['time_per_token'].append(data[run_name]['gen_time(s)'][-1] / data[run_name]['resp_len'][-1])
                        data[run_name]['KV_size'].append(4 * hid_dim_per_head * n_key_heads * n_layers * int(l.split("key': ")[1].split(",")[0]) / 1000000)
                        # if "hopf_False" in file:
                        #     data[run_name]['attn_size'].append(4 * n_attn_heads * n_layers * data[run_name]['out_size'][i] * int(l.split("attn_wts': ")[1].split("}")[0]) / 1000000)
                        # else:
                        try:
                            data[run_name]['attn_size'].append(4 * n_key_heads * n_layers * int(l.split("key': ")[1].split(",")[0]) * int(l.split("attn_wts': ")[1].split(",")[0]) / 1000000)
                        except ValueError: #for older logs, len is missing, so attn_wts is the last item
                            data[run_name]['attn_size'].append(4 * n_key_heads * n_layers * int(l.split("key': ")[1].split(",")[0]) * int(l.split("attn_wts': ")[1].split("}")[0]) / 1000000)
                        data[run_name]['total_cache'].append(data[run_name]['KV_size'][-1] + data[run_name]['attn_size'][-1])
                        last_time = curr_time
                        i+=1
                inp_size = data[run_name]['inp_size']
                out_size = data[run_name]['out_size']
    return data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert results JSON to CSV format')
    parser.add_argument("--model",default=None, type=str, help='Model name for the result parsing')
    parser.add_argument("--dataset", "-d", default="", type=str, help='Dataset name for the result parsing')
    parser.add_argument("--e",action='store_true',help='results for longbench-E')

    
    # Parse arguments
    args = parser.parse_args()
    print(args.dataset)
    
    # Convert file
    # try:
    data = parse(args,dataset=args.dataset)
    
    # import pdb; pdb.set_trace()
    all_indices = range(550)
    df = dict()
    for k,v in data.items():
        try:
            df[k] = pd.DataFrame(v)#.reindex(all_indices)
        except:
            print(f"Skipped {k}")
    df = pd.concat(df, axis=1)
    # df = pd.concat({k: pd.DataFrame(v).reindex(all_indices) for k, v in data.items()}, axis=1)
    df = df.fillna(0, inplace = False)
    # import pdb; pdb.set_trace()
    run_names = [c[0] for c in df.columns]
    configs = list(set([r for r in run_names]))
    data_names = [args.model]#list(set([r.split("_ws")[0] for r in run_names]))
    cache_data = pd.DataFrame(index=configs,columns=data_names)
    for c in df.columns:
        if "total_cache" in c:
            cache_data.loc[c[0]][args.model] = sum(df[c])
    
    time_data = pd.DataFrame(index=configs,columns=data_names)
    for c in df.columns:
        if "time_per_token" in c:
            time_data.loc[c[0]][args.model] = sum(df[c][1:])

    if args.e:
        with open(f"pred_e/{args.model}/{args.dataset}_parsed_log_data.json","w") as f:
            json.dump(data,f)
            try:
                with pd.ExcelWriter(f"pred_e/{args.dataset}_parsed_log_data.xlsx",mode='a',if_sheet_exists='replace') as writer:
                    df.to_excel(writer,sheet_name=args.model)
            except FileNotFoundError:
                with pd.ExcelWriter(f"pred_e/{args.dataset}_parsed_log_data.xlsx",mode='w') as writer:
                    df.to_excel(writer,sheet_name=args.model)
            try:
                with pd.ExcelWriter(f"pred_e/{args.dataset}_cache_data.xlsx",mode='a',if_sheet_exists='replace') as writer:
                    cache_data.to_excel(writer,sheet_name=args.model)
            except FileNotFoundError:
                with pd.ExcelWriter(f"pred_e/{args.dataset}_cache_data.xlsx",mode='w') as writer:
                    cache_data.to_excel(writer,sheet_name=args.model)
            print(f"Successfully parsed logs")
        f.close()
    else:
        with open(f"preds_rerun_new/{args.model}/{args.dataset}_parsed_log_data.json","w") as f:
            json.dump(data,f)
            try:
                with pd.ExcelWriter(f"preds_rerun_new/{args.dataset}_parsed_log_data.xlsx",mode='a',if_sheet_exists='replace') as writer:
                    df.to_excel(writer,sheet_name=args.model)
            except FileNotFoundError:
                with pd.ExcelWriter(f"preds_rerun_new/{args.dataset}_parsed_log_data.xlsx",mode='w') as writer:
                    df.to_excel(writer,sheet_name=args.model)
            try:
                with pd.ExcelWriter(f"preds_rerun_new/{args.dataset}_cache_data.xlsx",mode='a',if_sheet_exists='replace') as writer:
                    cache_data.to_excel(writer,sheet_name=args.model)
            except FileNotFoundError:
                with pd.ExcelWriter(f"preds_rerun_new/{args.dataset}_cache_data.xlsx",mode='w') as writer:
                    cache_data.to_excel(writer,sheet_name=args.model)
            
            try:
                with pd.ExcelWriter(f"preds_rerun_new/{args.dataset}_time_data.xlsx",mode='a',if_sheet_exists='replace') as writer:
                    time_data.to_excel(writer,sheet_name=args.model)
            except FileNotFoundError:
                with pd.ExcelWriter(f"preds_rerun_new/{args.dataset}_time_data.xlsx",mode='w') as writer:
                    time_data.to_excel(writer,sheet_name=args.model)


            print(f"Successfully parsed logs")
        f.close()
    # except Exception as e:
    #     print(f"Error converting file: {str(e)}")

if __name__ == "__main__":
    main()
    



                