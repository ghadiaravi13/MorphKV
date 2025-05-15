import json
import csv
import argparse
import re

def parse_key(key):
    """Parse a key string into its components."""
    # import pdb; pdb.set_trace()
    # Split the key by underscores
    parts = key.replace('.0','').split('_')
    
    # Extract dataset name (everything before 'ws' followed by digits)
    match = re.search(r'ws\d+', key)
    if match:
        dataset_end = parts.index(match.group(0))
        dataset = '_'.join(parts[:dataset_end])
    else:
        raise ValueError("Pattern 'ws' followed by digits not found in key")

    # Extract other components
    try:
        # import pdb; pdb.set_trace()
        components = {
        'window_size': int(parts[parts.index(re.search(r'ws\d+', key).group(0))].replace('ws', '')),
        'max_capacity': (int(parts[parts.index(re.search(r'st\d+', key).group(0))].replace('st', ''))+1)*int(parts[parts.index(re.search(r'ws\d+', key).group(0))].replace('ws', '')),
        'num_sinks': int(parts[parts.index(re.search(r'snks\d+', key).group(0))].replace('snks', '')),
        'hopf': parts[parts.index('hopf') + 1] == 'True',
        'fusion_type': "_".join(parts[parts.index('type') + 1:parts.index(re.search('len[a-zA-Z\d]*',key).group(0))]),
        'length': int(parts[parts.index(re.search(r'len\d+', key).group(0))].replace('len', '')) if re.search(r'len\d+', key) else None,
        'gumbel': parts[parts.index(re.search(r'gbl[a-zA-Z]*', key).group(0))] == 'True' if re.search(r'gbl[a-zA-Z]*', key).group(0) else False
        }
        if "snap" in components['fusion_type']:
            components['max_capacity'] = (components['max_capacity']/components['window_size'])-1
    except:
        components = {
        'window_size': int(parts[parts.index(re.search(r'ws\d+', key).group(0))].replace('ws', '')),
        'max_capacity': int(parts[parts.index(re.search(r'mc\d+', key).group(0))].replace('mc', '')),
        'num_sinks': int(parts[parts.index(re.search(r'snks\d+', key).group(0))].replace('snks', '')),
        'hopf': parts[parts.index('hopf') + 1] == 'True',
        'fusion_type': "_".join(parts[parts.index('type') + 1:parts.index(re.search('len[a-zA-Z\d]*',key).group(0))]),
        'length': int(parts[parts.index(re.search(r'len\d+', key).group(0))].replace('len', '')) if re.search(r'len\d+', key) else None,
        'gumbel': parts[parts.index(re.search(r'gbl[a-zA-Z]*', key).group(0))] == 'True' if re.search(r'gbl[a-zA-Z]*', key).group(0) else False
        }
    
    return dataset, components

def convert_json_to_csv(input_file):
    """Convert JSON results file to CSV format."""
    # Read JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Prepare CSV data
    csv_rows = []
    headers = ['dataset', 'window_size', 'max_capacity', 'num_sinks', 
              'hopf', 'fusion_type', 'length', 'gumbel', 'value']
    
    # Process each entry
    for key, value in data.items():
        dataset, components = parse_key(key)
        row = {
            'dataset': dataset,
            'value': value,
            **components
        }
        csv_rows.append(row)
    
    # Write to CSV file
    with open(input_file[:-4]+"csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    import pandas as pd
    df = pd.read_csv(input_file[:-4]+"csv")
    df['config'] = ["Hopf_ws"+ str(df['window_size'].iloc[i]) + "_mc" + str(df['max_capacity'].iloc[i]) + "_fused_" + str(df['fusion_type'].iloc[i]) if df['hopf'].iloc[i] else "No_Hopf" for i in range(len(df))]
    # df[['dataset','config','value']].to_csv(input_file[:-5]+"_summary.csv")

    new_df = pd.DataFrame(columns=sorted(df['dataset'].unique()),index=sorted(df['config'].unique()))
    # import pdb; pdb.set_trace()
    for c in df['config'].unique():
        # print(new_df.shape,len(df[df['config']==c]['value'].values))
        for d in new_df.columns:
            try:
                new_df[d].loc[c] = df[(df['config']==c) & (df['dataset']==d)]['value'].values[0]#list(df[df['config']==c].sort_values(by='dataset')['value'].values)
            except IndexError:
                new_df[d].loc[c] = 0
    # import pdb; pdb.set_trace()
    new_df.to_csv(input_file[:-5]+"_summary.csv")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert results JSON to CSV format')
    parser.add_argument("--model",default=None, type=str, help='Model name for the result parsing')
    parser.add_argument("--e",action='store_true',help='results for longbench-E')
    parser.add_argument("--ablation",action='store_true',help='results for longbench ablation mem')

    
    # Parse arguments
    args = parser.parse_args()
    
    file = f"pred_e/{args.model}/result.json" if args.e else f"pred/{args.model}/result.json"
    if args.ablation:
        file = f"pred_mem/{args.model}/result.json"
    # Convert file
    # try:
    convert_json_to_csv(file)
    print(f"Successfully converted {file} to {file[:-4]}.csv")
    # except Exception as e:
    #     print(f"Error converting file: {str(e)}")

if __name__ == "__main__":
    main()