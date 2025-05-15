import json
import csv
from numpy import add
import pandas as pd
import argparse

def json_to_csv(json_data, args):
    # Initialize lists to store the data
    rows = []
    
    # Process each model configuration
    for model_name, metrics in json_data.items():
        if args.e:
            row = {
                'model_name': model_name,
                '0-4k': metrics['0-4k'],
                '4-8k': metrics['4-8k'],
                '8k+': metrics['8k+']
            }
        else:
            row = {
                'model_name': model_name,
                'score': metrics
            }
        rows.append(row)
    
    # Convert to DataFrame for easy CSV export
    df = pd.DataFrame(rows)
    
    # Sort by model name for better readability
    df = df.sort_values('model_name')
    
    # Save to CSV
    df.to_csv(f'{args.output}.csv', index=False)
    
    return df
parser = argparse.ArgumentParser()
parser.add_argument("--model",default=None, type=str, help='Model name for the result parsing')
parser.add_argument("--e",action='store_true',help='results for longbench-E')
parser.add_argument("--output","-o",type=str) 
args = parser.parse_args()
file = f"pred_e/{args.model}/result.json" if args.e else f"pred/{args.model}/result.json"
json_data = json.load(open(file,'r'))
# Convert and save
df = json_to_csv(json_data,args)
print("CSV file has been created successfully!")
print("\nFirst few rows of the data:")
print(df.head())
df.to_csv(file[:-4]+"csv")