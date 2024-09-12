# merge_attention_ppl
import jsonlines
import json
import numpy as np
from tqdm import tqdm
import copy
if __name__ == '__main__':

    attention_file_path =  "../data/seg_attention.jsonl"
    chunk_ppl_file_path = "../data/seg_ppl.jsonl"
    output_file_path = "../data/cas.jsonl"

    
    attentions = []
    with open(attention_file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(tqdm(lines, desc="Processing lines", unit="line")):
            try:
                attentions.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error parsing JSON at line {i + 1}")
                print(line)
                break
    chunk_ppls = []
    with open(chunk_ppl_file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(tqdm(lines, desc="Processing lines", unit="line")):
            try:
                chunk_ppls.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error parsing JSON at line {i + 1}")
                print(line)
                break

    final_results = []
    for att in tqdm(attentions):
        result_item = copy.deepcopy(att)
        for ppl in chunk_ppls:
            if result_item['id'] == ppl['id']:
                result_item['chunk_ppl'] = ppl['single_ppl']
                
            else:
                pass
        final_results.append(result_item)
        

    with open(output_file_path, 'w') as jsonl_file:
        for entry in final_results:
            jsonl_file.write(json.dumps(entry) + '\n')
    # print(len(final_results))
    # print(final_results[0].keys())
    print("Done!")


