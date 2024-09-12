# process ppl
import jsonlines
import json
import numpy as np
from tqdm import tqdm
if __name__ == '__main__':

    chunk_ppl_file_path = "../data/ppl_final/processed/"
    output_file_path = "../data/seg_ppl.jsonl"

    chunk_ppls = []
    for i in range(0,8):
        with open(chunk_ppl_file_path+f"chunk_{str(i)}.jsonl", 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(tqdm(lines, desc="Processing lines", unit="line")):
                try:
                    chunk_ppls.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Error parsing JSON at line {i + 1}")
                    print(line)
                    break

    with open(output_file_path, 'w') as jsonl_file:
        for entry in chunk_ppls:
            jsonl_file.write(json.dumps(entry) + '\n')
    # print(len(chunk_ppls))
    print("Done!")