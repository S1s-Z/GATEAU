import os
import json
import torch
import argparse
from tqdm import tqdm
import jsonlines
import torch.nn as nn
import time
from scale_rope.consistent_rope_for_llama_patch import replace_llama_attn_with_consistent_ntk_rope
replace_llama_attn_with_consistent_ntk_rope()


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='../data/ori_LongAlign.json')
    parser.add_argument("--save_path", type=str, default='../data/HMG_4k.jsonl')
    parser.add_argument("--model_name_or_path", type=str, default='NousResearch/Llama-2-7b-hf')
    parser.add_argument("--max_length", type=int, default=64000)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()
    return args


def get_perplexity_for_response(tokenizer, model, text, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    response_text = text.split("### Response: ")[1]
    start_index = text.rfind(response_text)
    start_token = len(tokenizer.encode(text[:start_index]))
    labels = input_ids.clone()
    labels[0, :start_token] = -100

    model.eval()
    with torch.no_grad(): 
        outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    perplexity = torch.exp(loss)

    return perplexity.to('cpu').item()

def main():

    args = parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True, 
                                                use_cache=False,
                                                attn_implementation="flash_attention_2",
                                                )
    
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.truncation_side = "left"

    model.eval()

    if os.path.exists(args.save_path):
        print('save_path exists!')
        raise Exception

    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]

    strat_time = time.time()
    new_data = []

    for i in tqdm(range(len(sampled_data))):

        data_i = sampled_data[i]

        instruct_i = ''.join(data_i['messages'][0]['content'].split('\n\n')[-1])
        input_i = data_i['messages'][0]['content'][:-len(instruct_i)]
        output_i = data_i['messages'][1]['content']
        

        whole_text = '### Input:' + input_i+ '\n\n ### Instruction: ' + instruct_i + '\n\n ### Response: '+ output_i

        temp_data_i = {}
        temp_data_i['dataset'] = data_i['length']
        temp_data_i['id'] = data_i['id']
        temp_data_i['messages'] = data_i['messages']
        temp_data_i['length'] = data_i['length']

        ppl_ins_alone = get_perplexity_for_response(tokenizer, model, whole_text, args.max_length)
        temp_data_i['ppl'] = ppl_ins_alone
        print('ppl_ins_alone:',ppl_ins_alone)
        new_data.append(temp_data_i)

    with open(args.save_path, 'w') as jsonl_file:
        for entry in new_data:
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print('Time Used:',(time.time()-strat_time)/60,'(min)')

if __name__ == "__main__":
    main()