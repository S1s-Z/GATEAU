import requests
import time, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import torch
import numpy as np
import random
import codecs
import argparse
from copy import deepcopy
from tqdm import tqdm
import traceback
import re
import jsonlines
import openai
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from datasets import load_dataset, load_from_disk
import torch.distributed as dist
import torch.multiprocessing as mp

replace_llama_attn_with_flash_attn()


import os

os.makedirs("predictions", exist_ok=True)
os.makedirs("scores", exist_ok=True)

dataset2prompt = json.load(open("./config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("./config/dataset2maxlen.json", "r"))
system_prompt = "You're a good assistant at evaluating the quality of texts."
GPT_MODEL = 'gpt-4'

api_key = '' # Enter your openai api key here
    
def query_gpt4(prompt):
    msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    tries = 0    
    while tries < 5:
        tries += 1
        try:
            headers = {
                'Authorization': f"Bearer {api_key}"
            }
            # check whether you need to use proxy, this is my used proxy.
            resp = requests.post("https://mtu.mtuopenai.xyz/v1/chat/completions", json = {
                "model": GPT_MODEL,
                "messages": msg,
                "temperature": 0.
            }, headers=headers, timeout=120)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    
    return resp

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def chat(model, path, tokenizer, prompt, device, history=[], max_new_tokens=1024, temperature=1.0):
    valid_path = path.lower()
    if "internlm" in valid_path or "chatglm" in valid_path or "longalign-6b" in valid_path:
        response, history = model.chat(tokenizer, prompt, history=history, max_new_tokens=max_new_tokens, temperature=temperature)
        return response, history
    elif "longalign-7b" in valid_path or "longalign-13b" in valid_path:
        if history == []:
            prompt = f"[INST]{prompt}[/INST]"
        else:
            prompt = history+"\n\n"+f"[INST]{prompt}[/INST]"
    elif "mistral" in valid_path or "mixtral" in valid_path:
        if history == []:
            prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            prompt = history+f"</s> [INST] {prompt} [/INST]"
    elif "longchat" in valid_path or "vicuna" in valid_path:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    context_length = input.input_ids.shape[-1]
    output = model.generate(
        **input,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        temperature=temperature,
    )[0]
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return pred.strip(), prompt + pred.strip()

def load_model_and_tokenizer(path, device):
    valid_path = path.lower()
    if "longchat" in valid_path or "vicuna" in valid_path:
        from fastchat.model import load_model
        model, _ = load_model(path, device='cpu', num_gpus=0, load_8bit=False, cpu_offloading=False, debug=False)
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    elif "mistral" in valid_path or "mixtral" in valid_path:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, use_flash_attention_2=True, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        model.generation_config = GenerationConfig.from_pretrained(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    return model, tokenizer

def get_predictions(rank, world_size, path, max_length, data, out_path, dataset, task):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(path, device=device)
    for json_obj in data:
        seed_everything(42)
        history = []
        prompt_format = dataset2prompt[dataset]
        prompt = prompt_format.format(**json_obj)
        # query = json_obj['query']
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        response, history = chat(model, path, tokenizer, prompt, device=device, history=history, max_new_tokens=dataset2maxlen[dataset], temperature=1.0)
        # line = response.strip().replace('\n', ' ') + '\n'
        line = response.strip().replace('\n', ' ')
        # print("prompt",prompt)
        # print("response",response)

        out_dir = '/'.join(out_path.split("/")[:-1])
        
        if os.path.exists(out_dir):
            pass
        else:
            os.makedirs(out_dir, exist_ok=True)

        with open(out_path, "a", encoding="utf-8") as f:
            if 'qa' in task:
                json.dump({"pred": line, "query":json_obj["input"] ,"answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')
                f.flush()
            else:
                json.dump({"pred": line, "query":' ' ,"answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')
                f.flush()
    dist.destroy_process_group()
    
def get_score(path, task):

    predictions = []

    with open(path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            predictions.append(item)

    # with open(f"predictions/{save_name}.txt", "r", encoding='utf-8') as f:
    #     for line in f:
    #         predictions.append(line.strip())
    # assert len(predictions) == len(test_cases)
    result = []
    lines = []
    scores = []
    total_tokens = 0
    for case in tqdm(predictions):
        question, answer = case["query"], case["answers"]
        pred_response = case["pred"]

        if 'qa' in task:
            template = open("./prompt_for_qa.txt", encoding="utf-8").read()
            prompt = template.format(question, answer, pred_response)
        elif task =='sum':
            template = open("./prompt_for_sum.txt", encoding="utf-8").read()
            prompt = template.format(answer, pred_response)
        else:
            break
        
        # print(prompt)
        score = "none"
        trys = 0
        while (score == "none") and (trys < 5):
            response = query_gpt4(prompt)
            try:
                num_tokens = response["usage"]["total_tokens"]
                response = response["choices"][0]["message"]["content"]
                score = re.findall(r"\[\[([^\]]+)\]\]", response)[-1]
                matches = re.findall(r"\d+\.\d+|\d+", score)
                score = matches[0]
            except:
                trys += 1
                num_tokens = 0
                score = "none"
        total_tokens += num_tokens
        scores.append(score)
        lines.append(pred_response + '\t' + score + '\n')
        case.update({
            "prediction": pred_response,
            "gpt_analysis": response,
            "score": score,
            "used_tokens": num_tokens
        })
        # case.pop("prompt")
        result.append(case)
    try:
        scores = [float(score) for score in scores]
        total_score = sum(scores)
    except Exception as e:
        traceback.print_exc()
        total_score = "none"
    
    if 'qa' in task:
        result.append({
            "total_score": total_score,
            "total_tokens": total_tokens,
            "total_len": len(scores),
            "final_socore": total_score / (len(scores)*3),
        })
        print("total_score:", total_score)
        print("final_score:", total_score / (len(scores)*3))
    else:
        result.append({
            "total_score": total_score,
            "total_tokens": total_tokens,
            "total_len": len(scores),
            "final_socore": total_score / (len(scores)*5),
        })
        print("total_score:", total_score)
        print("final_score:", total_score / (len(scores)*5))


    dataset_name = out_path.split("/")[-1].replace(".jsonl", "").strip()
    
    out_dir = '/'.join(out_path.split("/")[:-1]).replace('./predictions', './scores')
    if os.path.exists(out_dir):
        pass
    else:
        os.makedirs(out_dir, exist_ok=True)

    with codecs.open(f"{out_dir}/{dataset_name}_result.json", 'w', encoding='utf-8') as fout:
        json.dump(result, fout, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=False)
    parser.add_argument("--max_length", default=64000, type=int)
    args = parser.parse_args()
    world_size = 8
    mp.set_start_method('spawn', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
    #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
    #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    single_doc_qa = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"]
    multi_doc_qa = ["hotpotqa", "2wikimqa", "musique", "dureader"]
    summarization = ["gov_report", "qmsum", "multi_news", "vcsum"]

    for dataset in single_doc_qa:
        template_for_eval = open("prompt_for_qa.txt", encoding="utf-8").read()
        # data = load_dataset('THUDM/LongBench', dataset, split='test')
        data = []
        with open(f'./data/{dataset}.jsonl', 'r') as file:
            lines = file.readlines()
            for line in lines:
                data.append(json.loads(line))
        model_name = args.model_path.split("llama")[-1].replace("/", "-")
        out_path = f"./predictions/llama{model_name}/{dataset}.jsonl"
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_predictions, args=(rank, world_size, args.model_path, \
                                                         args.max_length, data_subsets[rank], out_path, dataset, "single_doc_qa"))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # get_predictions(args.model_path, args.max_length, data_all, out_path, dataset, task="single_doc_qa")

        get_score(out_path, task="single_doc_qa")

    for dataset in multi_doc_qa:
        template_for_eval = open("prompt_for_qa.txt", encoding="utf-8").read()
        # data = load_dataset('THUDM/LongBench', dataset, split='test')
        data = []
        with open(f'./data/{dataset}.jsonl', 'r') as file:
            lines = file.readlines()
            for line in lines:
                data.append(json.loads(line))
        model_name = args.model_path.split("llama")[-1].replace("/", "-")
        out_path = f"./predictions/llama{model_name}/{dataset}.jsonl"
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_predictions, args=(rank, world_size, args.model_path, \
                                                         args.max_length, data_subsets[rank], out_path, dataset, "multi_doc_qa"))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # get_predictions(args.model_path, args.max_length, data_all, out_path, dataset, task="multi_doc_qa")

        get_score(out_path, task="multi_doc_qa")

    for dataset in summarization:
        template_for_eval = open("prompt_for_sum.txt", encoding="utf-8").read()
        # data = load_dataset('THUDM/LongBench', dataset, split='test')
        data = []
        with open(f'./data/{dataset}.jsonl', 'r') as file:
            lines = file.readlines()
            for line in lines:
                data.append(json.loads(line))
        model_name = args.model_path.split("llama")[-1].replace("/", "-")
        out_path = f"./predictions/llama{model_name}/{dataset}.jsonl"
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        # multi-gpu
        for rank in range(world_size):
            p = mp.Process(target=get_predictions, args=(rank, world_size, args.model_path, \
                                                         args.max_length, data_subsets[rank], out_path, dataset, "sum"))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # single-gpu
        # get_predictions(args.model_path, args.max_length, data_all, out_path, dataset, task="sum")

        get_score(out_path, task="sum")