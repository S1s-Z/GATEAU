# GATEAU

The code of our paper "Selecting Influential Samples for Long Context Alignment via Homologous Models’ Guidance and Contextual Awareness Measurement" 

## 🧁 Overview

The expansion of large language models to effectively handle instructions with extremely long contexts has yet to be fully investigated. The primary obstacle lies in constructing a high-quality long instruction-following dataset devised for long context alignment. Existing studies have attempted to scale up the available data volume by synthesizing long instruction-following samples. However, indiscriminately increasing the quantity of data without a well-defined strategy for ensuring data quality may introduce low-quality samples and restrict the final performance. To bridge this gap, we aim to address the unique challenge of long-context alignment, i.e., modeling the long-range dependencies for handling instructions and lengthy input contexts. 
We propose **GATEAU** to identify the influential and high-quality samples enriched with long-range dependency relations by utilizing crafted **Homologous Models' Guidance (HMG)** and **Contextual Awareness Measurement (CAM)**. Specifically, HMG attempts to measure the difficulty of generating corresponding responses due to the long-range dependencies, using the perplexity scores of the response from two homologous models with different context windows. Also, the role of CAM is to measure the difficulty of understanding the long input contexts due to long-range dependencies by evaluating whether the model’s attention is focused on important segments. Built upon both proposed methods, we select the most challenging samples as the influential data to effectively frame the long-range dependencies, thereby achieving better performance of LLMs. Comprehensive experiments indicate that GATEAU effectively identifies samples enriched with long-range dependency relations and the model trained on these selected samples exhibits better instruction-following and long-context understanding capabilities.



## 🎯 Usage

### Homologous Models’ Guidance (HMG)

You can find the corresponding code in `select/hmg`.

#### Setup

Install the requirements with pip: `pip install -r requirements.txt`. For Llama-based models, we recommend using FlashAttention 2 for optimization and saving GPU memory. Once the setup is complete, you can use the shell scripts to perform different calculations.

#### Run

```python
python compute_hmg_4k.py

python compute_hmg_64k.py
```

The results are stored in `select/data/`.



### Contextual Awareness Measurement (CAM)

You can find the corresponding code in `select/cam/`.

#### Setup

Install the requirements with pip: `pip install -r requirements.txt`. For Llama-based models, we recommend using FlashAttention 2 for optimization and saving GPU memory. Once the setup is complete, you can use the shell scripts to perform different calculations.

#### Run

To calculate the segment-level attention score:

```python
sh run_multiprocess_attn.sh

python compute_seg_attn.py
```

The results are stored in `select/data/`.

To calculate the segment-level perplexity score:

```python
sh run_multiprocess_ppl.sh

python compute_seg_ppl.py
```

The results are stored in `select/data/`.

To calculate the  Contextual Awareness Score:

```python
python compute_cas.py
```

The results are stored in `select/data/`.



### Rank

You can find the corresponding code in `select/`.

To calculate the final sore:

```python
python rank.py
```

The results are stored in `train/data/gateau_long.jsonl`.



### Train

You can find the corresponding code in `train`.

Install the requirements with pip: `pip install -r requirements.txt`. 

You can download and save the processed data in two different settings through the [Google Drive](https://drive.google.com/drive/folders/1nqP9S__1E7eJSuEy5lzneJHwL3wN6W2X?usp=drive_link) to train the model. Please put the data into `train/data/`. Meanwhile, you can use other data as a source for general instruction data, but please format your data as follows:

```json
{"messages": 
	[{"role": "user", "content": "..."}, 
	{"role": "assistant", "content": "..."}, ...]
}
```

#### Data preprocessing

Please refer to `train/data_raw.sh`

```python
# e.g., real-world setting for GATEAU-LLaMA-7B-5K

python pre_tokenize.py --input_long_dir ./data/gateau_long.jsonl  --input_share_dir ./data/sharegpt.jsonl --output_dir ./data/llama/7b-5k-100k --model llama --datanum 5k

python sort_and_group.py --group_size 8 --train_file ./data/llama/7b-5k-100k
```

1. First, tokenize the raw text data using the tokenizer of the model. The `--datanum` parameter here refers to the amount of long SFT data you want in your mixed training dataset (our paper investigates on 1k, 3k, and 5k). The tokenized data will be saved under `./data/llama/7b-5k-100k`.

2. We then organize the tokenized data for training.  We use the sorted batching strategy to speed up our training. You should set the `--group_size` parameter to the number of GPUs during training. We recommend using at least 8 80G GPUs for model training, otherwise, the 64k length may incur memory overflow.

#### Training

We provide training scripts under `train/scripts/` (e.g., 1k_100k.sh) for LLaMA-2-7B model series. Make sure to adjust `--model_name_or_path`, `--train_file`, and `--output_dir` to match your model path, data path, and output path. For 1k_100k.sh, 1k means the number of used long SFT data, and 100k refers to the number of used short SFT data.



## 🎲 Evaluation

### LongBench-Chat
The dataset and evaluation code are available under `train/LongBench_Chat/`. Remember to configure your OpenAI API key in `eval.py` since we adopt GPT-4 as the evaluator. Run
```python
python eval.py --model {model_path} --max_length {max_length}
```
`model_path` can either be your local model path or a Hugging Face model path. 

### LongBench

The dataset can be found in the original LongBench paper or this [Google Drive](https://drive.google.com/drive/folders/1xVbwiD477k1PAwOuzBZFfRrAO8diigB3?usp=drive_link). Evaluation code are available under `train/LongBench/`. Remember to configure your OpenAI API key in `eval.py` since we adopt GPT-4 as the evaluator.

```python
python eval.py --model {model_path} --max_length {max_length}
```

`model_path` can either be your local model path or a Hugging Face model path. 

You can use this code to get auto-scores of LongBench instead of GPT-4 evaluation. But you need to make sure you have generated the responses for LongBench.

```python
python eval_auto.py --model {model_path} --max_length {max_length}
```

`model_path` can either be your local model path or a Hugging Face model path. 

### Needle-test
We also provide the code for evaluating HuggingFace models on the "Needle In A Haystack" test under `train/Needle_test/`.See its [README.md](https://github.com/THUDM/LongAlign/blob/main/Needle_test/README.md) for more information.

### MT-bench

To reproduce our results on other benchmarks, we refer to the code in [FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) for evaluate MT-Bench tasks. Remember to set `--dtype` to `bfloat16`  when you attempt to  generate the response.



## 🤖 All available models

Here is the full list of models we released:

|Model|HF Link|Description|
|---|---|---|
|**GATEAU-5k-100k**| [🤗 Link](https://huggingface.co/ssz1111/GATEAU-5k-100k) | Chat model, training on 5k long SFT data from LongAlign and 100k short data from ShareGPT. |
|**GATEAU-3k-100k**| [🤗 Link](https://huggingface.co/ssz1111/GATEAU-3k-100k) | Chat model, training on 3k long SFT data from LongAlign and 100k short data from ShareGPT. |
|**GATEAU-1k-100k**| [🤗 Link](https://huggingface.co/ssz1111/GATEAU-1k-100k) | Chat model, training on 1k long SFT data from LongAlign and 100k short data from ShareGPT. |
|**GATEAU-5k-10k**| [🤗 Link](https://huggingface.co/ssz1111/GATEAU-5k-10k) | Chat model, training on 5k long SFT data from LongAlign and 10k short data from ShareGPT. |
|**GATEAU-3k-10k**| [🤗 Link](https://huggingface.co/ssz1111/GATEAU-3k-10k) | Chat model, training on 3k long SFT data from LongAlign and 10k short data from ShareGPT. |
|**GATEAU-1k-10k**| [🤗 Link](https://huggingface.co/ssz1111/GATEAU-1k-10k) | Chat model, training on 1k long SFT data from LongAlign and 10k short data from ShareGPT. |

A simple demo for deployment of the model:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("ssz1111/GATEAU-1k-10k", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("ssz1111/GATEAU-1k-10k", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
model = model.eval()
query = "\n\n Hello."
response, history = model.chat(tokenizer, query, history=[], max_new_tokens=512, temperature=1)
print(response)
```

