import torch
import time
from tqdm import tqdm
import json
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM,LlamaPreTrainedModel,LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention,apply_rotary_pos_emb,LLAMA_INPUTS_DOCSTRING,_CONFIG_FOR_DOC
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
) 
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from torch import nn
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from torch.nn import CrossEntropyLoss
from multiprocessing import Process
import gc
import os
import transformers
multiprocessing.set_start_method('spawn', force=True)
import xformers
from xformers.ops.fmha import (
    memory_efficient_attention,
)
from typing import List, Optional, Tuple, Union
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 128))  # Default to 128 if not set
CONTEXT_LENGTH = int(os.environ.get("WINDOW_SIZE", 64000))  # Default to 64000 if not set



IGNORE_INDEX = -100

def set_seed(seed):
    """ fix random seed """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class DataProcessor:
    def __init__(self, model_name, data_file, input_chunk_path, output_chunk_path, score_chunk_path, pic_dir, chunk_size, window_size, single_ppl_batch_size, seed, device,args):
        self.model_name = model_name
        self.data_file = data_file
        self.input_chunk_path = input_chunk_path
        self.output_chunk_path = output_chunk_path
        self.score_chunk_path = score_chunk_path
        self.pic_dir = pic_dir
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.chunk_num = window_size // chunk_size
        self.single_ppl_batch_size = single_ppl_batch_size
        self.seed = seed
        self.device = device
        self.args=args
        if args.do_attn:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                do_attn = args.do_attn,
                attn_implementation="xformer",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            do_attn = args.do_attn
        )
        self.model.to(device)
        # self.model = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if hasattr(self.tokenizer, "add_bos_token"): 
            setattr(self.tokenizer, "add_bos_token", False)
        if hasattr(self.tokenizer, "add_eos_token"):
            setattr(self.tokenizer, "add_eos_token", False)

    def sample_preserve_order(self, array, sample_size):
        indices = list(range(len(array)))
        assert sample_size <= len(indices)
        sampled_indices = sorted(random.sample(indices, sample_size))
        return [array[i] for i in sampled_indices]


    def construct_data(self, data, response,instruct, tokenizer, chunk_size, chunk_num):
        # whole_text = '### Input:' + input_i+ '\n\n ### Instruction: ' + instruct_i + '\n\n ### Response: '+ output_i
        
        tokenized_data = tokenizer(
            '### Input: '+data,
            padding='max_length',  
            truncation=True,      
            max_length=CONTEXT_LENGTH, 
            return_tensors='pt'    
        )
        instruct_data = tokenizer(
            '\n\n ### Instruction: ' + instruct+'\n\n ### Response: ',
            truncation=True,      
            max_length=500, 
            return_tensors='pt'    
        )
        response_data = tokenizer(
            response,
            truncation=True,      
            max_length=500, 
            return_tensors='pt',
            padding='longest'  # This will pad to the actual length of the longest sequence
        )
        assert tokenized_data['input_ids'].shape[-1] == CONTEXT_LENGTH, f"Shape mismatch: {tokenized_data['input_ids'].shape}"
        full_tokenized_data =  {'input_ids' : torch.cat((tokenized_data['input_ids'],instruct_data['input_ids'][:,1:], response_data['input_ids'][:,1:]), dim=1),
        "attention_mask" :torch.cat((tokenized_data['attention_mask'],instruct_data['attention_mask'][:,1:], response_data['attention_mask'][:,1:]), dim=1)}

        data_list = [tokenized_data['input_ids'][0][i:i + chunk_size] for i in range(0, len(tokenized_data['input_ids'][0]), chunk_size)]
        attention_mask_list = [tokenized_data['attention_mask'][0][i:i + chunk_size] for i in range(0, len(tokenized_data['attention_mask'][0]), chunk_size)]
        if len(data_list[-1]) < chunk_size:
            data_list = data_list[:-1]
        if len(data_list) > chunk_num:
            data_list = self.sample_preserve_order(array=data_list, sample_size=chunk_num)
        return data_list,attention_mask_list,tokenized_data,instruct_data,response_data,full_tokenized_data
    



    def compute_single_ppl(self, data_list,attention_mask_list,instruct_data, response_data,batch_size):
        single_ppl = [0 for _ in range(len(data_list))]
        instruct_ids = instruct_data['input_ids'][0][1:]
        instruct_mask = instruct_data['input_ids'][0][1:]
        response_ids = response_data['input_ids'][0][1:]
        response_mask = response_data['input_ids'][0][1:]
        with torch.no_grad():
            self.model.eval()
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i+batch_size]
                batch = [ torch.concat([each,instruct_ids,response_ids]) for each in batch]
                batch_mask = attention_mask_list[i:i+batch_size]
                batch_mask = [ torch.concat([each,instruct_mask,response_mask]) for each in batch_mask]
                
                nums = [len(b) - 1 for b in batch]


                inputs = torch.tensor(np.array(batch)).to(self.device)
                mask = torch.tensor(np.array(batch_mask)).to(self.device)
                labels = inputs.clone()
                labels[:,:-len(response_ids)] = -100
                logits = self.model(input_ids=inputs,attention_mask = mask)[0]

                batch_ppl = self.compute_ppl(logits, labels, nums)
                single_ppl[i:i+batch_size] = batch_ppl
        return single_ppl

    def compute_sentence_attn(self, tokenized_data_full,instruct_data,chunk_size):
        with torch.inference_mode():
            self.model.eval()
            inputs = tokenized_data_full['input_ids'].to(self.device) # not sure
            attention_mask = tokenized_data_full['attention_mask'].to(self.device) # not sure\
            outputs = self.model(input_ids=inputs,attention_mask=attention_mask, output_attentions=True)
            outputs_attention = outputs['attentions']
        all_chunk_attention_avg = []
        instruct_length = len(instruct_data['input_ids'][0][1:])
        for attention in outputs_attention:
            chunk_attention_avg = []
            
            seq_len = attention.shape[-1]
            input_len = min(CONTEXT_LENGTH, seq_len)
            
            # Split sequence into chunks only for the input part
            for i in range(0, input_len, chunk_size):
                chunk_edges = slice(i, min(i + chunk_size, input_len))

                # Mean over the chunked portion
                chunk_attention_part = attention[:, :, chunk_edges].mean(dim=-1)

                chunk_attention_avg.append(chunk_attention_part.cpu().numpy())

            # For the remaining part of the sequence (the response), keep it as-is
           
            # Concatenate chunk averages along the sequence length dimension
            chunk_attention_avg = np.stack(chunk_attention_avg, axis=-1)
            chunk_attention_avg = chunk_attention_avg[:,instruct_length:] # layers*res_len*chunks_num
            chunk_attention_avg = np.mean(chunk_attention_avg,axis=1) # (500+res_len)
            all_chunk_attention_avg.append(chunk_attention_avg)
        del outputs_attention
        gc.collect()
        torch.cuda.empty_cache()
        stacked_array = np.concatenate(all_chunk_attention_avg, axis=0)
        stacked_array_float16 = stacked_array.astype(np.float16)
        return stacked_array_float16




    def compute_ppl(self, logits, labels, nums):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(labels.size(0), -1)  # reshape loss back to sequence length

        batch_ppl = []
        for i, num in enumerate(nums):
            avg_loss = loss[i, -num:].mean()
            batch_ppl.append(torch.exp(avg_loss).float().cpu().item())
        return batch_ppl

    def process_data(self):
        base_name = self.data_file.split('.')[0]
        input_file = os.path.join(self.input_chunk_path, self.data_file)
        output_file = os.path.join(self.output_chunk_path, base_name+'.jsonl')
        pic_dir = os.path.join(self.pic_dir, base_name)
        if not os.path.exists(self.score_chunk_path):
            # create file
            with open(self.score_chunk_path, 'w', encoding='utf-8') as f:
                pass
        chunk_num = self.window_size // self.chunk_size
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        try:
            exist_score_lines = sum(1 for _ in open(self.score_chunk_path, 'r', encoding='utf-8'))
        except Exception as e:
            exist_score_lines = 0
        print(f'[INFO] file {self.score_chunk_path} exist_score_lines: {exist_score_lines}')

        with open(input_file, 'r', encoding='utf-8') as fin:
            total_lines = sum(1 for _ in fin)
            fin.seek(0)
            idx = 0
            for single_data in tqdm(fin, total=total_lines):
                # Loading Data
                if idx < exist_score_lines:
                    idx += 1
                    continue
                if input_file.endswith('.jsonl'):
                    json_data = json.loads(single_data)
                    if 'input' in json_data:
                        data = json_data['input']
                        response = json_data['response']
                        js_idx = json_data['id']
                        instruct = json_data['instruct']
                else:
                    data = single_data
                data_list,attention_mask_list,tokenized_data,instruct_data,response_data,tokenized_data_full = self.construct_data(data, response,instruct,self.tokenizer, self.chunk_size, chunk_num) 
                if self.args.do_attn:
                    outputs_attention =  self.compute_sentence_attn(tokenized_data_full,instruct_data,self.chunk_size)
                    new_data = {"id":js_idx,'text': data, "attention_score":outputs_attention.tolist()}

                else:
                    single_ppl = self.compute_single_ppl(data_list, attention_mask_list, instruct_data, response_data,self.single_ppl_batch_size)
                    print('single_ppl')
                    new_data = {"id":js_idx,'text': data, "single_ppl":single_ppl}

                with open(output_file, 'a+', encoding='utf-8') as fout:
                    fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                    print(f'[INFO] {output_file} write {idx} lines')
                    del new_data
                idx += 1
        os.remove(input_file)

def split_jsonl(input_file, output_path, lines_per_file, total_lines):
    with open(input_file, 'r', encoding='utf-8') as f:
        file_count, line_count = 0, 0
        current_out = open(f"{output_path}/chunk_{file_count}.jsonl", 'w', encoding='utf-8')
        for line in tqdm(f, total=total_lines):
            if line_count < lines_per_file:
                current_out.write(line)
                line_count += 1
            else:
                current_out.close()
                file_count += 1
                current_out = open(f"{output_path}/chunk_{file_count}.jsonl", 'w', encoding='utf-8')
                current_out.write(line)
                line_count = 1
        current_out.close()

def merge(file_list, output_file, output_chunk_path):
    count = 0
    with open(output_file, 'w', encoding='utf-8') as fout:
        for file_name in file_list:
            with open(os.path.join(output_chunk_path, file_name), 'r', encoding='utf-8') as fin:
                for line in fin:
                    print(line, end='', file=fout)
                    count += 1
    return count


def process_single_chunk(gpu_id, file_name, input_chunk_path, output_chunk_path, score_chunk_path, pic_dir, chunk_size, window_size, single_ppl_batch_size, model_name, seed,args):
    pseed = int(file_name.split('.jsonl')[0][6:])
    random.seed(seed + pseed)

    process_id = os.getpid()
    print(f'[PID-{process_id}] {file_name} start!')

    # cuda
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print('Cuda is available.')
        device = torch.device(f'cuda:{gpu_id}')
    else:
        print('Cuda is not available')
        device = torch.device('cpu')
    
    score_chunk_path =os.path.join(score_chunk_path, f"{gpu_id}.txt")

    data_processor = DataProcessor( model_name=model_name, 
                                    data_file=file_name, 
                                    input_chunk_path=input_chunk_path, 
                                    output_chunk_path=output_chunk_path, 
                                    score_chunk_path=score_chunk_path, 
                                    pic_dir=pic_dir, 
                                    chunk_size=chunk_size, 
                                    window_size=window_size, 
                                    single_ppl_batch_size=single_ppl_batch_size, 
                                    seed=seed,
                                    device=device,
                                    args=args)
    data_processor.process_data()
    print(f'[PID-{process_id}] {file_name} end!')
    return process_id

def parse_config():
    parser = argparse.ArgumentParser()

    # data parameter
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--root_path', type=str)
    # lds parameter 
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=32768)

    # model configuration
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m')

    # other 
    parser.add_argument('--seed', type=int, default=11)

    parser.add_argument('--single_ppl_batch_size', type=int, default=256)
    parser.add_argument('--do_attn', action='store_true')

    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0,1,2,3,4,5,6,7])

    return parser.parse_args()

class LlamaXFormersAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)




        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # dtype = query_states.dtype

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # copied from https://github.com/oobabooga/text-generation-webui/pull/950/files
        # This is a nasty hack. We know attention_mask in transformers is either LowerTriangular or all Zeros.
        # We therefore check if one element in the upper triangular portion is zero. If it is, then the mask is all zeros.
        res_attn_weights = None
        if output_attentions:

            import numpy as np
            res_q= query_states.transpose(1,2)[:,:,CONTEXT_LENGTH:]
            res_attn_weights = torch.matmul(res_q,key_states.transpose(1,2).transpose(2,3))/ math.sqrt(self.head_dim)
            if attention_mask is not None:
                c_attn_mask = attention_mask[:, :, CONTEXT_LENGTH:, : key_states.shape[-3]]
                res_attn_weights = res_attn_weights + c_attn_mask
            res_attn_weights= nn.functional.softmax(res_attn_weights, dim=-1,dtype=torch.float32).to(query_states.dtype)
            res_attn_weights = nn.functional.dropout(res_attn_weights, p=self.attention_dropout,training=self.training)
            res_attn_weights = res_attn_weights.transpose(1,2).contiguous()
            res_attn_weights = res_attn_weights.mean(dim=2)
            

        if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = memory_efficient_attention(query_states, key_states, value_states, attn_bias=None)
        else:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = memory_efficient_attention(query_states, key_states, value_states, attn_bias=xformers.ops.LowerTriangularMask())


        # attn_weights = None

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, res_attn_weights, past_key_value



class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config,do_attn=False):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.do_attn = do_attn

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        if not self.do_attn:
            hidden_states = outputs[0]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output =  outputs[1:]
            if not self.do_attn:
                output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if not self.do_attn:

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
                return CausalLMOutputWithPast(
                loss=loss,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


def set_up_attn():
    LLAMA_ATTENTION_CLASSES = {
            "eager": LlamaAttention,
            "flash_attention_2": LlamaFlashAttention2,
            "sdpa": LlamaSdpaAttention,
            "xformer": LlamaXFormersAttention,

    }
    transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES = LLAMA_ATTENTION_CLASSES
    transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

if __name__ == "__main__":
    # args
    args = parse_config()

    if args.do_attn:
        set_up_attn()

    seed = args.seed

    # Setting the seed
    set_seed(seed=seed)

    # Model parameters
    model_name = args.model_name

    # Data paths
    data_file = args.data_file
    root_path = args.root_path

    # Output paths
    input_chunk_path = f'{root_path}/raw'
    output_chunk_path = f'{root_path}/processed'
    save_file = f'{root_path}/merged'
    score_chunk_path = f'{root_path}/scored'
    pic_dir = f'{root_path}/pic'

    # Create directories if not existent
    for directory in [input_chunk_path, output_chunk_path, save_file, score_chunk_path, pic_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Hyperparameters
    chunk_size = args.chunk_size
    window_size = args.window_size
    single_ppl_batch_size = args.single_ppl_batch_size
    gpu_ids = args.gpu_ids
    num_process = len(gpu_ids)

    # Read the total lines in the data file
    with open(data_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)

    if total_lines != -1:
        print(f"The file {data_file} has {total_lines} lines.")
        lines_per_file = total_lines // num_process if total_lines % num_process == 0 else total_lines // num_process + 1

        # Split data into chunks
        split_jsonl(data_file, input_chunk_path, lines_per_file, total_lines)

        assert num_process < multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_process)

        # Get list of files and sort them
        file_list = [file_name for file_name in os.listdir(input_chunk_path) if file_name.endswith('.jsonl')]
        file_list.sort(key=lambda x: int(x.split('.jsonl')[0][6:]))  # Assuming file names are chunkX.jsonl

        print(f'[INFO] {len(file_list)} files found in {input_chunk_path}')
        assert len(file_list) == num_process

        # Process each chunk in parallel
        processes = []
    # Start processing each chunk in separate processes

        for idx, file_name in enumerate(file_list):
            gpu_id = gpu_ids[idx]        
            p = Process(
                target=process_single_chunk,
                args=(
                    gpu_id, file_name, input_chunk_path, output_chunk_path, score_chunk_path, pic_dir, chunk_size, window_size, single_ppl_batch_size, model_name, 
                     seed,args
                )
            )
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        print("All processes have finished.")

