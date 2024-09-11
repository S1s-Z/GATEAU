import jsonlines
import json
import numpy as np




# alpha = 0.7 # limited setting
alpha = 0.8 # real-world setting


def softmax(x):

    exps = np.exp(x)

    return exps / np.sum(exps)


def normalize_ppl(ppl_list):

    ppl_normalized = []
    max_ppl = max(ppl_list)
    min_ppl = min(ppl_list)

    for ppl in ppl_list:
        ppl = (ppl - min_ppl) / (max_ppl - min_ppl)
        ppl_normalized.append(ppl)

    return ppl_normalized


if __name__ == '__main__':

    base_json_file_path = "./data/data_ppl_base_7B_ntk.jsonl"
    long_json_file_path = "./data_ppl_long_7B.jsonl"
    ppl_json_file_path = "./norm_ppl_attention_7B.jsonl"

    data = []
    data_base = []
    data_long = []

    base_ppl = []
    with open(base_json_file_path, "r+", encoding="utf8") as f:
        for item in f:
            item = json.loads(item)
            data_base.append(item)
            base_ppl.append(item["ppl"])

    long_ppl = []
    with open(long_json_file_path, "r+", encoding="utf8") as f:
        for item in f:
            item = json.loads(item)
            data_long.append(item)
            long_ppl.append(item["ppl"])

    ppl_attention = []
    with open(ppl_json_file_path, "r+", encoding="utf8") as f:
        for item in f:
            item = json.loads(item)
            ppl_attention.append(item)

    进行归一化
    long_ppl = normalize_ppl(long_ppl)
    base_ppl = normalize_ppl(base_ppl)

    diff_ppl = []

    for i in range(len(data_long)):
        temp_diff = base_ppl[i] - long_ppl[i]
        diff_ppl.append(temp_diff)
    
    norm_diff_ppl = softmax(diff_ppl)

    final_data = []

    for i in range(len(data_long)):
        data_item = {}
        data_item['dataset'] = data_long[i]['dataset']
        data_item['id'] = data_long[i]['id']
        data_item['messages'] = data_long[i]['messages']
        data_item['length'] = data_long[i]['length']

        for j in range(len(ppl_attention)):
            if data_item['id'] == ppl_attention[j]['id']:
                align_score = ppl_attention[j]['ppl_attention_align_score']
                break
            else:
                pass

        # fixed
        ppl_score = norm_diff_ppl[i]

        # set your own alpha 
        final_score = alpha * ppl_score + (1-alpha) * align_score


        data_item['score'] = final_score
        final_data.append(data_item)


    sorted_data = sorted(final_data, key=lambda x: x["score"], reverse=True)
    output_file_path = "../train/data/gateau_long.jsonl"

    with open(output_file_path, 'w') as jsonl_file:
        for entry in sorted_data:
            jsonl_file.write(json.dumps(entry) + '\n')
    print("Done!")