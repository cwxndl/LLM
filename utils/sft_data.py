'''
此py文件用于处理SFT语料
created by NDL
2024-5-2 17:16
'''

from tqdm import tqdm
import json

def sft_zh():
    train_json_zh = json.load(open('/root/autodl-tmp/sft/alpaca_gpt4_data_zh.json', "r"))
    train_introduct_zh = json.load(open('/root/autodl-tmp/sft/introduction.json','r'))
    train_json =train_json_zh+train_introduct_zh
    new_data = []
    for i in tqdm(range(len(train_json))):
        original_data = train_json[i]
        tmp_data_input = original_data['instruction']+original_data['input']
        tmp_data_output = original_data['output']
        if len(tmp_data_input)+len(tmp_data_output)>512:
            continue
        ID = 'identity_'+str(i)
        # 转换为新格式
        new_data_ = {
                "id": ID,
                "conversations": [
                    {
                        "from": "user",
                        "value": tmp_data_input
                    },
                    {
                        "from": "assistant",
                        "value": tmp_data_output
                    }
                ]
            }
        
        new_data.append(new_data_)
    with open('/root/autodl-tmp/sft/train_3.5M_CN.json','r',encoding='utf-8') as file:
        index = 0
        for line in tqdm(file):
            item = json.loads(line)
            tmp_data = item['conversations']
            tmp_data_input_,tmp_data_output_ = '',''
            if len(item['conversations']) >2:
                # print('当前为多轮对话，删除！')
                continue
            for j, sentence in enumerate(tmp_data):
                role = sentence["from"]
                if role=='human':
                    tmp_data_input_ = sentence["value"]
                if role=='assistant':
                    tmp_data_output_ = sentence["value"]
            if len(tmp_data_input_)+len(tmp_data_output_)>512:
                continue
            ID = 'identity_'+str(len(train_json_zh)+index)
            new_data_ = {
            "id": ID,
            "conversations": [
                {
                    "from": "user",
                    "value": tmp_data_input_
                },
                {
                    "from": "assistant",
                    "value": tmp_data_output_
                }
            ]
        }
            index+=1
            new_data.append(new_data_) 
    
    print('一共收集{}条中文SFT数据'.format(len(new_data)))
    with open('/root/autodl-tmp/data/sft/zh1.json', 'w') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

if __name__=='__main__':
    sft_zh()