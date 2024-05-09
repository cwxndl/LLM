import os
from unicodedata import normalize
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import ujson
from rich import progress
from typing import List
import multiprocessing
from tqdm import tqdm
import sys
sys.path.append('/root/autodl-tmp')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/tokenize_me',trust_remote_code=True)
max_len = 512 ##由于本人算力资源有限，故指定最大的seq_len(每次输入模型的最大的token数量）
vocab_size = len(tokenizer)
map_dtype = np.uint16 if vocab_size < 65535 else np.uint32
'''
[EOS]是分词器的句子结束符号
'''
def split_txt_cropus_to_chunk_data(
    texts: list, batch_size: int = 512**2, max_len: int = 512, window_size: int = 2
) -> list:

    buffer, buffer_len = [], 0
    chunk_data = []

    for i, line in enumerate(texts):
        buffer_len += len(line)
        buffer.append(line)

        if buffer_len >= batch_size or i == len(texts) - 1: #当累计长度达到batch_size时，或者进入最后一个循环时，对累计的数据进行划分，保证最长的seq_len为512
            buffer_txt = "".join(buffer)

            # - window_size为滑动窗口，这样每个窗口都包含有window_size个上文
            for j in range(0, len(buffer_txt), max_len - window_size):

                chunk_data.append("".join(buffer_txt[j : j + max_len]))

            buffer, buffer_len = [], 0

    return chunk_data

def process_none(s: str) -> str: #处理空数据函数
    if s:
        return s
    return ""

def gen_baike(origin_file):
    baike_items = []
    eos_token = "[EOS]" 
    max_len = 512
    batch_size, batch_cnt = 2000000, 0
    with open(origin_file, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break

            item = ujson.loads(line)
            cur_txt, cur_len = [], 0

            if not item["title"]:
                continue

            temp_txt = f"{item['title']}：{process_none(item['summary'])}"

            cur_len += len(temp_txt)
            cur_txt.append(temp_txt)

            for section in item["sections"]:

                # 太长的截断不要了
                if cur_len > max_len:
                    break

                title = f"{section['title']}：" if section["title"] else ""
                temp_txt = f"{title}{process_none(section['content'])}"

                cur_len += len(temp_txt)
                cur_txt.append(temp_txt)
            temp_txt = normalize("NFKC", "".join(cur_txt))

            if len(temp_txt) > max_len:
                # 从 max_len 开始找第一个句号，叹号
                n, i = len(temp_txt), max_len
                while i < n and temp_txt[i] not in ("。", "！"):
                    i += 1
                temp_txt = "".join(temp_txt[0 : i + 1])

                # 添加 eos token
            temp_txt = f"{temp_txt}{eos_token}"

            baike_items.append(temp_txt)

            if len(baike_items) % batch_size == 0:

                chunk_data = split_txt_cropus_to_chunk_data(baike_items)
                tb = pa.Table.from_arrays([chunk_data], names=["text"])

                file_name = f"/root/autodl-tmp/data/baike/baike_chunk_512_5.6M_{batch_cnt}.parquet"
                pq.write_table(
                    table=tb,
                    where=file_name,
                    row_group_size=50000,
                )

                print(f"save to {file_name}")

                batch_cnt += 1
                baike_items = []
        
def gen_sky(input_folder, output_folder): #预处理天工数据集
    os.makedirs(output_folder, exist_ok=True)
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".jsonl"):  # 修改为处理JSON Lines文件
            origin_file = os.path.join(input_folder, filename)
            output_file = os.path.join(
                output_folder, filename.replace(".jsonl", ".parquet")
            )
            lines = []
            with open(origin_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = ujson.loads(line)
                    lines.append(item["text"] + "[EOS]")  # 确保每行都是一个有效的JSON对象
            if lines:  # 确保文件中有内容
                chunk_data = split_txt_cropus_to_chunk_data(lines)
                tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
                pq.write_table(
                    table=tb,
                    where=output_file,
                    row_group_size=50000,
                    data_page_size=50000,
                )
                print(f"处理原文件{origin_file} 保存至：{output_file}")
            else:
                print(f"No content in {origin_file}. Skipping.")

def gen_wiki_en(input_folder, output_folder): #处理维基百科的英文数据集
    os.makedirs(output_folder, exist_ok=True)
    eos_token = "[EOS]"
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".jsonl"):
            doc_ids=[]
            origin_file = os.path.join(input_folder, filename)
            output_file = os.path.join(
                output_folder, filename.replace(".jsonl", ".parquet")
            )
            print(f"当前正在处理文件： {origin_file}...")
            with open(origin_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = ujson.loads(line)
                    text = item["text"]+eos_token
                    doc_ids.append(text)
            if doc_ids:
                chunk_data = split_txt_cropus_to_chunk_data(doc_ids)
                        
            tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
            del chunk_data
            pq.write_table(
                    table=tb,
                    where=output_file,
                    row_group_size=20000,
                    data_page_size=50000,
                )
            print(f"处理原文件{origin_file}保存至{output_file}")
            
def gen_wiki_zh(origin_file, output_file):#处理维基百科的中文数据集
    liness = []
    with open(origin_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    items, content = [], []
    key_word, kw_line_idx = "", 0
    content_start = False  # 词条内容开始标记

    eos_token = "[EOS]"
    for i, line in enumerate(lines):

        line_strip = line.strip()

        # 词条以冒号`：`结尾
        if len(line_strip) > 0 and line_strip[-1] in (":", "："):
            key_word = "".join(line_strip[:-1])
            kw_line_idx = i
            continue

        # 词条key_word在下一行，则合并上个词条并保存
        if i == kw_line_idx + 1 and key_word in line_strip or i == len(lines) - 1:
            txt = "".join(content)

            if len(txt) > 0:
                items.append(f"{txt}{eos_token}")

            content = []
            content.append(f"{key_word}：")

        content.append(line)
    chunk_data = split_txt_cropus_to_chunk_data(items)
    tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
    pq.write_table(
        table=tb,
        where=output_file,
        row_group_size=50000,
        data_page_size=50000,
    )


def gen_code_github(input_folder,output_folder):  ##预处理github 代码数据集
    os.makedirs(output_folder, exist_ok=True)
    eos_token = "[EOS]"
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".parquet"):
            doc_ids=[]
            origin_file = os.path.join(input_folder,filename)
            output_file = os.path.join(output_folder,filename)
            print('当前正在处理文件：{}'.format(origin_file))
            table = pq.read_table(origin_file)
            tmp_df = table.to_pandas()
            code_text = tmp_df['code'].values
            for code in tqdm(code_text):
                text = code+eos_token
                doc_ids.append(text)
            if doc_ids:
                chunk_data = split_txt_cropus_to_chunk_data(doc_ids)
                        
            tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
            del chunk_data
            pq.write_table(
                    table=tb,
                    where=output_file,
                    row_group_size=20000,
                    # data_page_size=50000,
                )
            print(f"处理原文件{origin_file} 保存至{output_file}")
        

if __name__ =='__main__':
    # 按照您的需要处理相应的数据集
    
    # gen_baike("/root/autodl-tmp/baidubaike/563w_baidubaike.json") #处理百度百科数据集
    gen_sky('/root/autodl-tmp/sky_new','/root/autodl-tmp/data/sky_new')
    # gen_wiki_en('/root/autodl-tmp/wiki_en_new','/root/autodl-tmp/data/wiki_en_new')
    # gen_wiki_zh('/root/autodl-tmp/wikipedia-cn-20230720-filtered.json','/root/autodl-tmp/data/wiki_zh/wiki_zh.parquet')
    # gen_code_github('/root/autodl-tmp/code_new','/root/autodl-tmp/data/code_new')
    # print(tokenizer)
    