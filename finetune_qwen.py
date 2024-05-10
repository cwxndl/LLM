from dataclasses import dataclass, field
import json
from typing import Dict, Optional, List, Tuple
from transformers import Trainer
from torch.utils.data import Dataset
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
import os
import sys
sys.path.append('/root/autodl-tmp/') #如果你是在autodl上训练，否则改成你的环境父目
import time
from dataclasses import dataclass, field
import pandas as pd
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import os
import pickle
def cache_data(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
def load_cached_data(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
import argparse
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
from peft import LoraConfig, get_peft_model,PeftModel,AdaLoraConfig
from tqdm import tqdm
#### 加载分词器和预训练模型
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LoraArguments:
    lora_r: int = 64 
    lora_alpha: int = 16 
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = ['gate_proj','up_proj','down_proj','o_proj']
    
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

class MyTrainerCallback(TrainerCallback):
    log_cnt = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        在打印 n 次日志后清除cuda缓存，适合低显存设备，能防止OOM
        """
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        在on_epoch_end时保存一次模型。
        TrainingArguments的 save_strategy 中 epoch 和 steps 不兼容。要实现每隔 save_steps 步保存一次检查点，考虑到磁盘空间大小，最多只保存最近3个检查点。
        """
        # 设置should_save=True并返回即可
        control.should_save = True
        return control
def collate_fn(batch):
    batch = [item.to(device) for item in batch]
    return torch.utils.data.dataloader.default_collate(batch)


my_trainer_callback = MyTrainerCallback()
def preprocess(
    sources,
    tokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"} #角色扮演说明

    start_id = tokenizer.im_start_id #开始token_id
    end_id = tokenizer.im_end_id #结束token_id
    nl_tokens = tokenizer('\n').input_ids #获取\n的token_id
    _system = tokenizer('system').input_ids + nl_tokens #获取系统提示的id为：system+\n的提示id
    _user = tokenizer('user').input_ids + nl_tokens #获取 用户+\n的提示id
    _assistant = tokenizer('assistant').input_ids + nl_tokens #获取 Chat助手+\n的提示id
    input_ids, targets = [], []
    for i, source in tqdm(enumerate(sources)):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [start_id] + _system + tokenizer(system_message).input_ids + [end_id] + nl_tokens #获取系统提示的id: start_id+system对应id+'\n'id+‘你是一个人工智能’id+end_id+\n id
        input_id += system
        target += [start_id] + [IGNORE_TOKEN_ID] * (len(system)-3) + [end_id] + nl_tokens #获取初始标签Id: start_id+[-100]*(len(system)-3)+end_id+\n id 这里的3指的是 [start_id]+[end_id]+nl_tokens的列表长度
        assert len(input_id) == len(target) #确保input_id与target长度相同
        for j, sentence in enumerate(source):#遍历source中的每一句话
            role = roles[sentence["from"]] #读取角色： [BOS]用户 或 [BOS]Chat助手
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [end_id] + nl_tokens #读取当前数据的value
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [start_id] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [end_id] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [start_id] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [end_id] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id)) #利用pad_token_id填充，当token长度未达到max_len
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))#利用pad_token_id填充，当token长度未达到max_len
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len: int,cache_file):
        super(SupervisedDataset, self).__init__()

        print("****************开始预处理数据****************")
        sources = [example["conversations"] for example in raw_data]
        if os.path.exists(cache_file):
            print("Loading data from cache.")
            data_dict = load_cached_data(cache_file)
        else:
            print('第一次预处理数据')
            data_dict = preprocess(sources, tokenizer, max_len)
            cache_data(data_dict, cache_file)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
def make_supervised_data_module(
    tokenizer, data_path,eval_data_path:None, max_len,cache_file
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset
    print("**************开始加载数据****************")

    train_json = json.load(open(data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len,cache_file=cache_file)

    if eval_data_path:
        pass
        # eval_json = json.load(open(data_args.eval_data_path, "r"))
        # eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def main():
    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--model_save_dir", type=str, default="./model_save")
    parser.add_argument("--train_batch_size", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--sft_data_path", type=str, default='demo1.json')
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument('--model_path_name',type=str,default="Ndlcwx/NDLSLM_0.8B-base")
    parser.add_argument('--use_lora',type=bool,default=False)
    parser.add_argument('--cache_data_path',type=str,default='zh2.pkl')
    args = parser.parse_args()
    lora_args = LoraArguments() #Lora微调的配置
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.im_end_id
    model = AutoModelForCausalLM.from_pretrained(args.model_path_name, trust_remote_code=True).to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"当前模型总参数大小为: {model_size / 10000**2/10:.2f}B ")
    if args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules= lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print('********当前使用Lora进行微调**********')
        model.print_trainable_parameters()
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_path=args.sft_data_path,eval_data_path='' ,max_len=args.max_len,cache_file=args.cache_data_path
    )
    training_args = TrainingArguments(
        output_dir=args.model_save_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=False,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        report_to="tensorboard",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=args.logging_steps,
        log_level="info",
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # data_collator=data_collator,
        **data_module,
        callbacks=[my_trainer_callback],
    )

    # trainer.train(resume_from_checkpoint='./model_save/pre/checkpoint-3762') #如果是断点训练
    trainer.train()
    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
    trainer.save_model(args.model_save_dir)

if __name__ == "__main__":
    main()
    