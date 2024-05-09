import os
import sys
sys.path.append('/root/autodl-tmp/') #如果你是在autodl上训练，否则改成你的环境父目
import time
import random 
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import argparse
from transformers.trainer_callback import TrainerControl, TrainerState
from datasets import Dataset, load_dataset
from model.model_config import NDLConfig
config = NDLConfig()
from model.modeling_ndl import NDLForCausalLM
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/tokenize_me',trust_remote_code=True)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tokenizer.pad_token_id = tokenizer.eos_token_id
sky_train = ['/root/autodl-tmp/data/sky_new/2020-50_zh_middle_0008.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_middle_0006.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_middle_0000.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_middle_0002.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_head_0009.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_head_0008.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_head_0007.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_head_0005.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_head_0004.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_head_0003.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_head_0002.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_head_0001.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_head_0000.parquet',
 '/root/autodl-tmp/data/sky_new/2020-50_zh_head_0006.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0013.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0007.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0004.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0002.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0001.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0010.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0011.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0009.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0008.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0006.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0003.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0000.parquet',
 '/root/autodl-tmp/data/sky_new/2020-45_zh_middle_0005.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0001.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0002.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0009.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0004.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0011.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0003.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0005.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0006.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0007.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0010.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0014.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0013.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0016.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_middle_0000.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_middle_0003.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_middle_0002.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_middle_0005.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_middle_0006.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_middle_0009.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_middle_0010.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_middle_0008.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_middle_0011.parquet',
 '/root/autodl-tmp/data/sky_new/2020-40_zh_head_0015.parquet']
code_train = ['/root/autodl-tmp/data/code_new/train-00004-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00005-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00007-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00008-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00019-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00018-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00021-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00010-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00009-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00012-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00011-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00013-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00014-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00015-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00016-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00017-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00020-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00022-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00028-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00029-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00023-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00026-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00027-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00025-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00030-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00024-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00031-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00032-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00033-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00034-of-00880.parquet',
 '/root/autodl-tmp/data/code_new/train-00035-of-00880.parquet']
wiki_en_train = ['/root/autodl-tmp/data/wiki_en_new/train-00007-of-00042-3877aae3fec33422.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00009-of-00042-c9908579eeaf4175.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00013-of-00042-af0983718d50d5b0.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00006-of-00042-9efd9286dc070409.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00014-of-00042-02ca0950368ff729.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00008-of-00042-b851398463f73678.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00010-of-00042-81271737f1362069.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00005-of-00042-180bf29cdace3aa1.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00002-of-00042-a930e3e6733eeedd.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00004-of-00042-bbae804ae18a9fd4.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00003-of-00042-a17e9f8763b78cdf.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00001-of-00042-7aefbfce07009caf.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00000-of-00042-fb075d4d7bcac9ca.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00015-of-00042-4c861e51c53c1271.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00016-of-00042-53643c66a84d0263.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00017-of-00042-b0ce5a338667aec0.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00020-of-00042-44a4db60e9884736.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00022-of-00042-19e84dddc15d3df9.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00018-of-00042-4299b93584494629.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00019-of-00042-e6e41f11f581c2f9.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00021-of-00042-92e38583c5e1ed11.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00024-of-00042-1ba9e84168529381.parquet',
 '/root/autodl-tmp/data/wiki_en_new/train-00023-of-00042-9a33c5735dcf5db7.parquet']
random.seed(42) #设置随机种子
# 根据您的数据集情况自己设置
TRAIN_FILES = sky_train[len(sky_train)//2:]+code_train[len(code_train)//2:]+wiki_en_train[len(wiki_en_train)//2:]
random.shuffle(TRAIN_FILES) #打乱顺序

vocab_size = len(tokenizer)
if vocab_size % 64 != 0:
    vocab_size = (vocab_size // 64 + 1) * 64
print(f"final vocab sieze: {vocab_size}")

# ## token to id缓存到文件，使用的时候不用再次tokenize
# 如果词表大小小于 65535 用uint16存储，节省磁盘空间，否则用uint32存储
map_dtype = np.uint16 if vocab_size < 65535 else np.uint32
def token_to_id(samples: dict) -> dict:

    batch_txt = samples["text"]
    outputs = tokenizer(
        batch_txt,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    input_ids = [np.array(item, dtype=map_dtype) for item in outputs["input_ids"]]

    return {"input_ids": input_ids}


print(token_to_id({'text':['判断给定的文章是否符合语法规则。如果不符合，请提供修改建议。\n','下面是一篇文章的开头: "为了探讨这个主题，本文将提供一系列数据和实例，以证明这一观点。']}))

def get_maped_dataset(files) -> Dataset:
    dataset = load_dataset(
        path="parquet",
        data_files=files,
        split="train",
        cache_dir=".cache",
        keep_in_memory=False,
    )
    print("Loaded dataset size:", len(dataset))
    # 确保 dataset 不为空
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the data files and their content.")
    maped_dataset = dataset.map(
        token_to_id,
        batched=True,
        batch_size=10000,
        remove_columns=dataset.column_names,
        num_proc=45,
        keep_in_memory=False,
    )
    return maped_dataset


train_dataset = get_maped_dataset(TRAIN_FILES)
print(train_dataset)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
config = NDLConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =NDLForCausalLM(config).to(
    device
) 
model_size_shared_experts = sum(t.numel() for name, t in model.named_parameters() if 'shared' in name)
model_size_act_experts = sum(t.numel() for name, t in model.named_parameters() if 'shared' not in name and 'experts' in name)
# print(f"当前MOE模型共享专家层的激活参数大约为：{ model_size_shared_experts / 10000**2/10:.1f}B parameters")
# print(f"当前MOE模型激活专家层的激活参数大约为：{ model_size_act_experts / 10000**2/10/(config.n_routed_experts//config.num_experts_per_tok):.1f}B parameters")
# print(f"当前MOE模型的总参数大小为: {model_size / 10000**2/10:.1f}B parameters")
model_size = sum(t.numel() for t in model.parameters())
print(f"当前模型总参数大小为: {model_size / 10000**2/10:.1f}B ")
# print(f"当前MOE模型的总参数大小为: {model_size / 10000**2/10:.1f}B parameters")
# # 6. cuda cache回调函数

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
    
    args = parser.parse_args()

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
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[my_trainer_callback],
    )

    # trainer.train(resume_from_checkpoint='./model_save/pre/checkpoint-3762') #如果是断点训练
    trainer.train()
    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
    trainer.save_model(args.model_save_dir)

if __name__ == "__main__":
    main()
