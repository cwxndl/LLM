import os
import time
import sys
sys.path.append('/root/autodl-tmp/') 
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
from model.moe_config import NDLConfig
config = NDLConfig()
from model.modeling_moe import NDLMOEForCausalLM
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./tokenize_me',trust_remote_code=True)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sky_train = [os.path.join('/hy-tmp/data/sky',filename) for  filename in os.listdir('/hy-tmp/data/sky') if filename.endswith('.parquet')]
baike_train = [os.path.join('/hy-tmp/data/baike',filename) for  filename in os.listdir('/hy-tmp/data/baike') if filename.endswith('.parquet')]
code_train = [os.path.join('/hy-tmp/data/code_github',filename) for  filename in os.listdir('/hy-tmp/data/code_github') if filename.endswith('.parquet')]
wiki_en_train =[os.path.join('/hy-tmp/data/wiki_en',filename) for  filename in os.listdir('/hy-tmp/data/wiki_en') if filename.endswith('.parquet')]
wiki_zh_train = [os.path.join('/hy-tmp/data/wiki_zh',filename) for  filename in os.listdir('/hy-tmp/data/wiki_zh') if filename.endswith('.parquet')]

TRAIN_FILES = sky_train+baike_train+code_train+wiki_en_train+wiki_zh_train

@dataclass
class PretrainArguments:
    tokenizer_dir: str = "/root/autodl-tmp/tokenize_me"
    model_save_dir: str = "./model_save/pretrain1/"
    logs_dir: str = "./logs/"
    train_files: list = field(default_factory=lambda: TRAIN_FILES)
    # eval_file: str = EVAL_FILE
    max_seq_len: int = 512
 

pretrain_args = PretrainArguments()
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

# step 3 加载数据集

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
        num_proc=30,
        keep_in_memory=False,
    )
    return maped_dataset


train_dataset = get_maped_dataset(pretrain_args.train_files)
# eval_dataset = get_maped_dataset(pretrain_args.eval_file)

print(train_dataset)
# # 4. 定义data_collator
# `mlm=False`表示要训练CLM模型，`mlm=True`表示要训练MLM模型

# 
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

config = NDLConfig()
model =NDLMOEForCausalLM(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model_size_shared_experts = sum(t.numel() for name, t in model.named_parameters() if 'shared' in name)
model_size_act_experts = sum(t.numel() for name, t in model.named_parameters() if 'shared' not in name and 'experts' in name)
print(f"当前MOE模型共享专家层的激活参数大约为：{ model_size_shared_experts / 10000**2/10:.1f}B parameters")
print(f"当前MOE模型激活专家层的激活参数大约为：{ model_size_act_experts / 10000**2/10/(config.n_routed_experts//config.num_experts_per_tok):.1f}B parameters")
# print(f"当前MOE模型的总参数大小为: {model_size / 10000**2/10:.1f}B parameters")
model_size = sum(t.numel() for t in model.parameters())
print(f"当前模型总参数大小为: {model_size / 10000**2/10:.1f}B ")
# print(f"当前MOE模型的总参数大小为: {model_size / 10000**2/10:.1f}B parameters")
# # 6. cuda cache回调函数


# %%
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

    # trainer.train(resume_from_checkpoint='./model_save/pre/checkpoint-3762')
    trainer.train()
    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
    trainer.save_model(args.model_save_dir)

if __name__ == "__main__":
    main()
