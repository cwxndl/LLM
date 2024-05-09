'''
利用transformers库以及tokenizers库训练我们的分词模型（tokenizer）
--------使用算法：BPE算法----------
'''
import os  # 导入os模块，提供与操作系统交互的功能，例如文件路径处理、创建目录等
import tokenizers  # 导入Hugging Face的tokenizers库，用于高级文本分词和预处理
from tokenizers import Tokenizer, decoders  # 导入tokenizer库中的核心类Tokenzier，用于构建分词器；以及decoders模块，包含了解码器，用于将编码后的id序列还原为原始文本
from tokenizers.models import BPE  # 导入Byte Pair Encoding模型，BPE是一种流行的子词单位生成算法，用于构建高效的语言模型
from tokenizers.trainers import BpeTrainer  # 导入BPETrainer，这是用于训练BPE模型的类，基于给定的数据集学习BPE编码规则
from tokenizers.pre_tokenizers import Punctuation, Digits, Metaspace  # 导入预分词处理器，如Punctuation用于单独处理标点符号，Digits用于数字处理，Metaspace用于特殊字符（如空格）的处理
from tokenizers.normalizers import NFKC  # 导入NFKC正则化器，这是一种Unicode标准化形式，用于文本规范化，减少变体和不一致性
from transformers import PreTrainedTokenizerFast  # 导入Hugging Face transformers库中的PreTrainedTokenizerFast类，这是一个预训练好的快速tokenizer接口，常用于与预训练模型配套使用

ROOT_PATH = '/root/autodl-tmp' # 预训练分词模型所在的根目录----可以在这里修改你的根目录

# 定义函数检查文件夹是否存在，如果不存在则直接新建目标文件夹，用户无需自己建立文件夹
def check_dir_exits(dir: str) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)
    
def train_tokenizer(cropus_file: str, max_train_line: int=None, vocab_size: int=40960,token_type: str='char') -> None:
    tokenizer_save_path = ROOT_PATH + '/tokenizer'
    check_dir_exits(tokenizer_save_path)
    
    ## get_training_corpus函数是将训练语料分块传送给分词训练模型进行训练，可以节省内存开支
    def get_training_corpus(buffer_size: int=1000, chunk_len: int=2048) -> list:
        '''
        从文本文件中生成训练语料库的生成器函数。

        Args:
            buffer_size (int): 每次生成的文本块数量。
            chunk_len (int): 每个文本块的最大长度。

        Returns:
            list: 生成的训练语料库。
        '''
        line_cnt = 0  # 记录当前行数
        buffer = []  # 初始化一个空列表用于存储文本块
        with open(cropus_file, 'r', encoding='utf-8') as f_read:  # 打开文本文件进行读取
            cur_chunk_txt, txt_len = [], 0  # 初始化当前文本块和文本长度
            for line in f_read:  # 遍历文件的每一行

                cur_chunk_txt.append(line)  # 将当前行添加到当前文本块中
                txt_len += len(line)  # 更新文本长度
                line_cnt += 1  # 更新行数计数

                if txt_len >= chunk_len:  # 如果当前文本块长度达到设定的最大长度
                    buffer.append(''.join(cur_chunk_txt))  # 将当前文本块添加到buffer中
                    cur_chunk_txt, txt_len = [], 0  # 重置当前文本块和文本长度
                    
                if len(buffer) >= buffer_size:  # 如果buffer中的文本块数量达到设定的缓冲区大小
                    yield buffer  # 生成一个训练语料库(整个buffer列表)
                    buffer = []  # 重置buffer

                if isinstance(max_train_line, int) and line_cnt > max_train_line:  # 如果指定了最大训练行数且行数超过最大训练行数
                    break  # 跳出循环停止生成
                    
            # yield last
            if len(buffer) > 0:  # 如果还有剩余的文本块
                yield buffer  # 生成最后一个训练语料库
       

    special_tokens = ["[PAD]","[EOS]","[BOS]", "[SEP]","[CLS]","[MASK]", "[UNK]"] # 特殊token----预训练所需要
    
    if token_type =='char':

        model = BPE(unk_token="[UNK]")
        tokenizer = Tokenizer(model)
        
        # 用兼容等价分解合并对utf编码进行等价组合，比如全角A转换为半角A
        tokenizer.normalizer = tokenizers.normalizers.Sequence([NFKC()])

        # 标点符号，数字，及Metaspace预分割（否则decode出来没有空格）
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
            [Punctuation(), Digits(individual_digits=True), Metaspace()]
        )

        tokenizer.add_special_tokens(special_tokens)
        tokenizer.decoder = decoders.Metaspace()
    elif token_type == 'byte':

        # byte BPE n不需要unk_token
        model = BPE() 
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)

        tokenizer.add_special_tokens(special_tokens)
        tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.post_processor = tokenizers.processors.ByteLevel(trim_offsets=False)
    else:
        raise Exception(f'token type must be `char` or `byte`, but got {token_type}')

    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=100, show_progress=True, special_tokens=special_tokens)
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # add \t \n 
    if '\t' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\t'])
    if '\n' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\n'])

    tokenizer.save(tokenizer_save_path)

    # 将训练的tokenizer转换为PreTrainedTokenizerFast并保存
    # 转换是为了方便作为`AutoTokenizer`传到其他`huggingface`组件使用。

    # 转换时要手动指定`pad_token`、`eos_token`等特殊token，因为它不指定你原来的tokenizer中哪些字符是这些特殊字符

    slow_tokenizer = tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=slow_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token='[BOS]',
        eos_token='[EOS]',                  
    )

    fast_tokenizer.save_pretrained(tokenizer_save_path)

    print(f'tokenizer save in path: {tokenizer_save_path}')

    print(f"\ntrain tokenizer finished. you can use `AutoTokenizer.from_pretrained('{tokenizer_save_path}')` to load and test your tokenizer.")




    # 模型文件保存在my_tokenizer下


if __name__ == '__main__':

    cropus_file = ROOT_PATH + '/wiki.simple.txt'

    train_tokenizer(cropus_file=cropus_file, token_type='byte',vocab_size=64000) # token_type must be 'char' or 'byte'