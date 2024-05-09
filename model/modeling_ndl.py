'''
Decoder only model for trainging LLM 

created by NDL 

Date: 2024/4/20
'''
import sys
sys.path.append('/root/autodl-tmp/')
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from typing import Any, Optional, Tuple,Callable
import torch
import torch.nn.functional as F
from torch.nn import  CrossEntropyLoss
from torch import nn
from model.model_config import NDLConfig
from transformers.utils import logging,is_flash_attn_greater_or_equal_2_10,is_flash_attn_2_available
from transformers.generation.utils import GenerateOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers import GenerationConfig, PreTrainedTokenizer, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
logger = logging.get_logger(__name__)
from transformers import PreTrainedModel
from model.import_flash_attn import _import_flash_attn #导入flash_attn模块加速训练
ndlconfig = NDLConfig()

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  

def _get_unpad_data(attention_mask):
    # 计算每个序列的有效长度（非填充部分）
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 获取所有非填充元素的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 找出批中最长的序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算序列的累积长度，并在前面填充一个0，以便于后续索引操作
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # 返回非填充索引、累积长度和最大序列长度
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

class NDLRMSNorm(nn.Module): ##不使用flash-attn的RMSNorm
    def __init__(self, dim:int, eps:float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  #self.weight代表了模型学习到的权重参数
        self.epsilon = eps # eps通常为很小的正数，防止出现分母为0的情况（计算溢出）
    def _rms_norm(self, input):
        input_dtype = input.dtype # 获取输入数据的原始数据类型（如torch.float64, torch.float32等）
        input = input.to(torch.float32)# 将输入数据转换成32位单精度浮点数，以便于计算（假设模型在此精度下运行更高效）
        # 计算输入数据的每个通道（最后一维）的均方值，然后增加一个维度（keepdim=True），保持输出与原输入形状相同
        variance = input.pow(2).mean(-1, keepdim=True)
        # 使用贝塞尔修正（防止除以零错误）计算归一化后的隐藏状态，rsqrt()函数计算平方根倒数
        # 即对variance加上一个很小的数（self.epsilon）后求其平方根倒数，再乘以原始输入以实现批量归一化操作
        hidden_states = input * torch.rsqrt(variance + self.epsilon)
        return self.weight * hidden_states.to(input_dtype) # 将归一化后的隐藏状态按原始数据类型转换回去，然后与权重矩阵相乘，完成最终的特征映射
    def forward(self,x):
        return self._rms_norm(x)

class NDLFlash_attnRMSNorm(nn.Module): #使用flash-attn的RMSNorm归一化方法
    print('正在使用flash_attn的RMS')
    def __init__(self, dim:int, eps:float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  #self.weight代表了模型学习到的权重参数
        self.epsilon = eps # eps通常为很小的正数，防止出现分母为0的情况（计算溢出）
        # self.flash_rms_norm = None
        from flash_attn.ops.rms_norm import rms_norm as __rms_norm
        self.flash_rms_norm = __rms_norm
    def forward(self,x):
        return self.flash_rms_norm(x,self.weight,self.epsilon)

         
####参考Llama2源码修改的ROPE代码#########
class NDLRotryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings: int, base=10000.0, device='cuda',scaling_factor=1.0):
        super().__init__()# 调用父类的初始化方法
        self.dim = dim# 设置维度大小，表示位置编码的向量维度
        self.max_position_embeddings = max_position_embeddings# 设置最大位置嵌入数量，即能处理的最大序列长度
        self.base = base
        self._ntk_alpha_cached = 1.0
        self.device=device
        self.scaling_factor = scaling_factor
        # 根据给定的基数值计算频率逆周期向量inv_freq，遵循公式theta_i = 1/(base^(2i/dim))
        # 其中i是维度索引，范围从0到dim-1，每隔2取一个值
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        # 这里得到的self.inv_freq是一维张量
        self.register_buffer("inv_freq", inv_freq, persistent=False)# 注册存储inv_freq的缓冲区，persistent=False表示它不会被pickle保存
    @torch.no_grad()
    def forward(self ,x, position_ids):# x:[batch_size,num_attention_heads, seq_len, head_size]
        # position_ids :[batch_size, sequence_length]
        # None 表示在该维度添加一个新的轴（unsqueeze 操作），第一步将self.inv_freq转换为1*inv_freq_length*1的形状
        # expand方法将其转换为 position_ids.shape[0](bsz),第二个维度自动计算，最后一个维度为1的张量
        # 但expand方法是对某个维度的重复，适用于广播机制的情况，区别于reshape方法
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) 
        position_ids_expanded = position_ids[:, None, :].float()# 将position_ids扩展为bsz,1,sequence_length的形状
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        #torch.autocast 是 PyTorch 中的一个上下文管理器，它用于自动混合精度训练（Automatic Mixed Precision, AMP）
        #这段代码的作用是创建一个上下文，在该上下文中自动混合精度被禁用（enabled=False）
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # inv_freq与position_ids相乘，并转置，交换第1维与第2维
            emb = torch.cat((freqs, freqs), dim=-1) #将freqs与freqs按照最后一个维度进行拼接
            cos = emb.cos() #计算余弦值
            sin = emb.sin() #计算正弦值
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# 定义名为rotate_half的函数，功能是对输入tensor的最后一维（隐藏维度）的一半进行旋转操作
def rotate_half(x):
    # 切片操作，获取输入tensor的最后一维（隐藏维度）的前半部分
    x1 = x[..., : x.shape[-1] // 2] # ...代表当前开始的所有维度，省去中间表示
    # 切片操作，获取输入tensor的最后一维（隐藏维度）的后半部分
    x2 = x[..., x.shape[-1] // 2:]
    # 使用torch.cat函数，沿最后一维（-1）将后半部分x2取负后与前半部分x1拼接
    return torch.cat((-x2, x1), dim=-1)# 实现了将最后一维的一半内容旋转的效果

class NDLNTKRotaryEmbedding(NDLRotryEmbedding):
   # 当输入序列长度大于设置的最大序列长度时，使用NTK进行外推
    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False) 
        cos, sin = super().forward(x, position_ids)
        return cos, sin

def apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    """
 输入参数:
    q (`torch.Tensor`): q向量.
    k (`torch.Tensor`): k向量.
    cos (`torch.Tensor`): 旋转编码的余弦值.
    sin (`torch.Tensor`): 旋转编码的正弦值.
    unsqueeze_dim (`int`, *optional*, defaults to 1):
    为了便于后面进行矩阵乘法:
    (1):假设q,k张量的形状为：[batch_size,heads,seq_len,head_dim],而cos[position_ids]的形状为[batch_size,seq_len,head_dim]
    为了使得矩阵能够相乘，需要在cos与sin的第一个维度添加新的维度变为[batch_size,1,seq_len,head_dim]
    (2):假设q,k张量的形状为：[batch_size, seq_len, heads, head_dim],而cos[position_ids]的形状为[batch_size,seq_len,head_dim]
    为了使得矩阵能够相乘，需要在cos与sin的第二个维度添加新的维度变为[batch_size,seq_len,1,head_dim]
    返回:
        q,k张量的旋转位置编码
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class NDLFFN(nn.Module): # transformer block继承attention block之后的前馈层
    def __init__(self, config:ndlconfig):
        super().__init__()
        self.config = config #模型的配置文件
        self.hidden_size = config.hidden_size #隐藏层数量，attention层的输出维度---attention层经过残差相加，即输入的hidden_size维度
        self.intermediate_size = config.intermediate_size#模型中MLP的中间层的维度，用来捕获更多的信息以及提高的泛化能力
        self.ffn1_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) # 第一个Linear层，不加偏置项
        self.ffn2_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) # 第二个Linear层，不加偏置项
        self.o_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)# 第三个Linear层，不加偏置项
    
    def forward(self,attention_out):
        o1 = self.ffn1_proj(attention_out) #计算第一个Linear层的输出
        o2 = self.ffn2_proj(attention_out) #计算第二个Linear层的输出
        intermediate_out = o1*F.silu(o2) #将o1与激活函数激活后的o2逐个元素相乘
        output = self.o_proj(intermediate_out) #将逐个元素相乘的结果作为输入进行第三个Linear输出当前Transformer Block的输出
        return output

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 该函数的作用是将hidden_states张量复制
    batch, num_key_value_heads, seqlen, head_dim = hidden_states.shape #获得hidden_states张量的shape
    if n_rep == 1:
        return hidden_states  #如果n_rep=1,则无需进行复制
    else:
        # 如果n_rep>1,则需要将hidden_states的第二维度添加一个维度，复制为batch_size,num_key_value_heads,n_rep,seqlen,head_dim的形状
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seqlen, head_dim) #最后将张量reshape为batch, num_key_value_heads * n_rep, seqlen, head_dim

    
class NDLNormAttention(nn.Module): # Transformer Block的自注意力机制-----GQA分组查询注意力机制---如果设置了num_key_value_heads大于1会使用GQA机制，否则就是全局注意力机制
    def __init__(self, config:ndlconfig,layer_idx):
        super().__init__()
        self.config = config #模型的配置文件
        self.layer_idx = layer_idx #Decoder层的index(索引)
        self.attention_dropout = config.attention_dropout # attention 机制的dropout率
        self.hidden_size = config.hidden_size #hidden_size大小
        self.num_heads = config.num_attention_heads #多头注意力机制的头数
        self.head_dim = self.hidden_size // self.num_heads #平均每个注意力头处理的特征维度
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads is not None else 1 #self.num_key_value_heads 表示每个注意力头中键和值的子头数量，也就是说，每个注意力头可以进一步细分为多个处理键和值的小注意力头。
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # 它表示在所有注意力头中共有多少组键值子头
        self.max_position_embeddings = config.max_position_embeddings 
        self.rope_theta = config.rope_theta
        self.is_causal = True
        #self.training = True 无需事先指定self.training参数，
        # 在PyTorch中，self.training属性是在nn.Module类的forward方法中自动设置的。当调用model.train()时，PyTorch会将模型设置为训练模式，同时设置self.training为True。同样，当调用model.eval()时，PyTorch会将模型设置为评估模式，同时将self.training设置为False。
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size 一定要被 num_heads 整除才可以 ( `hidden_size`: {self.hidden_size}"
                f" `num_heads`: {self.num_heads})."
            )
        self.q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)# 输入维度hidden_size,输出维度num_head(头数)*head_dim(每个头的维度)，相当于多头输出的cat
        self.k = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.get_rope()
    
    def get_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = NDLRotryEmbedding(dim=self.head_dim,max_position_embeddings=self.max_position_embeddings,base=self.rope_theta)
        elif self.config.rope_scaling =='NTK':
            # print('当前正在使用NTK进行缩放')
            self.rotary_emb = NDLNTKRotaryEmbedding(dim=self.head_dim,max_position_embeddings=self.max_position_embeddings,base=self.rope_theta)
        else:
            raise ValueError(f"你只能输入None与NTK中的某一个缩放方式")
    
    def forward(
        self,
        hidden_states: torch.Tensor, #输入attention_block的张量
        attention_mask:  None, # mask矩阵
        position_ids: None, # 位置Id
        past_key_value: None, #存储过去时间步的key,value
        output_attentions: bool = False,
        use_cache: bool = False,
        # cache_position: Optional[torch.LongTensor] = None, #为了维护静态缓存所需的位置信息
        **kwargs):
        batch_size, seq_len, _ = hidden_states.size() 
        
        if self.config.pretraining_tp > 1: #分布式训练，将权重划分为多个块，提高预训练效率
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k.weight.split(key_value_slicing, dim=0)
            value_slices = self.v.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else: # 如果self.pretrain_tp设置为1或小于1的时候，直接计算q,k,v
            query_states = self.q(hidden_states)
            key_states = self.k(hidden_states)
            value_states = self.v(hidden_states)
        
        # 原始query_states为(batch_size, seq_len, self.num_heads*self.head_dim)，首先转化为(batch_size, seq_len, self.num_heads, self.head_dim)
        # 对重塑后的张量进行转置操作，交换了张量的第1维（seq_len）和第2维（self.num_heads）。这使得张量的维度变为 (batch_size, self.num_heads, seq_len, self.head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        #原始key_states为(batch_size, seq_len,self.num_key_value_heads * self.head_dim),转换为（batch_size,seq_len,self.num_key_value_heads, self.head_dim)
        #对重塑的张量进行转置操作，交换了张量的第一维度和第二维度，变为（batch_size,self.num_key_value_heads,seq_len, self.head_dim)
        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids) #旋转位置编码获取cos,sin
        query_states, key_states = apply_rope(query_states, key_states, cos, sin) #将k,q进行ROPE获得相对位置信息
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            # 调用 past_key_value.update() 方法来结合新的 key_states 和 value_states 更新缓存
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups) #
        value_states = repeat_kv(value_states, self.num_key_value_groups) # 重复复制 self.num_key_value_groups
        
            
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) #softmax的输入 [batch_size,self.num_heads,seq_len,seq_len]
        # query_states:[batch_size, self.num_heads, seq_len, self.head_dim] key_states.transpose(2,3):[batch_size, self.num_heads, self.head_dim, seq_len]
        # 下面将进行注意力掩码，确保模型仅关注序列中的历史信息
        #注意力掩码通常是一个二进制张量，用于指示哪些位置的输入应当参与注意力计算，哪些位置应该被忽略（通常是未来位置）      
        if attention_mask is not None: 
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) #计算softmax(Q*KT)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)  #计算自注意力机制的输出 softmax(Q*KT)*V/\sqrt(dim)
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output`的形状应该为：{(batch_size, self.num_heads, seq_len, self.head_dim)}, 但是当前的形状为："
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous() #转置并连续化 [batch_size,seq_len,self.num_heads,self.head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size) # reshape为【batch_size, seq_len, self.hidden_size】
        
        if self.config.pretraining_tp > 1: #分片操作，提高预训练效率
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o(attn_output) #输出形状为 [batch_size,seq_len,hidden_size]
            

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
class NDLSdpaAttention(NDLNormAttention):
    # 此类是使用torch的scale-dot-product-attention方法进行flash加速
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask:  None,
        position_ids:  None,
        past_key_value: None,
        output_attentions: bool = False,
        use_cache: bool = False,
        # cache_position: Optional[torch.LongTensor] = None,
    ) :
        if output_attentions:
            logger.warning_once(
                "NDLModel 正在使用 NDLSdpaAttention, 但是 `torch.nn.functional.scaled_dot_product_attention` 不支持`output_attentions=True`. 需要重新手动计算attention_out  "
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                # cache_position=cache_position,
            )
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(value_states, position_ids) #旋转位置编码获取cos,sin
        query_states, key_states = apply_rope(query_states, key_states, cos, sin) #将k,q进行ROPE获得相对位置信息
 
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        attn_output = self.o(attn_output)

        return attn_output, None, past_key_value

NDL_ATTENTION_CLASSES = {
    "eager": NDLNormAttention, #不使用torch中的flash
    # "flash": NDLFlashAttention2, #使用flash_attn
    "sdpa": NDLSdpaAttention, #使用torch中的flash
}
class NDLDecoderlayer(nn.Module):
    def __init__(self, config:ndlconfig, layer_idx: int,select_rmsnorm:'NDLRMSNorm'):
        super().__init__()
        self.hidden_size = config.hidden_size
        config.attn_implementation = 'sdpa'
        # print(config)
        self.self_attn = NDL_ATTENTION_CLASSES[config.attn_implementation](config=config, layer_idx=layer_idx)
        
        # print(NDL_ATTENTION_CLASSES[config.attn_implementation](config=config, layer_idx=layer_idx))
        self.mlp = NDLFFN(config) #前馈神经网络层
        self.select_rmsnorm = select_rmsnorm #选择RMSNorm层是否使用flash_attn
        if 'Flash' not in self.select_rmsnorm:
            self.input_layernorm = NDLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = NDLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else: 
            self.input_layernorm = NDLFlash_attnRMSNorm(config.hidden_size, eps=config.rms_norm_eps) #输入层的RMSNorm
            self.post_attention_layernorm = NDLFlash_attnRMSNorm(config.hidden_size, eps=config.rms_norm_eps) #attention之后的RMSNorm
    def forward(
        self,
        hidden_states: torch.Tensor, #输入：(batch, seq_len, embed_dim)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        # cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states #第一个残差块的原始x
        hidden_states = self.input_layernorm(hidden_states) # 对输入张量进行RMSNorm
        # 计算自注意力机制的输出
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            # cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states #残差输出
        #进入全连接层
        residual = hidden_states #FFN的残差输入X
        hidden_states = self.post_attention_layernorm(hidden_states) #进行RMSNorm归一化
        hidden_states = self.mlp(hidden_states) #进入FFN块输出结果
        hidden_states = residual + hidden_states #残差连接
        outputs = (hidden_states,)
        if output_attentions: #如果要输出out_attention
            outputs += (self_attn_weights,)
        
        if use_cache: #如果使用缓存机制
            outputs += (present_key_value,)
        return outputs

class NDLPreTrainedModel(PreTrainedModel):
    
    config_class = ndlconfig
    base_model_prefix = "NDLmodel"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NDLDecoderlayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    # 模型参数初始化
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear): #如果当前module是线性层
            module.weight.data.normal_(mean=0.0, std=std) #初始化权重为均值为0，方差为std=0.02的数据
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding): #如果当前module是embedding层
            module.weight.data.normal_(mean=0.0, std=std)#初始化权重为均值为0，方差为std=0.02的数据
            if module.padding_idx is not None: # 并且如果存在填充索引（padding_idx），则填充位置的权重设为0。
                module.weight.data[module.padding_idx].zero_()
        for name, p in module.named_parameters():
            if name == "o_proj.weight": #初始化FFN层的权重参数为均值为0，方差为：0.02/math.sqrt(2*num_hidden_layers)
                p.data.normal_(
                    mean=0.0,
                    std=(
                        self.config.initializer_range
                        / math.sqrt(2 * self.config.num_hidden_layers)
                    ),
                )
    
        
class NDLModel(NDLPreTrainedModel):
    def __init__(self, config: ndlconfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id # 分词器中的pad_token_id
        self.vocab_size = config.vocab_size #分词器的词表规模大小
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == 'flash'
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if config.use_flash_attn:
            try:
                from flash_attn.ops.rms_norm import rms_norm 
                select_norm = 'NDLFlash_attnRMSNorm'
                self.norm = NDLFlash_attnRMSNorm(config.hidden_size, eps=config.rms_norm_eps)#RMSNorm层
            except:
                select_norm = 'NDLRMSNorm'
                self.norm = NDLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)#RMSNorm层
        self.layers = nn.ModuleList(
            [NDLDecoderlayer(config, layer_idx,select_norm) for layer_idx in range(config.num_hidden_layers)]
        ) # 32个decoder层
        # self.norm = NDLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)#RMSNorm层
        self.gradient_checkpointing = False # 允许模型在训练期间不对所有的中间层计算结果存储梯度，从而降低内存占用
        self.post_init() #权重初始化
        
    def get_input_embeddings(self): #得到输入embeddings向量
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    def forward(
        self,
        input_ids: torch.LongTensor = None, #input_ids是分词后的token_id,代表了语义信息
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # position_ids代表了位置信息，从0开始递增
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # cache_position: Optional[torch.LongTensor] = None,
    ) :
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        past_key_values_length = 0
        if use_cache: #如果使用kv-cache
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions: #如果使用sdpa进行加速以及不输出attention
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # 对定义输入的embeddings为hidden-states 
        hidden_states = inputs_embeds
        
        # decoder层
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        hidden_states = self.norm(hidden_states) #最后经过decoder layer 需要进行RMSNorm归一化
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


class NDLForCausalLM(NDLPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = NDLModel(config).bfloat16()
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).bfloat16() #经过所有decoder层后的输出层
        self.post_init()  #权重初始化
       
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
    
    def forward(
        self,
        input_ids: torch.LongTensor = None, #text2id的token_id序列
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, #序列中的
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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
        )
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
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        cache_position = None
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
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
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                # "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        generation_config = (
            generation_config
            if generation_config is not None
            else self.generation_config
        )
        stop_words_ids = kwargs.pop("stop_words_ids", None)
        if stop_words_ids is None and generation_config is not None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)
        if stop_words_ids is None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)

        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)

        return super().generate(
            inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            **kwargs,
        )

    
        

                 
if __name__=='__main__':
    # 实例化model
    ndlconfig = NDLConfig()
    model = NDLForCausalLM(ndlconfig)
    print(model)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"当前预训练模型的参数大小为: {model_size / 10000**2/10:.1f}B parameters")
    # for name, param in model.named_parameters():
    #     print(f'参数名称: {name}')
    #     # print(f'Parameter Value:\n{param}\n')
    #     print('===='*20)

