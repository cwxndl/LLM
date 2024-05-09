from transformers import PretrainedConfig

class NDLConfig(PretrainedConfig):
    model_type = "NdlMOE"
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        vocab_size=60930,
        hidden_size=1600,
        intermediate_size=14336,
        moe_intermediate_size = 1024, #MOE层的隐藏层维度
        n_shared_experts = 2, #共享专家个数
        n_routed_experts = 15,#MOE专家的个数（除共享专家以外）
        num_experts_per_tok = 4,#激活专家的数量
        moe_layer_freq = 1,#MOE层的间隔距离
        first_k_dense_replace = 1, #设置第二个decoder开始是MOE层，第一个decoder为普通的FFN层
        norm_topk_prob = False, #不对top_k专家权重进行归一化
        aux_loss_alpha = 0.001, #专家平衡因子
        scoring_func = 'softmax',#门控单元的归一化函数
        seq_aux = True,
        num_hidden_layers=12,
        num_attention_heads=32,
        num_key_value_heads=None,
        max_position_embeddings=1024,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=3,
        eos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_flash_attn=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_flash_attn = use_flash_attn
        self.moe_intermediate_size = moe_intermediate_size
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok =num_experts_per_tok
        self.moe_layer_freq =moe_layer_freq 
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.scoring_func = scoring_func
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

if __name__ =='__main__':
    config = NDLConfig(
        # _attn_implementation='sdpa'
    )
    print(config)