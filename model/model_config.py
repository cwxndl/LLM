from transformers import PretrainedConfig

class NDLConfig(PretrainedConfig):
    model_type = "Ndl"
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        vocab_size=60930,
        hidden_size=1600,
        intermediate_size=6*1300,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=None,
        max_position_embeddings=512,
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
        attn_implementation='sdpa',
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_flash_attn = use_flash_attn

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