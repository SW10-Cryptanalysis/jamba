from transformers import JambaConfig, JambaForCausalLM

def get_jamba_model(config):
    # JambaConfig uses specific parameters for its hybrid architecture
    conf = JambaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.dims,
        num_hidden_layers=config.layers,
        intermediate_size=config.dims * 4, # Standard MLP expansion
        num_attention_heads=config.dims // 64, # Standard head dimension
        num_key_value_heads=1, # Uses Grouped Query Attention
        attn_layer_period=8, # One attention layer every 8 layers
        attn_layer_offset=4,
        num_experts=1, # Set to 1 for a dense model, or e.g., 4-8 for a small MoE
        num_experts_per_tok=1,
        max_position_embeddings=config.max_context,
        use_mamba_kernels=True # Assumes mamba-ssm is installed
    )
    
    model = JambaForCausalLM(conf)
    print("Jamba Model loaded!")
    print(f"Parameters:       {model.num_parameters():,}")
    print(f"VRAM for Weights: {(model.get_memory_footprint() / 1e9):.4f} GB")

    return model