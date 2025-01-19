import torch
from model135m import SmolLM2Config135M, SmolLM2ForCausalLM135M
from transformers import AutoTokenizer

def count_parameters(model):
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(config, model):
    """Print detailed model information."""
    print("\nModel Architecture:")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"Number of KV heads: {config.num_key_value_heads}")
    print(f"Vocabulary size: {config.vocab_size}")
    
    # Calculate parameter counts for each component
    embed_params = config.vocab_size * config.hidden_size
    attention_params_per_layer = (
        3 * config.hidden_size * config.hidden_size +  # q, k, v projections
        config.hidden_size * config.hidden_size        # output projection
    )
    mlp_params_per_layer = (
        2 * config.hidden_size * config.intermediate_size +  # up & gate proj
        config.intermediate_size * config.hidden_size        # down proj
    )
    norm_params_per_layer = 2 * config.hidden_size  # 2 layer norms
    final_norm_params = config.hidden_size
    lm_head_params = config.vocab_size * config.hidden_size
    
    total_per_layer = attention_params_per_layer + mlp_params_per_layer + norm_params_per_layer
    total_transformer = total_per_layer * config.num_hidden_layers
    total_params = embed_params + total_transformer + final_norm_params + lm_head_params
    
    print("\nParameter Counts:")
    print(f"Embeddings: {embed_params:,}")
    print(f"Per layer:")
    print(f"  - Attention: {attention_params_per_layer:,}")
    print(f"  - MLP: {mlp_params_per_layer:,}")
    print(f"  - Layer Norms: {norm_params_per_layer:,}")
    print(f"Total per layer: {total_per_layer:,}")
    print(f"All layers: {total_transformer:,}")
    print(f"Final norm: {final_norm_params:,}")
    print(f"LM head: {lm_head_params:,}")
    print(f"\nTotal theoretical params: {total_params:,}")
    
    actual_params = count_parameters(model)
    print(f"Actual model params: {actual_params:,}")
    
    # Memory estimation
    bytes_per_param = 2  # for bfloat16/float16
    model_size_gb = actual_params * bytes_per_param / (1024**3)
    print(f"\nEstimated model size in memory (FP16): {model_size_gb:.2f} GB")

def test_generation():
    """Test the model's generation capabilities."""
    print("\nInitializing model and tokenizer...")
    config = SmolLM2Config135M()
    model = SmolLM2ForCausalLM135M(config)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    
    print_model_info(config, model)
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    model = model.to(device)
    model.eval()
    
    # Test input
    test_prompt = "Once upon a time"
    print(f"\nTest prompt: {test_prompt}")
    
    inputs = tokenizer(
        test_prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)
    
    print("Input shape:", inputs.input_ids.shape)
    print("Input tokens:", tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))
    
    try:
        with torch.no_grad():
            # Test forward pass
            outputs = model(**inputs)
            print("\nForward pass successful!")
            print("Output logits shape:", outputs.logits.shape)
            
            # Test generation
            generation_config = {
                "max_length": 50,
                "min_length": 20,
                "do_sample": True,
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "use_cache": True,
            }
            
            output_sequences = model.generate(
                **inputs,
                **generation_config
            )
            
            generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            print("\nGeneration successful!")
            print(f"Generated text: {generated_text}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing SmolLM2-135M model...")
    test_generation() 