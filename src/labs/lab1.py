import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch.nn as nn

# Visualization of transformer internals
# See: https://poloclub.github.io/transformer-explainer/

# Attention Matrix multiplication visualization
# See: https://pytorch.org/blog/inside-the-matrix/

def gpt2_complete(
    input: list[str],
    max_seq_length: int = 1024,
) -> tuple[list[str], torch.Tensor]:
    """Generate greedy completions with a from-scratch GPT-2 Small implementation.

    Load pretrained GPT-2 Small weights into manually implemented transformer
    blocks. Generate for the entire batch at once, choosing the highest-logit
    token for every unfinished sequence at each step. Stop each sequence at EOS
    or max_seq_length total tokens, including the prompt.

    Return newly generated text for each prompt and a tensor of pre-selection
    logits shaped (batch_size, decoding_steps, 50257). Fill logits with zero
    after a row has finished while other rows continue.
    """
    raise NotImplementedError("Implement GPT-2 Small here")

# Prepare the model for inference

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

weight_path = hf_hub_download(repo_id="openai-community/gpt2", filename="model.safetensors")

model_weights = load_file(weight_path)

def construct_input_tensor(input: str):
    '''
    output: token embedding + positional embedding
    output shapt: 1024 * 768
    '''
    # tokenize and encode
    tokens = tokenizer.encode(input)

    # transform into embedding
    tokens = [model_weights["wte.weight"][t] for t in tokens] # wte.weight is the word embedding for each token id
    tokens = torch.stack(tokens)

    # add positional embeddings
    positions = torch.arange(tokens.size(0)) # create a tensor of position. [0, 1, 2, ...]
    positions = [model_weights["wpe.weight"][p] for p in positions] # wpe.weight is the positional embedding for each position
    positions = torch.stack(positions)

    tokens = tokens + positions

    return tokens

# shared layers
layer_norm = nn.LayerNorm(768)

def transformer_block(x: torch.Tensor, layer_idx: int):
    original_x = x.clone() # save the original input for residual connection

    # Layer norm 1
    with torch.no_grad():
        ln_1 = nn.LayerNorm(768, eps=1e-5)
        ln_1.weight.copy_(model_weights[f"h.{layer_idx}.ln_1.weight"])
        ln_1.bias.copy_(model_weights[f"h.{layer_idx}.ln_1.bias"])

    x = ln_1(x) # 1024 * 768

    # Self attention block
    W = model_weights[f"h.{layer_idx}.attn.c_attn.weight"] # 768 * 2304 = [Q K V]
    b = model_weights[f"h.{layer_idx}.attn.c_attn.bias"] # 2304
    proj_w = model_weights[f"h.{layer_idx}.attn.c_proj.weight"] # 768 * 768
    proj_b = model_weights[f"h.{layer_idx}.attn.c_proj.bias"] # 768

    qkv = torch.matmul(x, W) + b # 1024 * 2304
    Q, K, V = qkv.split(768, dim=1) # 1024 * 768 each. each include 12 heads' matrix in horizaontal direction. ex: Q == (head 1 q weight) (head 2 q weight) ... (head 12 q weight)

    # Attention matrix
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (64 ** 0.5) # 1024 * 1024, scaled by sqrt of head dimension (64) to stabilize calculations
    attn_probs = torch.softmax(attn_scores, dim=-1) # 1024 * 1024
    value_output = torch.matmul(attn_probs, V) # 1024 * 768
    attention_output = torch.matmul(value_output, proj_w) + proj_b # 1024 * 768
    x = original_x + attention_output # residual connection

    original_x = x.clone() # save the original input for residual connection

    # Layer norm 2
    with torch.no_grad():
        ln_2 = nn.LayerNorm(768, eps=1e-5)
        ln_2.weight.copy_(model_weights[f"h.{layer_idx}.ln_2.weight"])
        ln_2.bias.copy_(model_weights[f"h.{layer_idx}.ln_2.bias"])

    x = ln_2(x)

    # Feed-forward MLP
    mlp_w = model_weights[f"h.{layer_idx}.mlp.c_fc.weight"] # 768 * 3072
    mlp_b = model_weights[f"h.{layer_idx}.mlp.c_fc.bias"] # 3072
    mlp_proj_w = model_weights[f"h.{layer_idx}.mlp.c_proj.weight"] # 3072 * 768
    mlp_proj_b = model_weights[f"h.{layer_idx}.mlp.c_proj.bias"] # 768

    mlp = torch.matmul(x, mlp_w) + mlp_b # 1024 * 3072
    mlp_activation = torch.nn.GELU()(mlp) # activation
    mlp_result = torch.matmul(mlp_activation, mlp_proj_w) + mlp_proj_b # 1024 * 768
    x = original_x + mlp_result

    return x

x = construct_input_tensor("Hello World")
x = transformer_block(x, 1)

print(x)
