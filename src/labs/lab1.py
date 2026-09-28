import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch.nn as nn

# Visualization of transformer internals
# See: https://poloclub.github.io/transformer-explainer/

# Attention Matrix multiplication visualization
# See: https://pytorch.org/blog/inside-the-matrix/

# Load the model for inference
print("loading models...")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

weight_path = hf_hub_download(repo_id="openai-community/gpt2", filename="model.safetensors")

model_weights = load_file(weight_path)
print("models loaded.")

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

    # print(f"input = {input}")

    output_text: list[str] = ["" for _ in input]
    output_logits: list[list[torch.Tensor]] = [[] for _ in input]
    max_output_length = 0

    for i, input_text in enumerate(input):
        # print(f"i = {i}")
        while True:
            # print(f"input_text = {input_text}")
            input_tokens = construct_input_tensor(input_text, max_seq_length)

            # print(input_tokens.size(0))
            if input_tokens.size(0) >= max_seq_length:
                break

            next_word, next_token_logits = predict_next_token(input_tokens)

            if next_word == tokenizer.eos_token:
                break

            # record for output
            output_logits[i].append(next_token_logits)
            output_text[i] += next_word

            # update the input text with the newly predicted word
            input_text += next_word
            max_output_length = max(max_output_length, len(output_logits[i]))

    # append zero logits for sequences that have finished early
    for i, logits in enumerate(output_logits):
        if logits:
            seq_len = logits[0].size(0)
            while len(logits) < max_output_length:
                output_logits[i].append(torch.zeros_like(logits[0]))

    return output_text, torch.stack([torch.stack(logits) for logits in output_logits])
    # raise NotImplementedError("Implement GPT-2 Small here")

def construct_input_tensor(input: str, max_seq_length: int = 1024):
    '''
    output: token embedding + positional embedding
    output shape: max_seq_length * 768
    '''
    # tokenize and encode
    tokens = tokenizer.encode(input)[:max_seq_length]

    # transform into embedding
    tokens = [model_weights["wte.weight"][t] for t in tokens] # wte.weight is the word embedding for each token id
    tokens = torch.stack(tokens)

    # add positional embeddings
    positions = torch.arange(tokens.size(0)) # create a tensor of position. [0, 1, 2, ..., len(tokens) - 1]
    positions = [model_weights["wpe.weight"][p] for p in positions] # wpe.weight is the positional embedding for each position
    positions = torch.stack(positions)

    tokens = tokens + positions

    return tokens

def transformer_block(x: torch.Tensor, layer_idx: int):
    original_x = x.clone() # save the original input for residual connection

    # Layer norm 1
    with torch.no_grad():
        ln_1 = nn.LayerNorm(768, eps=1e-5)
        ln_1.weight.copy_(model_weights[f"h.{layer_idx}.ln_1.weight"])
        ln_1.bias.copy_(model_weights[f"h.{layer_idx}.ln_1.bias"])

    x = ln_1(x) # 1024 * 768

    # Self attention block
    n_head, head_dim, T = 12, 64, x.size(0)

    W = model_weights[f"h.{layer_idx}.attn.c_attn.weight"] # 768 * 2304 = [Q K V]
    b = model_weights[f"h.{layer_idx}.attn.c_attn.bias"] # 2304
    proj_w = model_weights[f"h.{layer_idx}.attn.c_proj.weight"] # 768 * 768
    proj_b = model_weights[f"h.{layer_idx}.attn.c_proj.bias"] # 768

    qkv = torch.matmul(x, W) + b # 1024 * 2304
    Q, K, V = qkv.split(768, dim=1) # 1024 * 768 each. each include 12 heads' matrix in horizaontal direction. ex: Q == (head 1 q weight) (head 2 q weight) ... (head 12 q weight)

    # Split Q, K, V into multiple heads for multi-head attention
    Q = Q.view(T, n_head, head_dim).transpose(0, 1) # 12 * 1024 * 64
    K = K.view(T, n_head, head_dim).transpose(0, 1) # 12 * 1024 * 64
    V = V.view(T, n_head, head_dim).transpose(0, 1) # 12 * 1024 * 64
    
    # Attention matrix
    mask = model_weights[f"h.{layer_idx}.attn.bias"] # 1 * 1 * 1024 * 1024, attention mask
    mask = mask[0, 0, :T, :T] 

    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5) # 12 * 1024 * 1024, scaled by sqrt of head dimension (64) to stabilize calculations
    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf')) # apply the attention mask before softmax

    attn_probs = torch.softmax(attn_scores, dim=-1) # 12 * 1024 * 1024
    
    value_output = torch.matmul(attn_probs, V) # 12 * 1024 * 64
    value_output = value_output.transpose(0, 1).contiguous().view(T, n_head * head_dim) # 1024 * 768, merge the head back
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
    mlp_activation = torch.nn.GELU(approximate='tanh')(mlp) # activation
    mlp_result = torch.matmul(mlp_activation, mlp_proj_w) + mlp_proj_b # 1024 * 768
    x = original_x + mlp_result

    return x

def predict_next_token(input_tokens):
    x = input_tokens

    for transformer_layer in range(12):
        x = transformer_block(x, transformer_layer)

    # Output

    ## Layer Norm (Final)
    ln = nn.LayerNorm(768, eps=1e-5)
    with torch.no_grad():
        ln.weight.copy_(model_weights[f"ln_f.weight"])
        ln.bias.copy_(model_weights[f"ln_f.bias"])
    x = ln(x) # 1024 * 768

    ## Logits
    wte = model_weights["wte.weight"] # 50257 * 768
    logits = torch.matmul(x, wte.t()) # 1024 * 50257

    ## softmax
    ## every row is the next token probability (it is the training objective). so we take the last row for the next token prediction
    next_token_probs = torch.softmax(logits[-1], dim=-1) # 1024 (sequence length) * 50257

    ## decode the next token
    next_token = torch.argmax(next_token_probs).item()

    return tokenizer.decode(next_token), logits

if __name__ == "__main__":
    text, logits = gpt2_complete(input=["Hello, my name is", "How are"], max_seq_length=10)
    print(text)

    # print(tokenizer.decode(tokenizer.eos_token_id))
    # print(tokenizer.eos_token)