# Tracing `scaled_dot_product_attention`: Python → CPU flash kernel

How PyTorch's `F.scaled_dot_product_attention` gets from a Python call down to the
C++ CPU flash-attention kernel. Paths and names are for the **`v2.11.0`** tag of
[pytorch/pytorch](https://github.com/pytorch/pytorch).

## Why this is worth tracing

The PyTorch docs show a Python snippet labeled *"Efficient implementation
**equivalent to** the following."* That is the mathematical spec, **not** the code
that runs. The real kernel is a fused C++ routine that:

1. **Accumulates in fp32** even for fp16 inputs (scratch buffers are `float`),
   rounding to fp16 only once at the end.
2. **Tiles** Q and K/V into blocks (`qSplitSize`, `kvSplitSize`).
3. Uses an **online (streaming) softmax** (running max + running normalizer).

Because floating-point addition isn't associative, this fused program differs from
a naive `Q@Kᵀ → softmax → @V` by ~1 ULP in fp16 — enough to flip a greedy argmax at
a razor-tie.

## Which backend actually runs?

Confirmed on torch **2.11.0**, CPU, fp16, with an attention mask:

```
      MATH: matches default? False  maxdiff=1.95e-03
     FLASH: matches default? True   maxdiff=0.00e+00   ← this is what runs
 EFFICIENT: unavailable
```

Detection script:

```python
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

q = torch.randn(1, 12, 10, 64, dtype=torch.float16)
k = torch.randn(1, 12, 10, 64, dtype=torch.float16)
v = torch.randn(1, 12, 10, 64, dtype=torch.float16)
mask = torch.tril(torch.ones(10, 10, dtype=torch.bool))[None, None]

default = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
for name, backend in [("MATH", SDPBackend.MATH),
                      ("FLASH", SDPBackend.FLASH_ATTENTION),
                      ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION)]:
    try:
        with sdpa_kernel([backend]):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        print(f"{name:>10}: matches default? {torch.equal(out, default)}  "
              f"maxdiff={(out-default).abs().max().item():.2e}")
    except Exception as e:
        print(f"{name:>10}: unavailable ({type(e).__name__})")
```

## The call chain (hop by hop)

### Hop 1 — Python entry
`torch/nn/functional.py`

```python
scaled_dot_product_attention = _add_docstr(
    torch._C._nn.scaled_dot_product_attention, ...)
```

`F.scaled_dot_product_attention` is only a docstring wrapper; the real callable is
the C-binding `torch._C._nn.scaled_dot_product_attention`.

### Hop 2 — the generated-glue gap (the catch)
`torch._C._nn....` → `at::scaled_dot_product_attention` passes through binding +
dispatcher code that is **generated at build time and is NOT in the repo** (it lands
in `torch/csrc/autograd/generated/python_nn_functions.cpp` and
`build/aten/.../Operators.cpp` after compilation).

You don't read those files — you cross the gap using the index:

`aten/src/ATen/native/native_functions.yaml` — grep for `scaled_dot_product_attention`:

```yaml
- func: scaled_dot_product_attention(Tensor query, ...) -> Tensor
  python_module: nn
  dispatch:
    CompositeImplicitAutograd: scaled_dot_product_attention   # ← next hop
```

`CompositeImplicitAutograd` means the op runs a plain C++ function (no per-device
kernel), which is why `torch._C._nn` lands directly in the next file.

### Hop 3 — the composite dispatcher (C++)
`aten/src/ATen/native/transformers/attention.cpp` — `Tensor scaled_dot_product_attention(...)` (~L490)

```cpp
Tensor scaled_dot_product_attention(
    const Tensor& query_, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& attn_mask_, double dropout_p,
    bool is_causal, std::optional<double> scale, bool enable_gqa)
```

Calls `_fused_sdp_choice_stub` to pick a backend, then switches (~L502–541):

```cpp
case SDPBackend::flash_attention:   // CPU branch:
    return std::get<0>(at::_scaled_dot_product_flash_attention_for_cpu(...));
case SDPBackend::math:
    return std::get<0>(at::_scaled_dot_product_attention_math(...));
// also: cudnn_attention, efficient_attention, overrideable
```

For CPU + fp16 the choice is `flash_attention`.

### Hop 4 — the CPU flash op → dispatch stub
`native_functions.yaml`:

```yaml
- func: _scaled_dot_product_flash_attention_for_cpu(...)
  dispatch:
    CPU: _scaled_dot_product_flash_attention_cpu
```

`aten/src/ATen/native/transformers/attention.cpp` — `_scaled_dot_product_flash_attention_cpu(...)` (~L565)

```cpp
std::tuple<at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_cpu(
    const Tensor& query, const Tensor& key, const Tensor& value,
    double dropout_p, bool is_causal,
    const std::optional<Tensor>& attn_mask, std::optional<double> scale)
```

Allocates `output`/`logsumexp` and calls the dispatch stub (~L593):

```cpp
flash_attention_kernel(kCPU, output, logsumexp,
    query, key, value, dropout_p, is_causal, attn_mask, scale);
```

`flash_attention_kernel` is a `DECLARE_DISPATCH` / `DEFINE_DISPATCH` stub — a
function pointer resolved per CPU architecture.

### Hop 5 — the actual algorithm (the "magic")
`aten/src/ATen/native/cpu/FlashAttentionKernel.cpp`

```cpp
REGISTER_DISPATCH(flash_attention_kernel, &flash_attention_kernel_impl)
```

`flash_attention_kernel_impl` switches on dtype and calls the template
`cpu_flash_attention<scalar_t, ...>` — the tiled, fp32-accumulating, online-softmax
loop. This is where the three properties above live.

## The math fallback (for contrast)
`aten/src/ATen/native/transformers/attention.cpp` — `_scaled_dot_product_attention_math(...)` (~L605).
This is the readable version closest to the docs' snippet; it is what runs under
`with sdpa_kernel([SDPBackend.MATH])` and is ~1e-3 off the flash kernel in fp16.

## How to navigate / verify it yourself

- **Index-first**: always start at `native_functions.yaml`, grep the op name, read
  the `dispatch:` block — it names the exact C++ symbol to grep for next. This is how
  you cross every generated-code gap.
- **Grep terms** (repo at tag `v2.11.0`):
  `Tensor scaled_dot_product_attention(`, `_fused_sdp_choice`,
  `_scaled_dot_product_flash_attention_cpu`,
  `DEFINE_DISPATCH(flash_attention_kernel`,
  `REGISTER_DISPATCH(flash_attention_kernel`, `cpu_flash_attention`.
- **Runtime confirmation without C++** — call the leaf ops directly:

  ```python
  # matches the default output (flash path)
  torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(q, k, v, 0.0, False, attn_mask=m)
  # the ~1e-3-off version (math path)
  torch.ops.aten._scaled_dot_product_attention_math(q, k, v, attn_mask=m)
  ```

- **Real step-through** (optional): the pip wheel is stripped, so gdb/lldb won't step
  in. Build from source instead:

  ```bash
  git clone --branch v2.11.0 https://github.com/pytorch/pytorch
  cd pytorch && DEBUG=1 python setup.py develop
  # then lldb can break on cpu_flash_attention
  ```

  Or set `TORCH_SHOW_DISPATCH_TRACE=1` (a debug-build env var) to print dispatch keys
  per op.

## Key file map

| Hop | File | Symbol |
|-----|------|--------|
| 1 | `torch/nn/functional.py` | `scaled_dot_product_attention` (docstring wrapper) |
| 2 | `aten/src/ATen/native/native_functions.yaml` | `scaled_dot_product_attention` entry (index) |
| 3 | `aten/src/ATen/native/transformers/attention.cpp` | `scaled_dot_product_attention(...)` + `_fused_sdp_choice_stub` |
| 4 | `aten/src/ATen/native/transformers/attention.cpp` | `_scaled_dot_product_flash_attention_cpu(...)` → `flash_attention_kernel(...)` |
| 5 | `aten/src/ATen/native/cpu/FlashAttentionKernel.cpp` | `REGISTER_DISPATCH(flash_attention_kernel, ...)` → `cpu_flash_attention<...>` |
| — | `aten/src/ATen/native/transformers/attention.cpp` | `_scaled_dot_product_attention_math(...)` (fallback) |
