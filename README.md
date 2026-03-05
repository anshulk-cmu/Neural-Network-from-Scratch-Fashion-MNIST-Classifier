# Neural Network from Scratch — Fashion-MNIST Classifier

## Overview

This project implements a **single-hidden-layer neural network from scratch** using PyTorch tensors (but **no built-in modules** like `nn.Linear`, `nn.Sigmoid`, etc.). The network classifies 28×28 grayscale images from the **Fashion-MNIST** dataset into 10 clothing categories.

### Network Architecture

```
Input (784) → Linear → Sigmoid → Linear → Softmax → Output (10)
```

| Layer       | Input Size | Output Size | Description                        |
|-------------|------------|-------------|------------------------------------|
| Flatten     | (1, 28, 28) | (784,)    | Reshape image to vector            |
| Linear 1    | 784        | 256         | Weights α ∈ ℝ^(256×784), bias ∈ ℝ^256 |
| Sigmoid     | 256        | 256         | σ(a) = 1/(1+e^(-a))               |
| Linear 2    | 256        | 10          | Weights β ∈ ℝ^(10×256), bias ∈ ℝ^10  |
| Softmax     | 10         | 10          | Inside CrossEntropyFunction        |

---

## File Structure

```
nn_implementation_code/
├── custom_functions.py    ← Core math: forward & backward for each operation
├── custom_modules.py      ← Wraps Functions into nn.Module-compatible layers
├── base_experiment.py     ← Model definition + training/evaluation functions
├── weights.pt             ← Initial weights (provided, must be loaded before training)
└── __init__.py

check/
├── a.txt                  ← Expected first 5 hidden pre-activations for data point 1
├── z.txt                  ← Expected first 5 sigmoid outputs for data point 1
├── b.txt                  ← Expected first 5 output logits for data point 1
└── updated_params.pt      ← Expected weights after 1 SGD step on data point 1
```

---

## Phase 1: Core Math (`custom_functions.py`)

All four custom autograd `Function` classes live here. Each has a `forward()` (compute the output) and `backward()` (compute gradients via chain rule). PyTorch's autograd engine calls `backward()` automatically during `loss.backward()`.

### 1.1 — `SigmoidFunction.forward(ctx, input)`

**What it does:** Computes the sigmoid activation element-wise.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Implementation:**
```python
output = 1.0 / (1.0 + torch.exp(-input))
ctx.save_for_backward(output)
return output
```

**Why save `output` (not `input`)?**  
The backward formula is σ·(1−σ), which is expressed in terms of the *output*. Saving the output avoids recomputing sigmoid during backward.

**Handles:** Any tensor shape — 1D vectors (batch_size=1) or 2D matrices (mini-batches).

---

### 1.2 — `SigmoidFunction.backward(ctx, grad_output)`

**What it does:** Applies the chain rule. The local derivative of sigmoid is:

$$\frac{d\sigma}{da} = \sigma(a) \cdot (1 - \sigma(a))$$

**Implementation:**
```python
(output,) = ctx.saved_tensors
return grad_output * output * (1.0 - output)
```

**How the chain rule works here:**
- `grad_output` arrives from the layer above (Linear 2's backward) — it's ∂Loss/∂z
- We multiply element-wise by the local derivative σ·(1−σ) — that's ∂z/∂a
- Result is ∂Loss/∂a, which gets passed down to Linear 1's backward

---

### 1.3 — `LinearFunction.forward(ctx, inp, weight, bias)`

**What it does:** Computes the standard linear (affine) transformation:

$$\text{output} = \text{inp} \cdot W^T + b$$

Where `inp` is (batch, in_features), `weight` is (out_features, in_features), `bias` is (out_features,).

**Implementation:**
```python
ctx.save_for_backward(inp, weight, bias)
output = inp @ weight.t() + bias
return output
```

**Why save all three?** Each is needed to compute one of the three gradients in backward.

---

### 1.4 — `LinearFunction.backward(ctx, grad_output)`

**What it does:** Computes three gradients — one for each input to `forward`:

| Gradient | Formula | Shape | Meaning |
|----------|---------|-------|---------|
| `grad_inp` | `grad_output @ weight` | (batch, in_features) | How loss changes w.r.t. layer input |
| `grad_weight` | `grad_output.T @ inp` | (out_features, in_features) | How loss changes w.r.t. each weight |
| `grad_bias` | `grad_output.sum(dim=0)` | (out_features,) | How loss changes w.r.t. each bias |

**Implementation:**
```python
inp, weight, bias = ctx.saved_tensors

# Handle 1D (single sample) by unsqueezing to 2D
if grad_output.dim() == 1:
    grad_output_2d = grad_output.unsqueeze(0)
    inp_2d = inp.unsqueeze(0)
else:
    grad_output_2d = grad_output
    inp_2d = inp

grad_inp = grad_output_2d @ weight
grad_weight = grad_output_2d.t() @ inp_2d
grad_bias = grad_output_2d.sum(dim=0)

# Restore original shape
if inp.dim() == 1:
    grad_inp = grad_inp.squeeze(0)

return grad_inp, grad_weight, grad_bias
```

**Why sum for bias?** Bias is shared across all samples in a batch, so gradients from each sample are summed.

**Why handle 1D?** When batch_size=1, PyTorch may pass 1D tensors. Matrix multiplications (`@`) require 2D, so we unsqueeze and then squeeze back.

---

### 1.5 — `CrossEntropyFunction.forward(ctx, logits, target)`

**What it does:** Computes softmax + cross-entropy loss in one numerically stable step.

**The math:**

$$\text{Loss} = -\frac{1}{N} \sum_{n=1}^{N} \log\left(\frac{e^{b_{y_n}}}{\sum_k e^{b_k}}\right)$$

**Implementation (step by step):**

```python
# (a) Numerical stability: subtract max per sample
max_logits = logits.max(dim=1, keepdim=True).values
shifted = logits - max_logits

# (b) Log-softmax (stays in log-space to avoid log(tiny_number))
log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
log_softmax = shifted - log_sum_exp

# (c) Softmax probabilities (for backward)
softmax_probs = torch.exp(log_softmax)

# (d) Pick log-prob at true class, negate, average
loss_per_sample = -log_softmax[torch.arange(batch_size), target]
loss = loss_per_sample.mean()
```

**Why subtract max?**  
Without it, `exp(1000)` = infinity → NaN. With the max subtracted, the largest exponent is `exp(0) = 1`, and everything stays finite. The math is identical:

$$\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - b}}{\sum_j e^{x_j - b}}$$

**Why log-softmax instead of log(softmax)?**  
`softmax` can produce values like 1e-38, and `log(1e-38)` = -87.5 which is fine. But if softmax underflows to exactly 0.0, then `log(0)` = -∞. Computing log-softmax directly avoids this.

**Why save `softmax_probs` but store `target` on `ctx` directly?**  
`ctx.save_for_backward()` only accepts float tensors. `target` is a `LongTensor` (integers), so it's stored as `ctx.target = target`.

---

### 1.6 — `CrossEntropyFunction.backward(ctx, grad_output)`

**What it does:** The gradient of cross-entropy w.r.t. logits has a famously clean form:

$$\frac{\partial J}{\partial b_k} = \frac{\hat{y}_k - y_k}{N}$$

Where ŷ is the softmax output and y is one-hot encoded target.

**Implementation:**
```python
(softmax_probs,) = ctx.saved_tensors
target = ctx.target
batch_size = ctx.batch_size

one_hot = torch.zeros_like(softmax_probs)
one_hot[torch.arange(batch_size), target] = 1.0

grad_logits = (softmax_probs - one_hot) / batch_size * grad_output
return grad_logits, None  # None for target (not differentiable)
```

**Why `/ batch_size`?** Because forward computed the *mean* loss, so the gradient includes 1/N.

**Why return `None` for target?** Integer class labels have no gradient — they're discrete, not continuous.

---

### 1.7 — Gradcheck Validation

Running `python custom_functions.py` executes three `torch.autograd.gradcheck()` tests:

```
✅ Backward pass for sigmoid function is implemented correctly
✅ Backward pass for linear function is implemented correctly
✅ Backward pass for crossentropy function is implemented correctly
```

**What gradcheck does:** For each input element, it perturbs the value by a tiny ε, recomputes the forward output, and compares the numerical gradient `(f(x+ε) - f(x-ε)) / 2ε` against your analytical backward gradient. They must match within tolerance. This is the gold-standard test for backward correctness.

---

## Phase 2: Module Wrappers (`custom_modules.py`)

This file was **already provided and fully implemented**. It wraps each `Function` into an `nn.Module` subclass:

| Module | Wraps | Has Parameters? |
|--------|-------|-----------------|
| `Linear` | `LinearFunction` | Yes: `weight`, `bias` |
| `Sigmoid` | `SigmoidFunction` | No |
| `CrossEntropyLoss` | `CrossEntropyFunction` | No |

**Why modules?** `Function` handles the math. `Module` adds:
- Parameter management (`weight`, `bias` as `nn.Parameter`)
- State dict for saving/loading weights
- Integration with `nn.Module` ecosystem (e.g., `model.parameters()` for optimizers)

**Validation:**
```
✅ Our fully connected layer has exactly the same interface as torch.nn.Linear
```

This confirmed that `Linear(3, 2)` with the same weights as `torch.nn.Linear(3, 2)` produces identical output.

---

## Phase 3: Model Definition (`base_experiment.py`)

### 3.1 — `FashionMNISTModel.__init__()`

```python
self.lin1 = Linear(784, 256)    # Input → Hidden
self.sigmoid = Sigmoid()         # Activation
self.lin2 = Linear(256, 10)     # Hidden → Output
```

**Critical naming:** The names `lin1` and `lin2` must match the keys in `weights.pt`:
- `lin1.weight` (256×784), `lin1.bias` (256,)
- `lin2.weight` (10×256), `lin2.bias` (10,)

If named differently (e.g., `self.fc1`), `model.load_state_dict()` would fail with a key mismatch.

### 3.2 — `FashionMNISTModel.forward(x)`

```python
def forward(self, x):
    if x.dim() == 1:
        x = x.unsqueeze(0)           # Ensure batch dimension
    x_flat = x.view(x.size(0), -1)   # (batch, 1, 28, 28) → (batch, 784)
    a = self.lin1(x_flat)             # Pre-activations
    z = self.sigmoid(a)               # Sigmoid activations
    logits = self.lin2(z)             # Output logits
    return logits
```

**No softmax here!** Softmax is inside `CrossEntropyFunction.forward()`. The model outputs raw logits.

---

## Phase 4: Forward + Backward Verification

### 4.1 — Forward Pass Check

Loaded `weights.pt`, passed the first training image through the model layer-by-layer, and compared against check files:

| Check File | Layer Output | Our Values | Expected Values | Match? |
|------------|-------------|------------|-----------------|--------|
| `a.txt` | Pre-activation (first 5) | `[1.4837, 1.5228, 1.5733, 1.4949, 1.4668]` | `[1.4837, 1.5228, 1.5733, 1.4949, 1.4668]` | ✅ |
| `z.txt` | Sigmoid (first 5) | `[0.8151, 0.8210, 0.8282, 0.8168, 0.8126]` | `[0.8151, 0.8210, 0.8282, 0.8168, 0.8126]` | ✅ |
| `b.txt` | Logits (first 5) | `[0.9807, 1.0680, 1.0840, 0.9548, 1.0708]` | `[0.9807, 1.0680, 1.0840, 0.9548, 1.0708]` | ✅ |

### 4.2 — Full Pipeline Check (Forward + Backward + SGD Step)

After one complete cycle (forward → loss → backward → SGD step) on the first training image, compared all model parameters against `updated_params.pt`:

| Parameter | Max Difference | Match? |
|-----------|---------------|--------|
| `lin1.weight` | 9.31e-10 | ✅ |
| `lin1.bias` | 4.09e-12 | ✅ |
| `lin2.weight` | 1.86e-09 | ✅ |
| `lin2.bias` | 2.33e-10 | ✅ |

All differences are within float32 precision (~1e-7), confirming the entire pipeline is correct.

---

## Data Flow Diagram

```
                    FORWARD PASS
                    ============
Image (1,28,28)
    │
    ▼ flatten
x (1, 784)
    │
    ▼ lin1: x @ α^T + α_bias
a (1, 256)  ← pre-activations [check: a.txt]
    │
    ▼ sigmoid: 1/(1+exp(-a))
z (1, 256)  ← hidden activations [check: z.txt]
    │
    ▼ lin2: z @ β^T + β_bias
b (1, 10)   ← logits [check: b.txt]
    │
    ▼ softmax + cross-entropy (inside CrossEntropyFunction)
loss (scalar)


                    BACKWARD PASS
                    =============
loss = 1.0 (grad_output)
    │
    ▼ CrossEntropy backward: (softmax - one_hot) / N
∂L/∂b (1, 10)
    │
    ▼ Linear2 backward:
    ├── ∂L/∂z = ∂L/∂b @ β          → passed to sigmoid backward
    ├── ∂L/∂β = ∂L/∂b^T @ z        → used by SGD to update β
    └── ∂L/∂β_bias = sum(∂L/∂b)    → used by SGD to update β_bias
    │
    ▼ Sigmoid backward: ∂L/∂z * σ * (1-σ)
∂L/∂a (1, 256)
    │
    ▼ Linear1 backward:
    ├── ∂L/∂x = ∂L/∂a @ α          → not needed (input layer)
    ├── ∂L/∂α = ∂L/∂a^T @ x        → used by SGD to update α
    └── ∂L/∂α_bias = sum(∂L/∂a)    → used by SGD to update α_bias


                    SGD UPDATE
                    ==========
    θ_new = θ_old - lr * ∂L/∂θ     (for each parameter θ)
```

---

## Remaining Work (Phases 5+)

- **Phase 5:** Implement `q1_to_q6()` — full training loop with batch_size=1, 15 epochs
- **Phase 6:** Implement `q7()` — batch_size=5, 50 epochs
- **Phase 7:** Questions 8–13 — confusion matrices, error analysis, batch size experiments, hyperparameter exploration

---

## Key Constraints

1. **No built-in PyTorch modules** — everything in `custom_functions.py` is manual
2. **No data shuffling** — process in original dataset order
3. **Initial weights from `weights.pt`** — must load before training
4. **Hyperparameters:** lr=0.01, hidden=256, epochs=15 (Q1-Q6), batch_size=1 (Q1-Q6)
