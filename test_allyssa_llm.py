import pytest
import torch
import torch.nn as nn
from alyssa_llm import AllyssaLLM, CausalSelfAttention, Block, vocab_size, block_size, n_embd

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def model(device):
    torch.manual_seed(42)
    m = AllyssaLLM().to(device)
    m.eval()
    return m

# ---------------------------------------------------------
# Test Suite
# ---------------------------------------------------------

def test_forward_output_shape(model, device):
    """Test output tensor dimensions match (B, T, vocab_size)."""
    B, T = 4, 16
    x = torch.randint(0, vocab_size, (B, T), device=device)
    
    logits, loss = model(x)
    
    assert logits.shape == (B, T, vocab_size), f"Expected {(B, T, vocab_size)}, got {logits.shape}"
    assert loss is None, "Loss should be None when no targets are supplied"


def test_loss_computation(model, device):
    """Test loss computation returns a valid, non-zero scalar."""
    B, T = 2, 8
    x = torch.randint(0, vocab_size, (B, T), device=device)
    targets = torch.randint(0, vocab_size, (B, T), device=device)
    
    logits, loss = model(x, targets=targets)
    
    assert loss is not None
    assert loss.dim() == 0, "Loss must be a scalar"
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss contains NaN or Inf"
    assert loss.item() > 0.0, "Loss must be positive"


def test_gradient_flow_and_backprop(device):
    """Verify gradients propagate through all trainable parameters."""
    torch.manual_seed(42)
    model = AllyssaLLM().to(device)
    model.train()
    
    x = torch.randint(0, vocab_size, (2, 8), device=device)
    targets = torch.randint(0, vocab_size, (2, 8), device=device)
    
    _, loss = model(x, targets=targets)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients."
            assert not torch.all(param.grad == 0), f"Parameter {name} has zero gradients."


def test_causal_mask_no_leakage(model, device):
    """
    CRITICAL LLM TEST:
    Verify that modifying token at index t > 0 does NOT affect
    the output representation of tokens at indices < t.
    """
    seq_len = 10
    tokens_base = torch.randint(0, vocab_size, (1, seq_len), device=device)
    tokens_modified = tokens_base.clone()
    
    # Mutate only the last token
    tokens_modified[0, -1] = (tokens_base[0, -1] + 1) % vocab_size

    with torch.no_grad():
        logits_base, _ = model(tokens_base)
        logits_mod, _ = model(tokens_modified)

    # Logits for tokens 0 through seq_len - 2 MUST be strictly identical
    diff = torch.abs(logits_base[:, :-1, :] - logits_mod[:, :-1, :]).max().item()
    assert diff == 0.0, f"Future token leaked backward! Max difference: {diff}"


def test_generation_extension(model, device):
    """Verify autoregressive generation produces correct sequence lengths."""
    start_tokens = torch.tensor([[1, 2, 3]], device=device)  # (1, 3)
    num_new_tokens = 5
    
    generated = model.generate(start_tokens, max_new_tokens=num_new_tokens, temperature=0.8, top_k=5)
    
    assert generated.shape == (1, 3 + num_new_tokens)
    # Check that initial prefix remains intact
    assert torch.equal(generated[:, :3], start_tokens)


def test_context_window_clipping(model, device):
    """Verify generation gracefully handles inputs larger than block_size."""
    long_tokens = torch.randint(0, vocab_size, (1, block_size + 10), device=device)
    
    # Should not raise IndexError or CUDA assert
    generated = model.generate(long_tokens, max_new_tokens=2)
    assert generated.shape == (1, block_size + 12)