import pytest
import torch

from src.models.modules import MultiheadAttention
from src.utils import set_seed


@pytest.fixture
def attention():
    return MultiheadAttention(heads=4, embedding_size=512, dropout=0.0, bias=False)


@pytest.fixture
def input_seq():
    return torch.randn(16, 32, 512)


@pytest.fixture
def mask():
    return torch.tril(torch.ones(32, 32))


def test_output_shape(attention, input_seq, mask):
    output_seq = attention(input_seq, mask)
    assert output_seq.shape == (16, 32, 512)


def test_forward_without_mask(attention, input_seq):
    output_seq = attention(input_seq)
    assert output_seq.shape == (16, 32, 512)


def test_masked_self_attention(attention, input_seq, mask, capsys):
    # Test if masked self-attention return -inf for the values that should be ignored
    set_seed(42)
    batch_size, seq_length = input_seq.shape[:-1]
    keys = torch.randn(batch_size * attention.heads, seq_length, attention.head_dim)
    queries = torch.randn(batch_size * attention.heads, seq_length, attention.head_dim)
    values = torch.randn(batch_size * attention.heads, seq_length, attention.head_dim)

    attention_out = attention._self_attention(queries, keys, values, mask=mask)

    dot_product = torch.bmm(queries, keys.clone().transpose(1, 2)) / (attention.head_dim**0.5)
    dot_product = dot_product.masked_fill(mask == 0, float("-inf"))
    attention_weights = attention.softmax(dot_product)
    # attention_weights = attention.dropout(attention_weights)
    expected_output = torch.bmm(attention_weights, values)  # Skipping dropout
    assert torch.allclose(attention_out, expected_output, rtol=1e-03, atol=1e-03)
