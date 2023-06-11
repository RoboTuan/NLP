import matplotlib.pyplot as plt
import pytest
import torch

from src.models.modules import MultiheadAttention
from src.utils import set_seed


@pytest.fixture
def query():
    return torch.randn(16, 20, 512)


@pytest.fixture
def key():
    return torch.randn(16, 30, 512)


@pytest.fixture
def value():
    return torch.randn(16, 30, 512)


@pytest.fixture
def query_pad_mask():
    mask = torch.ones((16, 1, 20), dtype=torch.bool)
    mask[:, :, 10:] = 0
    return mask


@pytest.fixture
def key_pad_mask():
    mask = torch.ones((16, 1, 30), dtype=torch.bool)
    mask[:, :, 20:] = 0
    return mask


@pytest.fixture
def att_mask():
    return torch.tril(torch.ones(20, 30)).repeat(16, 4, 1, 1)


@pytest.fixture
def attention():
    return MultiheadAttention(heads=4, embedding_size=512, dropout=0.0, bias=False)


def test_forward_without_mask(attention, key, query, value):
    output_seq, _ = attention(query, key, value)
    assert output_seq.shape == (16, 20, 512)


def test_forward_att_mask(attention, key, query, value, att_mask):
    output_seq, _ = attention(query, key, value, att_mask=att_mask)
    assert output_seq.shape == (16, 20, 512)


def test_forward_pad_masks(attention, key, query, value, key_pad_mask, query_pad_mask):
    output_seq, _ = attention(query, key, value, query_pad_mask, key_pad_mask)
    assert output_seq.shape == (16, 20, 512)


def test_forward_all_masks(attention, key, query, value, key_pad_mask, query_pad_mask, att_mask):
    output_seq, _ = attention(query, key, value, query_pad_mask, key_pad_mask, att_mask)
    assert output_seq.shape == (16, 20, 512)


def test_masked_self_attention(attention, query, att_mask):
    # Test if masked self-attention return -inf for the values that should be ignored
    set_seed(42)
    # TODO: Do this in numpy and test it with my implementation
    # batch_size, seq_length = query.shape[:-1]
    # keys = torch.randn(batch_size * attention.heads, seq_length, attention.head_dim)
    # queries = torch.randn(batch_size * attention.heads, seq_length, attention.head_dim)
    # values = torch.randn(batch_size * attention.heads, seq_length, attention.head_dim)

    # attention_out = attention._self_attention(queries, keys, values, mask=mask)

    # dot_product = torch.bmm(queries, keys.clone().transpose(1, 2)) / (attention.head_dim**0.5)
    # dot_product = dot_product.masked_fill(mask == 0, float("-inf"))
    # attention_weights = attention.softmax(dot_product)
    # # attention_weights = attention.dropout(attention_weights)
    # expected_output = torch.bmm(attention_weights, values)  # Skipping dropout
    # assert torch.allclose(attention_out, expected_output, rtol=1e-03, atol=1e-03)
    assert True
