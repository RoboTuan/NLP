import torch
import torch.nn as nn
from torch import Tensor


class MultiheadAttention(nn.Module):
    r"""
    Transformers paper: https://arxiv.org/abs/1706.03762

    Multi-head self-attention module that maps input sequences to output sequences using a weighted sum of
    key-value pairs, where the weight assigned to each pair is determined by a compatibility function between
    the input and the key. This module supports dropout and masking for use in sequence-to-sequence models.

    Args:
        heads (int): The number of parallel self-attention heads to use.
        embedding_size (int): The size of the input embedding.
        dropout (float): The dropout probability to use after the attention weights are computed.
        bias (bool): Whether to include a bias term in the key, query, value, and output linear transformations.

    Examples:
        >>> attention = MultiheadAttention(heads=4, embedding_size=32, dropout=0.1, bias=False)
        >>> input_seq = torch.randn(16, 32, 512)
        >>> mask = torch.tril(torch.ones(32, 32))
        >>> output_seq = attention(input_seq, mask)
    """

    def __init__(self, heads: int, embedding_size: int, dropout: float, bias: bool) -> None:
        super().__init__()

        if embedding_size % heads != 0:
            raise ValueError(f"embedding_size ({embedding_size}) must be divisible by heads ({heads})")

        self.heads = heads  # Number of heads
        self.embedding_size = embedding_size  # Size of the input embedding
        self.head_dim = embedding_size // heads  # Size of each head

        # Key, Query, Value mappings. Can be done more efficiently with a single linear layer with the
        # output size equal to 3 * embedding_size
        self.keys_map = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.queries_map = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.values_map = nn.Linear(embedding_size, embedding_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        # Output mapping for multihead attention
        self.output_map = nn.Linear(embedding_size, embedding_size, bias=bias)

    def _self_attention(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor = None) -> Tensor:
        r"""
        Compute the scaled self-attention for a set of queries, keys, and values. To use it as an
        auto-regressive model, it supports masking the attention weights (i.e. not looking forward).

        Args:
            queries (Tensor): The tensor representing the queries.
            keys (Tensor): The tensor representing the keys.
            values (Tensor): The tensor representing the values.
            mask (Tensor): An optional mask applied to the attention dot product. If not None, it should be a lower
                triangular matrix of ones (diagonal ingluded) and zeros in the upper half.

        Returns:
            The output tensor after applying the self-attention mechanism.
        """

        dot_product = torch.bmm(queries, keys.transpose(1, 2)) / (self.head_dim**0.5)

        if mask is not None:
            dot_product = dot_product.masked_fill(mask == 0, float("-inf"))

        attention_weights = self.softmax(dot_product)
        attention_weights = self.dropout(attention_weights)
        attention_out = torch.bmm(attention_weights, values)

        return attention_out

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # The input tensor 'x' has shape (batch_size, seq_length, embedding_size)
        batch_size, seq_length = x.shape[:-1]

        # Tet the keys, queries and values from the input tensor. Reshape from (batch_size, seq_length, embedding_size)
        # to (batch_size, seq_length, heads, head_dim), considering that embedding_size = heads * head_dim.
        keys = self.keys_map(x).view(batch_size, seq_length, self.heads, self.head_dim)
        queries = self.queries_map(x).view(batch_size, seq_length, self.heads, self.head_dim)
        values = self.values_map(x).view(batch_size, seq_length, self.heads, self.head_dim)

        # We firs need to transpose the tensor to have shape (batch_size, heads, seq_length, head_dim) and then
        # reshape it to (batch_size * heads, seq_length, head_dim).
        # See the PyTorch implementation of nn.MultiheadAttention for a more efficient implementation.
        keys = keys.transpose(1, 2).reshape(batch_size * self.heads, seq_length, self.head_dim)
        queries = queries.transpose(1, 2).reshape(batch_size * self.heads, seq_length, self.head_dim)
        values = values.transpose(1, 2).reshape(batch_size * self.heads, seq_length, self.head_dim)

        attention_out = self._self_attention(queries, keys, values, mask)
        attention_out = attention_out.transpose(1, 2).reshape(batch_size, seq_length, self.embedding_size)

        # Output mapping resulting in a tensor of shape (batch_size, seq_length, embedding_size)
        output = self.output_map(attention_out)

        return output
