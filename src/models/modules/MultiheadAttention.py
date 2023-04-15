import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class MultiheadAttention(nn.Module):
    r"""
    Transformer paper: https://arxiv.org/abs/1706.03762

    Multi-head self-attention module that maps input sequences to output sequences using a weighted sum of
    key-value pairs, where the weight assigned to each pair is determined by a compatibility function between
    the input and the key. This module supports dropout and masking for use in sequence-to-sequence models. It also
    supports different shape between the query and key/value tensors, allowing for rectangular attention and
    padding/attention masks (check `forward`and `_self_attention`methods).

    Args:
        - heads (int): The number of parallel self-attention heads to use.
        - embedding_size (int): The size of the input embedding.
        - dropout (float): The dropout probability to use after the attention weights are computed.
        - bias (bool): Whether to include a bias term in the key, query, value, and output linear transformations.

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

    def _self_attention(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        query_pad_mask: Tensor = None,
        key_pad_mask: Tensor = None,
        att_mask: Tensor = None,
    ) -> Tensor:
        """
        Compute the scaled self-attention for a set of queries, keys, and values. To use it as an
        auto-regressive model, it supports masking the attention weights (i.e. not looking forward). It also support
        different shape between the query and key/value tensors, allowing for rectangular attention and
        padding/attention masks.

        Shape:
            - queries: :math:`(batch_size, query_length, embedding_size)`.
            - keys: :math:`(batch_size, key_length, embedding_size)`.
            - values: :math:`(batch_size, value_length, embedding_size)`.
            - query_pad_mask: :math:`(batch_size, 1, query_length)`.
            - key_pad_mask: :math:`(batch_size, 1, key_length)`.
            - att_mask: :math:`(batch_size, 1, query_length, key_length)`.
            - output: :math:`((batch_size, query_length, embedding_size),
                (batch_size, heads, query_length, key_length))`.

        where :math:`query_length == value_length` but can be different from :math:`key_lenght`.

        Returns:
            - The output tensor after applying the self-attention mechanism and the attention weights.
            - The attention weights.

        """

        dot_product = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim**0.5)

        if query_pad_mask is not None:  # Shape: (batch_size, 1, query_length, 1)
            dot_product = dot_product.masked_fill(query_pad_mask.unsqueeze(-1) == torch.tensor(False), -1e9)

        if key_pad_mask is not None:  # Shape: (batch_size, 1, 1, key_length)
            dot_product = dot_product.masked_fill(key_pad_mask.unsqueeze(1) == torch.tensor(False), -1e9)

        if att_mask is not None:  # Shape: (batch_size, 1, query_length, key_length)
            dot_product = dot_product.masked_fill(att_mask == torch.tensor(False), -1e9)

        attention_weights = self.softmax(dot_product)
        attention_weights = self.dropout(attention_weights)
        attention_out = torch.matmul(attention_weights, values)
        return attention_out, attention_weights

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pad_mask: Tensor = None,
        key_pad_mask: Tensor = None,
        att_mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            - query (Tensor): The tensor representing the quert.
            - key (Tensor): The tensor representing the key.
            - value (Tensor): The tensor representing the value.
            - query_pad_mask (Tensorm optional): An optional padding mask applied to the queries (default: None).
            - key_pad_mask (Tensor): An optional padding mask applied to the keys (default: None).
            - att_mask (Tensor): An optional mask applied to the attention dot product. If not None, it should be a
                lower triangular matrix of ones (diagonal ingluded) and zeros in the upper half. It supports for
                rectangular attention masks (default: None).

        Shape:
            - queries: :math:`(batch_size, query_length, embedding_size)`.
            - keys: :math:`(batch_size, key_length, embedding_size)`.
            - values: :math:`(batch_size, value_length, embedding_size)`.
            - query_pad_mask: :math:`(batch_size, 1, query_length)`.
            - key_pad_mask: :math:`(batch_size, 1, key_length)`.
            - att_mask: :math:`(batch_size, 1, query_length, key_length)`.
            - output: :math:`((batch_size, query_length, embedding_size),
                (batch_size, heads, query_length, key_length))`.

        where :math:`query_length == value_length` but can be different from :math:`key_lenght`.

        Returns:
            - The output tensor after applying the self-attention mechanism and the attention weights.
            - The attention weights.
        """
        # The input tensor query has shape (batch_size, seq_length, embedding_size). Query, key, value can be from the
        # same input (standard self-attention) or differnt inputs (encoder-decoder cross-attention).
        batch_size, query_length, _ = query.size()
        _, key_length, _ = key.size()

        # Tet the keys, queries and values from the input tensor. Reshape from (batch_size, seq_length, embedding_size)
        # to (batch_size, seq_length, heads, head_dim), considering that embedding_size = heads * head_dim.
        queries = self.queries_map(query).view(batch_size, query_length, self.heads, self.head_dim)
        keys = self.keys_map(key).view(batch_size, key_length, self.heads, self.head_dim)
        values = self.values_map(value).view(batch_size, key_length, self.heads, self.head_dim)

        # We need to transpose the tensor to have shape (batch_size, heads, seq_length, head_dim).
        # See the PyTorch implementation of nn.MultiheadAttention for a more efficient implementation.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_out, attention_weights = self._self_attention(
            queries, keys, values, query_pad_mask, key_pad_mask, att_mask
        )
        attention_out = attention_out.transpose(1, 2).reshape(batch_size, query_length, self.embedding_size)

        # Output mapping resulting in a tensor of shape (batch_size, seq_length, embedding_size)
        output = self.output_map(attention_out)

        return output, attention_weights


if __name__ == "__main__":
    query = torch.randn(16, 40, 512)
    key = torch.randn(16, 30, 512)
    value = torch.randn(16, 30, 512)

    query_pad_mask = torch.ones((16, 1, 40), dtype=torch.bool)
    query_pad_mask[:, :, 30:] = 0

    key_pad_mask = torch.ones((16, 1, 30), dtype=torch.bool)
    key_pad_mask[:, :, 20:] = 0

    att_mask = torch.tril(torch.ones(1, 1, 40, 30))

    mha = MultiheadAttention(heads=4, embedding_size=512, dropout=0.0, bias=False)

    _, weights = mha(query, key, value, query_pad_mask, key_pad_mask, att_mask)
    # save weights to file as imae with matplotlib
    plt.imshow(np.log(weights[5, 0, :, :].detach().numpy()))
    plt.colorbar()
    plt.savefig("./attention_weights.png")
