import torch
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module):
    r"""
    A simple embedding layer that maps each token in the input sequence to a dense vector. The final output is
    multiplied by the square root of the embedding size as in the transfomer paper.

    Args:
        - vocab_size (int): The size of the vocabulary, i.e., the number of distinct tokens in the input.
        - embedding_size (int): The size of the embedding for each token.
    """

    def __init__(self, vocab_size: int, embedding_size: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size

    def forward(self, x: Tensor) -> Tensor:
        embeds = self.embeddings(x)
        output = embeds * (self.embedding_size**0.5)
        return output


class PositionalEncoding(nn.Module):
    r"""
    The `PositionalEncoding` class implements the Positional Encoding layer, which is used in the Transformer
    architecture to provide the model with information about the position of each token in the input sequence.
    The layer adds a sinusoidal signal to the input embeddings to encode the position information. The positional
    encoding is a fixed, non-learnable parameter of the model, registered as a buffer, and can be saved and loaded
    using the state_dict.

    Args:
        - embedding_size (int): The size of the input embeddings.
        - dropout (float): The dropout probability to use after the positional encoding is added to the input
            (default: 0.1).
        - max_seq_len (int): The maximum sequence length that the model can handle (default: 5000).
    """

    def __init__(self, embedding_size: int, dropout: float = 0.1, max_seq_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Check the formula in the original paper!
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        # The `10000`is probably to assure a huge period and cover long senquences
        frequences = torch.pow(10000, torch.arange(0, embedding_size, 2) / embedding_size)

        # Sine wave for even indices and cosine wave for odd indices
        pos_encoding = torch.zeros(max_seq_len, embedding_size)
        pos_encoding[:, 0::2] = torch.sin(pos / frequences)
        pos_encoding[:, 1::2] = torch.cos(pos / frequences)

        # Register to buffer non-learnable parameters so that can be saved and loaded in the state_dict
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, embedding: Tensor) -> Tensor:
        # Just get the positional encoding for the current sequence size
        pos_encoding = self.pos_encoding[: embedding.shape[1]]
        # Dropout after adding positional encoding and input embedding as explained in the paper
        new_embedding = self.dropout(embedding + pos_encoding)
        return new_embedding
