import torch
import torch.nn as nn
from torch import Tensor

from .modules import Embedding, MultiheadAttention, PositionalEncoding


class Transformer(nn.Module):
    """
    Transformer paper: https://arxiv.org/abs/1706.03762

    Transformer architecture for sequence-to-sequence tasks. It takes as input a source sequence and a target sequence,
    and returns the predicted target sequence. During the forward pass, the input sequence is passed through an input
    embedding layer and a positional encoding layer. The output sequence is also passed through an output embedding
    layer and a positional encoding layer. The resulting input embedding is then passed through a stack of encoder
    layers, and the resulting output is used as input to the decoder, which consists of a stack of decoder layers.
    Finally, the decoder output is returned as the output of the model.

    Args:
        - vocab_size (int): The size of the vocabulary.
        - embedding_size (int): The size of the embedding for each token.
        - heads (int): The number of attention heads.
        - forward_expansion (int): The expansion factor for the feed-forward layer.
        - n_encoders (int): The number of encoder layers.
        - n_decoders (int): The number of decoder layers.
        - dropout (float, optional): The dropout probability (Default: 0.1)
        - bias (bool, optional): Whether to include bias terms in the attention layers (Default: False).

    Example:
        >>> model = Transformer(src_vocab_size=1000,
                                trt_vocab_size=1000,
                                embedding_size=256,
                                heads=8,
                                forward_expansion=4,
                                enc_layers=2,
                                dec_layers=2,
                                dropout=0.1,
                                bias=False,
    """

    def __init__(
        self,
        src_vocab_size: int,
        trt_vocab_size: int,
        embedding_size: int,
        heads: int,
        forward_expansion: int,
        n_encoders: int,
        n_decoders: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.input_embedding = Embedding(src_vocab_size, embedding_size)
        self.output_embedding = Embedding(trt_vocab_size, embedding_size)

        self.input_pos_embedding = PositionalEncoding(embedding_size, dropout)
        self.output_pos_embedding = PositionalEncoding(embedding_size, dropout)

        enc_layer = EncoderLayer(heads, embedding_size, forward_expansion, dropout, bias)
        self.encoder = Encoder(enc_layer, n_encoders, norm=True)
        dec_layer = DecoderLayer(heads, embedding_size, forward_expansion, dropout, bias)
        self.decoder = Decoder(dec_layer, n_decoders, norm=True)

        self.generator = nn.Linear(embedding_size, trt_vocab_size)

    def encode(self, src_seq: Tensor, src_pad_mask: Tensor = None) -> Tensor:
        src_emb = self.input_embedding(src_seq)
        src_emb = self.input_pos_embedding(src_emb)
        return self.encoder(src_emb, src_pad_mask)

    def decode(
        self,
        trg_seq: Tensor,
        memory: Tensor,
        trg_pad_mask: Tensor = None,
        src_pad_mask: Tensor = None,
        trg_att_mask: Tensor = None,
    ) -> Tensor:
        trg_emb = self.output_embedding(trg_seq)
        trg_emb = self.output_pos_embedding(trg_emb)
        return self.decoder(trg_emb, memory, trg_pad_mask, src_pad_mask, trg_att_mask)

    def forward(
        self,
        src_seq: Tensor,
        trg_seq: Tensor,
        src_pad_mask: Tensor = None,
        trg_pad_mask: Tensor = None,
        trg_att_mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            - src_seq (Tensor): The source sequence.
            - trg_seq (Tensor): The target sequence.
            - src_pad_mask (Tensor, optional): The source padding mask.
            - trg_pad_mask (Tensor, optional): The target padding mask.
            - trg_att_mask (Tensor, optional): The target attention mask.

        Shape:
            - src_seq (Tensor): (batch_size, src_len)
            - trg_seq (Tensor): (batch_size, trg_len)
            - src_pad_mask (Tensor): (batch_size, 1, src_len)
            - trg_pad_mask (Tensor): (batch_size, 1, trg_len)
            - trg_att_mask (Tensor): (1, 1, trg_len, trg_len)

        where :math:`src_len` is the length of the source sequence and :math:`trg_len` is the length of the target
        sequence.

        Returns:
            - Tensor: The predicted target distribution.
        """
        memory = self.encode(src_seq, src_pad_mask)
        out_dec = self.decode(trg_seq, memory, trg_pad_mask, src_pad_mask, trg_att_mask)

        return self.generator(out_dec)


class Encoder(nn.Module):
    r"""
    Transformer encoder that consists of a stack of N encoder layers.

    Args:
        - n_layers (int, optional): Number of encoder layers.
        - encoder_layer (nn.Module): An instance of EncoderLayer.
        - norm (bool, optional): If True, applies layer normalization to the output of the encoder.

    Example:
        >>> encoder = Encoder(n_layers=6, encoder_layer=EncoderLayer())
        >>> input_emb = torch.randn(8, 10, 512)
        >>> output = encoder(input_emb)
    """

    def __init__(self, encoder_layer: nn.Module, n_layers: int = 6, norm: bool = None) -> None:
        super().__init__()

        self.enc_layers = nn.ModuleList([encoder_layer for _ in range(n_layers)])
        self.norm = norm
        if norm:
            self.norm = nn.LayerNorm(encoder_layer.embedding_size)

    def forward(self, in_emb: Tensor, pad_mask: Tensor = None) -> Tensor:
        """
        Args:
            - in_emb (Tensor): The input embedding.
            - pad_mask (Tensor, optional): The key/query padding mask.

        Shape:
            - in_emb (Tensor): (batch_size, src_len, embedding_size)
            - pad_mask (Tensor): (batch_size, 1, src_len)

        where :math:`src_len` is the length of the source sequence.

        Returns:
            - output (Tensor): The output of the encoder.
        """
        # Iterate through the encoder layers
        for layer in self.enc_layers:
            in_emb = layer(in_emb, pad_mask)
        output = in_emb
        # Not in original paper but check PyTorch implementation
        if self.norm is not None:
            output = self.norm(output)
        return output


class Decoder(nn.Module):
    r"""
    Transformer decoder that consists of a stack of N decoder layers.

    Args:
        - n_layers (int, optional): Number of encoder layers.
        - encoder_layer (nn.Module): An instance of EncoderLayer.
        - norm (bool, optional): If True, applies layer normalization to the output of the encoder.

    Example:
        >>> encoder = Encoder(n_layers=6, encoder_layer=EncoderLayer())
        >>> input_emb = torch.randn(8, 10, 512)
        >>> output = encoder(input_emb)
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        n_layers: int = 6,
        norm: bool = None,
    ) -> None:
        super().__init__()

        self.dec_layers = nn.ModuleList([decoder_layer for _ in range(n_layers)])
        self.norm = norm
        if self.norm is not None:
            self.norm = nn.LayerNorm(decoder_layer.embedding_size)

    def forward(
        self,
        trg_emb: Tensor,
        memory: Tensor,
        trg_query_pad_mask: Tensor = None,
        memory_key_pad_mask: Tensor = None,
        trg_att_mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            - trg_emb (Tensor): The target embeddings.
            - memory (Tensor): The output of the encoder.
            - trg_query_pad_mask (Tensor, optional): The query padding mask for the target embeddings (default: None).
            - memory_key_pad_mask (Tensor, optional): The key padding mask for the memory (default: None).
            - trg_att_mask (Tensor, optional): The attention mask for the target embeddings not to look at subsequent
                words (default: None).

        Shape:
            - trg_emb: :math:`(batch_size, trg_len, embedding_size)`
            - memory: :math:`(batch_size, src_len, embedding_size)`
            - trg_query_pad_mask: :math:`(batch_size, 1, trg_len)`
            - memory_key_pad_mask: :math:`(batch_size, 1, src_len)`
            - trg_att_mask: :math:`(1, 1, trg_len, trg_len)`
            - output: :math:`(batch_size, trg_len, embedding_size)`

        where :math:`trg_len` is the target sequence length, :math:`src_len` is the source sequence length and the '1's
        in the masks shape are added to broadcast along the different dimensions.

        Returns:
            Tensor: The output of the decoder.
        """
        # Iterate through the encoder layers
        for layer in self.dec_layers:
            out_emb = layer(trg_emb, memory, trg_query_pad_mask, memory_key_pad_mask, trg_att_mask)
        output = out_emb
        # Not in original paper but check PyTorch implementation
        if self.norm is not None:
            output = self.norm(output)
        return output


class EncoderLayer(nn.Module):
    """
    The Encoder layer takes the input embeddings and applies self-attention mechanism followed
    by a feed-forward network. It also performs normalization and dropout on the output of each
    sublayer. It allows for masking of the input sequence to prevent attention to subsequent words.

    Args:
        - heads (int, optional): The number of heads in the multihead attention layer (default: 8).
        - embedding_size (int, optional): The size of input embeddings and hidden layer (default: 512).
        - forward_expansion (int, optional): The expansion factor for the feed-forward network (default: 4).
        - dropout (float, optional): The dropout probability (default: 0.1).
        - bias (bool, optional): Whether to use bias in the layers (default: False).

    Example:
    >>> encoder_layer = EncoderLayer(heads=16, embedding_size=256, forward_expansion=2, dropout=0.2, bias=True)
    >>> input_emb = torch.randn(8, 10, 256)
    >>> output = encoder_layer(input_emb)
    """

    def __init__(
        self,
        heads: int = 8,
        embedding_size: int = 512,
        forward_expansion: int = 4,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_size = embedding_size

        self.dropout = nn.Dropout(dropout)

        self.attention = MultiheadAttention(heads, embedding_size, dropout, bias)
        self.norm1 = nn.LayerNorm(embedding_size)

        # 2-layer feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size),
        )
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, src_emb: Tensor, pad_mask: Tensor = None) -> Tensor:
        """
        Args:
            - src_emb (Tensor): The source embeddings.
            - pad_mask (Tensor, optional): The query/key padding mask for the source embeddings (default: None).

        Shape:
            - src_emb: :math:`(batch_size, src_len, embedding_size)`
            - pad_mask: :math:`(batch_size, 1, src_len)`
            - output: :math:`(batch_size, src_len, embedding_size)`

        where :math:`src_len` is the source sequence length and the '1's in the masks shape are added to broadcast
        along the different dimensions.

        Returns:
            Tensor: The output of the encoder layer.
        """
        # The query, key, and value come all from the input embedding for self-attention
        attention_out1, _ = self.attention(src_emb, src_emb, src_emb, query_pad_mask=pad_mask, key_pad_mask=pad_mask)
        attention_out1 = self.norm1(self.dropout(attention_out1)) + src_emb

        feed_forward_out = self.feed_forward(attention_out1)
        # Dropout of the feed-forward output is applied before the normalization
        feed_forward_out = self.norm2(self.dropout(feed_forward_out)) + attention_out1

        return feed_forward_out


class DecoderLayer(nn.Module):
    r"""
    The Decoder layer takes the output embeddings and applies self-attention mechanism followed by cross-attention with
    the output of the encoder layer followed finally by a feed-forward network. It also performs normalization and
    dropout on the output of each sublayer. It allows for masking of the input and output sequences to prevent
    attention to subsequent words. The standard masking is applied to the self attention of the output embeddings.

    Args:
        - heads (int, optional): The number of attention heads in the multi-head attention mechanism (default: 8).
        - embedding_size (int, optional): The dimensionality of the input and output embeddings (default: 512).
        - forward_expansion (int, optional): The scaling factor applied to the intermediate layer size in the
            feed-forward network (default: 4).
        - dropout (float, optional): The dropout probability (default: 0.1).
        - bias (bool, optional): If True, adds a learnable bias to the output (default: False).

    Example:
    >>> decoder_layer = DecoderLayer(heads=8, embedding_size=512, forward_expansion=4, dropout=0.1, bias=False)
    >>> mask = mask = torch.tril(torch.ones(10, 10))
    >>> output_emb = torch.randn(8, 10, 256)
    >>> enc_output = torch.randn(8, 10, 256)
    >>> output = decoder_layer(x, enc_output, emb_mask=mask)
    """

    def __init__(
        self,
        heads: int = 8,
        embedding_size: int = 512,
        forward_expansion: int = 4,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding_size = embedding_size

        self.attention = MultiheadAttention(heads, embedding_size, dropout, bias)
        self.norm1 = nn.LayerNorm(embedding_size)

        # Cross-attention between the output embeddings and the output of the encoder layer
        self.cross_attention = MultiheadAttention(heads, embedding_size, dropout, bias)
        self.norm2 = nn.LayerNorm(embedding_size)

        # 2-layer feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size),
        )
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(
        self,
        trg_emb: Tensor,
        memory: Tensor,
        trg_query_pad_mask: Tensor = None,
        memory_key_pad_mask: Tensor = None,
        trg_att_mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            - trg_emb (Tensor): The target embeddings.
            - memory (Tensor): The output of the encoder layer.
            - trg_query_pad_mask (Tensor, optional): The query padding mask for the target embeddings (default: None).
            - memory_key_pad_mask (Tensor, optional): The key padding mask for the output of the encoder layer
                (default: None)
            - trg_att_mask (Tensor, optional): The attention mask for the target embeddings not to look at subsequent
                words (default: None).

        Shape:
            - trg_emb: :math:`(batch_size, trg_len, embedding_size)`
            - memory: :math:`(batch_size, src_len, embedding_size)`
            - trg_query_pad_mask: :math:`(batch_size, 1, trg_len)`
            - memory_key_pad_mask: :math:`(batch_size, 1, src_len)`
            - trg_att_mask: :math:`(batch_size, trg_len, trg_len)`
            - output: :math:`(batch_size, trg_len, embedding_size)`

        where :math:`src_len` is the source sequence length, :math:`trg_len` is the target sequence length and the '1's
        in the masks shape are added to broadcast along the different dimensions.

        Returns:
            Tensor: The output of the decoder layer.
        """
        # If necessary, we can pass both a mask both for the output embeddings and the output of the encoder layer
        # The query, key, and value come all from the output embedding for self-attention
        attention_out1, _ = self.attention(
            trg_emb, trg_emb, trg_emb, trg_query_pad_mask, trg_query_pad_mask, trg_att_mask
        )
        attention_out1 = self.norm1(self.dropout(attention_out1)) + trg_emb

        # For cross-attention, the query comes from the output embedding while the key and value come from
        # the output of the encoder layer.
        attention_out2, _ = self.cross_attention(
            attention_out1, memory, memory, trg_query_pad_mask, memory_key_pad_mask
        )
        attention_out2 = self.norm2(self.dropout(attention_out2)) + attention_out1

        feed_forward_out = self.feed_forward(attention_out2)
        # Dropout of the feed-forward output is applied before the normalization
        feed_forward_out = self.norm3(self.dropout(feed_forward_out)) + attention_out2

        return feed_forward_out


if __name__ == "__main__":
    src_data = torch.randn(16, 20, 512)
    trg_data = torch.randn(16, 30, 512)
    src_mask = torch.ones((16, 1, 20), dtype=torch.bool)
    src_mask[:, :, 10:] = 0

    trg_key_pad_mask = torch.ones((16, 1, 30), dtype=torch.bool)
    trg_key_pad_mask[:, :, 20:] = 0

    trg_attention_mask = torch.tril(torch.ones((1, 1, 30, 30), dtype=torch.bool))

    enc = EncoderLayer()
    enc_out = enc(src_data, src_mask)
    print(enc_out.shape)

    dec = DecoderLayer()
    dec_out = dec(
        trg_data,
        enc_out,
        trg_query_pad_mask=trg_key_pad_mask,
        memory_key_pad_mask=src_mask,
        trg_att_mask=trg_attention_mask,
    )
    print(dec_out.shape)
