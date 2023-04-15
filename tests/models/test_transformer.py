import pytest
import torch

from src.models.transformer import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    Transformer,
)


@pytest.fixture
def src_emb():
    return torch.randn(16, 20, 512)  # batch_size=16, sequence_length=20, embedding_size=512


@pytest.fixture
def trg_emb():
    return torch.randn(16, 30, 512)  # batch_size=16, sequence_length=30, embedding_size=512


@pytest.fixture
def src_pad_mask():
    mask = torch.ones((16, 1, 20), dtype=torch.bool)
    mask[:, :, 10:] = 0
    return mask


@pytest.fixture
def trg_key_pad_mask():
    mask = torch.ones((16, 1, 30), dtype=torch.bool)
    mask[:, :, 20:] = 0
    return mask


@pytest.fixture
def trg_att_mask():
    return torch.tril(torch.ones(30, 30)).repeat(16, 8, 1, 1)


@pytest.fixture
def encoder_layer():
    return EncoderLayer()


@pytest.fixture
def encoder(encoder_layer):
    return Encoder(encoder_layer)


@pytest.fixture
def decoder_layer():
    return DecoderLayer()


@pytest.fixture
def decoder(decoder_layer):
    return Decoder(decoder_layer)


def test_encoder_layer_shape(encoder_layer, src_emb, src_pad_mask):
    out = encoder_layer(src_emb, src_pad_mask)
    assert out.shape == (16, 20, 512)


def test_encoder_shape(encoder, src_emb, src_pad_mask):
    out = encoder(src_emb, src_pad_mask)
    assert out.shape == (16, 20, 512)


def test_decoder_layer_shape(decoder_layer, trg_emb, src_emb, trg_key_pad_mask, src_pad_mask, trg_att_mask):
    out = decoder_layer(trg_emb, src_emb, trg_key_pad_mask, src_pad_mask, trg_att_mask)
    assert out.shape == (16, 30, 512)


def test_decoder_shape(decoder, trg_emb, src_emb, trg_key_pad_mask, src_pad_mask, trg_att_mask):
    out = decoder(trg_emb, src_emb, trg_key_pad_mask, src_pad_mask, trg_att_mask)
    assert out.shape == (16, 30, 512)
