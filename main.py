import os
import sys
from timeit import default_timer as timer
from typing import Iterable, List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k, multi30k
from torchtext.vocab import build_vocab_from_iterator

from src.models import Transformer
from src.utils import set_seed

set_seed(0)

# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL[
    "train"
] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL[
    "valid"
] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"

# Place-holders
token_transform = {}
vocab_transform = {}


# Create source and target language tokenizer. Make sure to install the dependencies.
token_transform[SRC_LANGUAGE] = get_tokenizer("spacy", language="de_core_news_sm")
token_transform[TGT_LANGUAGE] = get_tokenizer("spacy", language="en_core_web_sm")


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(
        yield_tokens(train_iter, ln), min_freq=1, specials=special_symbols, special_first=True
    )

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)


def generate_square_subsequent_mask(sz):
    # sz = 8
    mask = torch.tril(torch.ones(1, 1, sz, sz, device=DEVICE, dtype=torch.bool))
    return mask


def create_mask(src, tgt):
    tgt_seq_len = tgt.shape[1]
    batch_size = src.shape[0]

    tgt_atttention_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_padding_mask = (src != PAD_IDX).view(batch_size, 1, -1).to(DEVICE)
    tgt_padding_mask = (tgt != PAD_IDX).view(batch_size, 1, -1).to(DEVICE)

    return src_padding_mask, tgt_padding_mask, tgt_atttention_mask


SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 128
NHEAD = 4
FFN_HID_DIM = 128
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DEVICE = "mps"

transformer = Transformer(
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMB_SIZE, NHEAD, FFN_HID_DIM, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS
)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))


# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(
        token_transform[ln], vocab_transform[ln], tensor_transform  # Tokenization  # Numericalization
    )  # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=True)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # tgt_input = tgt[:, :-1]
        # Shift
        tgt_input = tgt.clone()
        tgt_input[tgt_input == EOS_IDX] = PAD_IDX

        src_padding_mask, tgt_padding_mask, tgt_attention_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask, tgt_attention_mask)

        optimizer.zero_grad()

        # tgt_out = tgt[:, 1:]
        # Shift
        tgt_out = torch.roll(tgt, shifts=-1, dims=1)
        tgt_out[:, -1] = PAD_IDX

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split="valid", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # tgt_input = tgt[:, :-1]
        tgt_input = tgt.clone()
        tgt_input[tgt_input == EOS_IDX] = PAD_IDX

        src_padding_mask, tgt_padding_mask, tgt_attention_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask, tgt_attention_mask)

        # tgt_out = tgt[:, 1:]
        tgt_out = torch.roll(tgt, shifts=-1, dims=1)
        tgt_out[:, -1] = PAD_IDX

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


NUM_EPOCHS = 10

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print(
        (
            f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"
        )
    )


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, max_len, start_symbol):
    src = src.to(DEVICE)
    src_padding_mask = (src != PAD_IDX).view(1, 1, -1).to(DEVICE)

    memory = model.encode(src, src_padding_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    # ys = nn.ConstantPad1d((0, src.shape[1] - ys.shape[0]), PAD_IDX)(ys)

    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        # tgt_mask = generate_square_subsequent_mask(ys.size(1))
        _, trg_padding_mask, trg_attention_mask = create_mask(src, ys)
        out = model.decode(ys, memory, trg_padding_mask, src_padding_mask, trg_attention_mask)
        # out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    # src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    src = text_transform[SRC_LANGUAGE](src_sentence).view(1, -1)
    num_tokens = src.shape[1]
    tgt_tokens = greedy_decode(model, src, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return (
        " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy())))
        .replace("<bos>", "")
        .replace("<eos>", "")
    )


print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))
