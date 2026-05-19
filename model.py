import copy
import json
import math
import os
from typing import Optional, Tuple

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    batch, length = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(torch.ones(length, length, device=tgt.device, dtype=torch.bool), diagonal=1)
    return pad_mask | causal_mask.unsqueeze(0).unsqueeze(1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

    def _split(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Q = self._split(self.w_q(query))
        K = self._split(self.w_k(key))
        V = self._split(self.w_v(value))
        x, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        x = self.dropout(x).transpose(1, 2).contiguous()
        x = x.view(query.size(0), query.size(1), self.d_model)
        return self.w_o(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, src_mask)))
        return self.norm2(x + self.dropout2(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, memory, memory, src_mask)))
        return self.norm3(x + self.dropout3(self.ffn(x)))


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, memory, src_mask, tgt_mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    CHECKPOINT_URL = "https://drive.google.com/file/d/1f59ku11WPg5GJeWm9WoVUriuG8S89cWb/view?usp=drive_link"
    VOCAB_URL = "https://drive.google.com/file/d/1hTgI0G6hIYE0WWBrLaFb_W_aIEPiq2R_/view?usp=drive_link"
    DEFAULT_CHECKPOINT_PATH = "best_checkpoint.pt"
    DEFAULT_VOCAB_PATH = "vocab.json"

    def __init__(
        self,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str = None,
        vocab_path: str = None,
        checkpoint_url: str = None,
        vocab_url: str = None,
        auto_load: bool = True,
    ) -> None:
        super().__init__()
        checkpoint_path = checkpoint_path or self.DEFAULT_CHECKPOINT_PATH
        vocab_path = vocab_path or self.DEFAULT_VOCAB_PATH
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(base_dir, checkpoint_path)
        if not os.path.isabs(vocab_path):
            vocab_path = os.path.join(base_dir, vocab_path)

        if auto_load:
            self._download_if_needed(vocab_path, vocab_url or self.VOCAB_URL, kind="json")
            self._download_if_needed(checkpoint_path, checkpoint_url or self.CHECKPOINT_URL, kind="torch")

        self.src_itos, self.tgt_itos = self._load_vocab(vocab_path)
        self.src_stoi = {tok: i for i, tok in enumerate(self.src_itos)}
        self.tgt_stoi = {tok: i for i, tok in enumerate(self.tgt_itos)}
        self.de_tokenizer = spacy.blank("de").tokenizer

        ckpt = self._load_checkpoint_dict(checkpoint_path) if auto_load else None
        if ckpt and ckpt.get("model_config"):
            cfg = ckpt["model_config"]
            src_vocab_size = src_vocab_size or cfg.get("src_vocab_size")
            tgt_vocab_size = tgt_vocab_size or cfg.get("tgt_vocab_size")
            d_model = cfg.get("d_model", d_model)
            N = cfg.get("N", N)
            num_heads = cfg.get("num_heads", num_heads)
            d_ff = cfg.get("d_ff", d_ff)
            dropout = cfg.get("dropout", dropout)

        src_vocab_size = src_vocab_size or len(self.src_itos)
        tgt_vocab_size = tgt_vocab_size or len(self.tgt_itos)
        self.config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
        }
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        if ckpt and "model_state_dict" in ckpt:
            self.load_state_dict(ckpt["model_state_dict"])

    @staticmethod
    def _download_if_needed(path, url, kind="torch"):
        checker = Transformer._looks_like_json_file if kind == "json" else Transformer._looks_like_torch_file
        if os.path.exists(path) and checker(path):
            return
        if os.path.exists(path) and url:
            os.remove(path)
        if not url:
            return
        if gdown is None:
            raise ImportError("Install gdown or place the required artifact beside model.py.")
        try:
            gdown.download(url, path, quiet=False, fuzzy=True)
        except TypeError:
            gdown.download(url, path, quiet=False)

    @staticmethod
    def _looks_like_torch_file(path):
        try:
            with open(path, "rb") as f:
                head = f.read(8)
            return head.startswith(b"PK") or head.startswith(b"\x80")
        except OSError:
            return False

    @staticmethod
    def _looks_like_json_file(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            return "src_itos" in vocab and "tgt_itos" in vocab
        except (OSError, json.JSONDecodeError):
            return False

    @staticmethod
    def _load_checkpoint_dict(path):
        if not os.path.exists(path) or not Transformer._looks_like_torch_file(path):
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            return None

    @staticmethod
    def _load_vocab(path):
        specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
        if not os.path.exists(path):
            return specials, specials
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return vocab["src_itos"], vocab["tgt_itos"]

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        return self.encoder(self.positional_encoding(x), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        return self.generator(self.decoder(self.positional_encoding(x), memory, src_mask, tgt_mask))

    def forward(self, src, tgt, src_mask, tgt_mask) -> torch.Tensor:
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    @staticmethod
    def _detokenize(tokens):
        text = " ".join(tokens)
        for p in [".", ",", "!", "?", ":", ";", "%"]:
            text = text.replace(" " + p, p)
        text = text.replace("( ", "(").replace(" )", ")")
        text = text.replace(" n't", "n't").replace(" 's", "'s").replace(" 're", "'re")
        text = text.replace(" 'm", "'m").replace(" 've", "'ve").replace(" 'll", "'ll")
        return text

    @torch.no_grad()
    def infer(self, german_sentence: str, max_len: int = 100, beam_size: int = 8, length_penalty: float = 0.8) -> str:
        self.eval()
        device = next(self.parameters()).device
        tokens = [tok.text.lower() for tok in self.de_tokenizer(german_sentence)]
        src_ids = [2] + [self.src_stoi.get(tok, 0) for tok in tokens] + [3]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, pad_idx=1)

        memory = self.encode(src, src_mask)
        beams = [(torch.tensor([[2]], dtype=torch.long, device=device), 0.0)]

        for _ in range(max_len - 1):
            candidates = []
            finished = True
            for seq, score in beams:
                if int(seq[0, -1].item()) == 3:
                    candidates.append((seq, score))
                    continue
                finished = False
                logits = self.decode(memory, src_mask, seq, make_tgt_mask(seq, pad_idx=1))
                log_probs = F.log_softmax(logits[:, -1], dim=-1)
                seq_ids = seq[0].tolist()
                if len(seq_ids) >= 4:
                    seen_trigrams = {
                        tuple(seq_ids[i:i + 3])
                        for i in range(len(seq_ids) - 2)
                    }
                    prefix = tuple(seq_ids[-2:])
                    for token_id in range(log_probs.size(-1)):
                        if prefix + (token_id,) in seen_trigrams:
                            log_probs[0, token_id] = -1e9
                values, indices = torch.topk(log_probs, beam_size, dim=-1)
                for value, index in zip(values[0], indices[0]):
                    next_id = int(index.item())
                    next_seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
                    candidates.append((next_seq, score + float(value.item())))
            beams = sorted(
                candidates,
                key=lambda item: item[1] / (item[0].size(1) ** length_penalty),
                reverse=True,
            )[:beam_size]
            if finished:
                break

        ys = max(beams, key=lambda item: item[1] / (item[0].size(1) ** length_penalty))[0]

        words = []
        for idx in ys[0].tolist():
            if idx in {1, 2}:
                continue
            if idx == 3:
                break
            if idx < len(self.tgt_itos):
                words.append(self.tgt_itos[idx])
        return self._detokenize(words)
