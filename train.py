from collections import Counter
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EOS, PAD, SOS, Multi30kDataset, collate_fn
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask
import wandb

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            true_dist[:, self.pad_idx] = 0
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[target == self.pad_idx] = 0
        denom = (target != self.pad_idx).sum().clamp_min(1)
        return -(true_dist * log_probs).sum() / denom


def run_epoch(data_iter,model: Transformer,loss_fn: nn.Module,optimizer: Optional[torch.optim.Optimizer],scheduler=None,epoch_num: int = 0,is_train: bool = True,device: str = "cpu",) -> float:
    
    model.train(is_train)
    total_loss, total_tokens = 0.0, 0

    for src, tgt in tqdm(data_iter, desc=f"{'train' if is_train else 'eval'} {epoch_num}"):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, tgt_in, make_src_mask(src, PAD), make_tgt_mask(tgt_in, PAD))
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        tokens = (tgt_out != PAD).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens

    return total_loss / max(1, total_tokens)


def greedy_decode(model: Transformer,src: torch.Tensor,src_mask: torch.Tensor,max_len: int,start_symbol: int,end_symbol: int = EOS,device: str = "cpu",) -> torch.Tensor:
    
    model.eval()
    src, src_mask = src.to(device), src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, make_tgt_mask(ys, PAD))
        next_word = out[:, -1].argmax(dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
        if next_word == end_symbol:
            break
    return ys


def _tokens(ids, vocab):
    words = []
    for idx in ids:
        idx = int(idx)
        if idx in {PAD, SOS}:
            continue
        if idx == EOS:
            break
        words.append(vocab.lookup_token(idx) if hasattr(vocab, "lookup_token") else vocab.itos[idx])
    return words


def _corpus_bleu(references, hypotheses, max_n=4):
    matches, totals = [0] * max_n, [0] * max_n
    ref_len = hyp_len = 0
    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n + 1):
            ref_counts = Counter(tuple(ref[i:i + n]) for i in range(max(0, len(ref) - n + 1)))
            hyp_counts = Counter(tuple(hyp[i:i + n]) for i in range(max(0, len(hyp) - n + 1)))
            matches[n - 1] += sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
            totals[n - 1] += max(0, len(hyp) - n + 1)
    precisions = [(matches[i] + 1) / (totals[i] + 1) for i in range(max_n)]
    bp = 1.0 if hyp_len > ref_len else torch.exp(torch.tensor(1 - ref_len / max(1, hyp_len))).item()
    logs = sum(torch.log(torch.tensor(p)) for p in precisions) / max_n
    return 100 * bp * torch.exp(logs).item()


@torch.no_grad()
def evaluate_bleu(model: Transformer,test_dataloader: DataLoader,tgt_vocab,device: str = "cpu",max_len: int = 100,) -> float:
    refs, hyps = [], []
    for src, tgt in test_dataloader:
        src, tgt = src.to(device), tgt.to(device)
        for i in range(src.size(0)):
            one_src = src[i:i + 1]
            pred = greedy_decode(model, one_src, make_src_mask(one_src, PAD), max_len, SOS, EOS, device)
            refs.append(_tokens(tgt[i].tolist(), tgt_vocab))
            hyps.append(_tokens(pred[0].tolist(), tgt_vocab))
    return _corpus_bleu(refs, hyps)


def save_checkpoint(model: Transformer,optimizer: torch.optim.Optimizer,scheduler,epoch: int,path: str = "checkpoint.pt",) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": getattr(model, "config", {}),}, path)


def save_vocab(src_vocab, tgt_vocab, path: str = "vocab.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"src_itos": src_vocab.itos, "tgt_itos": tgt_vocab.itos}, f, ensure_ascii=False)

def load_checkpoint(path: str,model: Transformer,optimizer: Optional[torch.optim.Optimizer] = None,scheduler=None,) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]


def run_training_experiment():
    
    config = {
        "batch_size": 64,"epochs": 10,"d_model": 256,"num_layers": 3,
        "num_heads": 8,"d_ff": 1024,"dropout": 0.1,"warmup_steps": 4000,
        "label_smoothing": 0.1,"learning_rate": 1.0,}

    wandb.init(
        project="MLT_german_to_english",name="transformer-multi30k-noam",config=config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = Multi30kDataset("train")
    val_dataset = Multi30kDataset("validation",train_dataset.src_vocab,train_dataset.tgt_vocab)
    save_vocab(train_dataset.src_vocab, train_dataset.tgt_vocab, "vocab.json")
    train_loader = DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset,batch_size=config["batch_size"],shuffle=False,collate_fn=collate_fn)

    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),tgt_vocab_size=len(train_dataset.tgt_vocab),d_model=config["d_model"],
        N=config["num_layers"],num_heads=config["num_heads"],d_ff=config["d_ff"],dropout=config["dropout"],auto_load=False,).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"],betas=(0.9, 0.98),eps=1e-9)

    scheduler = NoamScheduler(optimizer,d_model=config["d_model"],warmup_steps=config["warmup_steps"])
    
    loss_fn = LabelSmoothingLoss(vocab_size=len(train_dataset.tgt_vocab),pad_idx=PAD,smoothing=config["label_smoothing"])
    
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        train_loss = run_epoch(train_loader,model,loss_fn,optimizer,scheduler,epoch,is_train=True,device=device)

        val_loss = run_epoch(val_loader,model,loss_fn,optimizer=None,scheduler=None,epoch_num=epoch,is_train=False,device=device)
        
        lr = optimizer.param_groups[0]["lr"]
        
        wandb.log({"epoch": epoch,"train_loss": train_loss,"val_loss": val_loss,"learning_rate": lr,})
        print(f"Epoch {epoch + 1}/{config['epochs']} | "f"Train Loss: {train_loss:.4f} | "f"Val Loss: {val_loss:.4f} | "f"LR: {lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model,optimizer,scheduler,epoch,path="best_checkpoint.pt")
            save_vocab(train_dataset.src_vocab, train_dataset.tgt_vocab, "vocab.json")
            wandb.save("best_checkpoint.pt")
            wandb.save("vocab.json")
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
