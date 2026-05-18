# DA6401 Assignment 3 - Transformer for German to English Translation

This project implements a Transformer sequence-to-sequence model from scratch in PyTorch for the Multi30k German to English translation task.

## Files

- `dataset.py`: Multi30k loading, SpaCy tokenization, vocab creation, and padding collate function.
- `model.py`: Scaled dot-product attention, multi-head attention, masks, positional encoding, encoder, decoder, and full Transformer.
- `lr_scheduler.py`: Noam learning-rate scheduler.
- `train.py`: Label smoothing, epoch loop, greedy decoding, BLEU evaluation, checkpoints, and a basic training entry point.

## Run

```bash
pip install -r requirements.txt
python train.py
```

Build vocabularies only from the training split, then reuse them for validation and test splits.
