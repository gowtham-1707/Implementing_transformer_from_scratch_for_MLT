# DA6401 Assignment 3 - Transformer for Machine Translation

Clean PyTorch implementation of a Transformer for German to English translation on Multi30k.

## Files

- `dataset.py`: Multi30k loading, SpaCy tokenization, vocab, and padding.
- `model.py`: Attention, masks, positional encoding, encoder, decoder, Transformer.
- `lr_scheduler.py`: Noam learning-rate scheduler.
- `train.py`: Label smoothing, training loop, greedy decoding, BLEU, checkpoints.

## Run

```bash
pip install -r requirements.txt
python train.py
```
