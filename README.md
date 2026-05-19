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

Github link: https://github.com/gowtham-1707/Implementing_transformer_from_scratch_for_MLT
Wandb link: https://wandb.ai/ma24c051-iit-madras/MLT_german_to_english/reports/DA6401-Assignment-3-Report-Transformer-for-German-English-Translation--VmlldzoxNjkzNTY2OA?accessToken=15b72cajs4ehvwvtr9qcmgjmzl6kjti9t1llou36zl0jetdyb3yoj4bf1gsl2mxd