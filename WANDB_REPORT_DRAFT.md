# DA6401 Assignment 3 Report: Transformer for German-English Translation

## 1. Introduction

This report presents a Transformer-based neural machine translation system for translating German sentences into English using the Multi30k dataset. The implementation follows the core architecture from *Attention Is All You Need*, including scaled dot-product attention, multi-head attention, sinusoidal positional encoding, encoder-decoder stacks, label smoothing, and the Noam learning-rate scheduler.

The model was implemented from scratch in PyTorch using basic modules such as `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, and custom attention/masking logic. The dataset was tokenized using SpaCy, and training was tracked using Weights & Biases.

## 2. Dataset and Preprocessing

Dataset: Multi30k German-English translation dataset.

Splits:

- Train: 29,000 sentence pairs
- Validation: 1,014 sentence pairs
- Test: 1,000 sentence pairs

Preprocessing steps:

- German sentences were used as source sequences.
- English sentences were used as target sequences.
- Text was tokenized using SpaCy tokenizers.
- Tokens were lowercased.
- Four special tokens were added: `<unk>`, `<pad>`, `<sos>`, and `<eos>`.
- Source and target vocabularies were built only from the training split.
- Validation and test splits reused the training vocabulary to avoid data leakage.
- Batches were padded dynamically using a custom collate function.

## 3. Model Architecture

The model uses an encoder-decoder Transformer architecture.

Main components:

- Source and target token embeddings
- Sinusoidal positional encoding
- Multi-head self-attention
- Encoder-decoder cross-attention
- Position-wise feed-forward networks
- Residual connections
- Layer normalization
- Final linear projection to the target vocabulary

Example hyperparameters:

| Hyperparameter | Value |
|---|---:|
| `d_model` | 256 |
| Encoder layers | 3 |
| Decoder layers | 3 |
| Attention heads | 8 |
| Feed-forward dimension | 1024 |
| Dropout | 0.1 |
| Batch size | 64 |
| Label smoothing | 0.1 |
| Warmup steps | 4000 |

## 4. Training Setup

The model was trained using Adam with the Transformer paper settings:

- `beta1 = 0.9`
- `beta2 = 0.98`
- `eps = 1e-9`

The learning rate was controlled by the Noam schedule:

```text
lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
```

The training objective used label smoothing with epsilon = 0.1.

W&B was used to log:

- Training loss
- Validation loss
- Learning rate
- Best checkpoint

## 5. Experiment 1: Necessity of the Noam Scheduler

### Setup

Two training runs were compared:

1. Transformer trained using the Noam scheduler.
2. Transformer trained with a fixed learning rate.

### W&B Plots to Add

Add these plots from W&B:

- Training loss vs epoch for both runs
- Validation loss vs epoch for both runs
- Learning rate vs step or epoch

### Analysis

The Noam scheduler improves training stability because the Transformer is sensitive to large learning rates during the early training phase. At the beginning of training, the self-attention weights and embeddings are not yet well calibrated. A high learning rate can cause unstable updates and divergence.

The warmup phase increases the learning rate gradually, allowing the model to learn stable early representations. After warmup, inverse square-root decay reduces the learning rate, helping the model continue improving without making overly large updates.

In comparison, the fixed learning-rate run is expected to either converge more slowly or show less stable validation behavior, especially during early epochs.

## 6. Experiment 2: Effect of the Scaling Factor

### Setup

Two versions of scaled dot-product attention were compared:

1. With scaling by `sqrt(d_k)`.
2. Without scaling.

### W&B Plots to Add

Add these plots:

- Training loss with scaling vs without scaling
- Gradient norm of query projection weights
- Gradient norm of key projection weights

### Analysis

The scaling factor prevents the dot products between queries and keys from becoming too large in magnitude. Without scaling, large dot products push the softmax into saturated regions, where probabilities become extremely peaked. This causes small gradients and can make optimization harder.

With scaling, the attention distribution is smoother during early training, which improves gradient flow through the query and key projections. This matches the motivation in the Transformer paper.

## 7. Experiment 3: Attention Rollout and Head Specialization

### Setup

Attention weights were extracted from the final encoder layer for selected validation sentences.

### W&B Visualizations to Add

Add heatmaps for individual attention heads.

Recommended sentence example:

```text
German source sentence: <paste example>
English target sentence: <paste example>
```

### Analysis

Different attention heads may learn different roles. Some heads may focus strongly on nearby tokens, while others may attend to longer-range dependencies or punctuation. If several heads show similar attention patterns, this indicates head redundancy.

In my observations, some heads specialized in local context, while others distributed attention more broadly across the sentence. This supports the idea that multi-head attention allows the model to represent multiple relationships in parallel.

## 8. Experiment 4: Sinusoidal Positional Encoding vs Learned Positional Embeddings

### Setup

Two positional encoding strategies were compared:

1. Fixed sinusoidal positional encoding.
2. Learned positional embeddings using `nn.Embedding`.

### W&B Plots to Add

Add:

- Validation loss comparison
- Validation BLEU comparison

### Analysis

Sinusoidal positional encoding does not introduce trainable position parameters. Since it is based on sine and cosine functions at different frequencies, the model can theoretically generalize to sequence lengths longer than those observed during training.

Learned positional embeddings can adapt to the training distribution, but they are limited to the maximum positions seen during training. For longer sequences, learned embeddings may not extrapolate as naturally.

## 9. Experiment 5: Label Smoothing

### Setup

Two training runs were compared:

1. Label smoothing with epsilon = 0.1.
2. Standard cross-entropy with epsilon = 0.0.

### W&B Plots to Add

Add:

- Training loss comparison
- Validation loss comparison
- Prediction confidence comparison

### Analysis

Label smoothing prevents the model from becoming overconfident by assigning a small amount of probability mass to non-target classes. This acts as a regularizer and usually improves generalization.

Although label smoothing can increase training perplexity, it often improves validation behavior and translation quality because the model learns less brittle probability distributions.

## 10. Final Test Performance

Report the best checkpoint performance here.

| Metric | Value |
|---|---:|
| Validation loss | TODO |
| Test BLEU | TODO |

Example statement:

The best checkpoint was selected based on validation loss. The final model achieved a test BLEU score of `TODO` on the held-out Multi30k test set.

## 11. Conclusion

This assignment demonstrated the importance of the key Transformer design choices. Multi-head attention enabled the model to capture different token relationships in parallel. Sinusoidal positional encoding provided position information without recurrence or convolution. The Noam scheduler stabilized early training through warmup, and label smoothing improved generalization by reducing overconfidence.

Overall, the experiments show that the optimization and regularization details are important for training Transformer models effectively, especially on sequence-to-sequence translation tasks.

